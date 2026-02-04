#!/usr/bin/env python3
"""Profile execution graph for test-time compute analysis.

This script:
1. Starts the FastChat controller
2. Starts reward model and language model workers
3. Waits for workers to register
4. Runs beam search evaluation with execution tracing
5. Exports trace data for visualization
6. Cleans up all processes on exit

Usage:
    python src/scripts/profile_execution_graph.py --num_problems 5 --output_dir ./profiles

Output files:
    - execution_graph_{timestamp}.json: Structured data for analysis
    - chrome_trace_{timestamp}.json: Chrome trace for visualization
    - timeline_{timestamp}.csv: CSV for spreadsheet analysis
"""

import subprocess
import os
import sys
import time
import signal
import atexit
import json
import requests
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(src_path))

from transformers import AutoTokenizer

from reason.evaluation.methods import BeamSearchConfig, Task
from reason.evaluation.evaluator import TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RemoteRewardModelConfig,
    get_prm_special_tokens,
)
from reason.profiling.execution_tracer import ExecutionTracer, EventType
from reason.profiling.traced_callers import (
    TracedVLLMRemoteCaller,
    TracedRMRemoteCaller,
    wrap_existing_rm_call,
)
from reason.profiling.trace_export import (
    export_chrome_trace,
    export_structured_json,
    export_timeline_csv,
    print_timeline_ascii,
)
from envs import get_env_datasets


# === Default Configuration ===
DEFAULT_POLICY_MODEL_PATH = "/scratch/msa6093/models/Qwen2.5-1.5B-Instruct"
DEFAULT_VALUE_MODEL_PATH = "/scratch/msa6093/models/math-shepherd-mistral-7b-prm"
DEFAULT_HOST_ADDR = "0.0.0.0"
DEFAULT_CONTROLLER_PORT = 10014
DEFAULT_WORKER_BASE_PORT = 10081

# vLLM worker parameters
VLLM_CONFIG = {
    "gpu_memory_utilization": 0.88,
    "max_model_length": 8192,
    "swap_space": 16,
}

# Prompt configurations
cot_prompt_dict = {
    'llama_official': """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.""",
    'qwen': """Please reason step by step, and put your final answer within \\boxed{}.""",
    'default': """Please reason step by step, and put your final answer within \\boxed{}.""",
}

llm_step_tag_dict = {
    'llama': "## Step ",
    'qwen': "\nStep ",
    'default': "\nStep ",
}

sep_dict = {
    'llama': ["## Step"],
    'qwen': ["\nStep"],
    'default': ["\nStep"],
}

stop_str_dict = {
    'llama': ["\\boxed"],
    'qwen': ["\\boxed"],
    'default': ["\\boxed"],
}

# Global list to track background processes for cleanup
_background_processes = []


def cleanup():
    """Terminate all background processes."""
    print("\n=== Cleaning up background processes ===")
    for proc in _background_processes:
        if proc.poll() is None:
            print(f"Terminating process {proc.pid}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing process {proc.pid}...")
                proc.kill()
    print("Cleanup complete.")


def signal_handler(signum, frame):
    """Handle termination signals."""
    print(f"\nReceived signal {signum}, shutting down...")
    cleanup()
    sys.exit(1)


def setup_cleanup_handlers():
    """Register cleanup handlers for graceful shutdown."""
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def start_controller(host_addr: str, port: int, log_dir: Path) -> subprocess.Popen:
    """Start the FastChat controller."""
    print(f"Starting FastChat controller on port {port}...")

    log_file = log_dir / "controller.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "fastchat.serve.controller",
                "--port", str(port),
                "--host", host_addr,
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    _background_processes.append(proc)
    print(f"Controller started (PID: {proc.pid}), logging to {log_file}")
    return proc


def start_reward_model_worker(
    model_path: str,
    host_addr: str,
    controller_port: int,
    worker_port: int,
    log_dir: Path,
) -> subprocess.Popen:
    """Start the reward model worker."""
    print(f"Starting reward model worker on port {worker_port}...")
    print(f"  Model: {model_path}")

    log_file = log_dir / "reward_worker.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "reason.llm_service.workers.reward_model_worker",
                "--model-path", model_path,
                "--controller-address", f"http://{host_addr}:{controller_port}",
                "--host", host_addr,
                "--port", str(worker_port),
                "--worker-address", f"http://{host_addr}:{worker_port}",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    _background_processes.append(proc)
    print(f"Reward model worker started (PID: {proc.pid}), logging to {log_file}")
    return proc


def start_language_model_worker(
    model_path: str,
    host_addr: str,
    controller_port: int,
    worker_port: int,
    log_dir: Path,
) -> subprocess.Popen:
    """Start the language model worker using vLLM."""
    print(f"Starting language model worker on port {worker_port}...")
    print(f"  Model: {model_path}")

    log_file = log_dir / "lm_worker.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "reason.llm_service.workers.vllm_worker",
                "--model-path", model_path,
                "--controller-address", f"http://{host_addr}:{controller_port}",
                "--host", host_addr,
                "--port", str(worker_port),
                "--worker-address", f"http://{host_addr}:{worker_port}",
                "--gpu_memory_utilization", str(VLLM_CONFIG["gpu_memory_utilization"]),
                "--max_model_length", str(VLLM_CONFIG["max_model_length"]),
                "--swap_space", str(VLLM_CONFIG["swap_space"]),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    _background_processes.append(proc)
    print(f"Language model worker started (PID: {proc.pid}), logging to {log_file}")
    return proc


def wait_for_workers(
    host_addr: str,
    controller_port: int,
    policy_model_path: str,
    value_model_path: str,
    timeout_seconds: int = 300,
) -> bool:
    """Wait for workers to register with the controller."""
    print(f"\nWaiting for workers to register (timeout: {timeout_seconds}s)...")
    controller_url = f"http://127.0.0.1:{controller_port}/list_models"

    start_time = time.time()
    policy_model_name = os.path.basename(policy_model_path)
    value_model_name = os.path.basename(value_model_path)

    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.post(controller_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                has_policy = any(policy_model_name in m for m in models)
                has_value = any(value_model_name in m for m in models)

                elapsed = int(time.time() - start_time)
                print(f"  [{elapsed}s] Registered models: {models}")

                if has_policy and has_value:
                    print("Both workers registered successfully!")
                    return True
        except requests.exceptions.RequestException:
            pass

        for proc in _background_processes:
            if proc.poll() is not None:
                print(f"ERROR: Process {proc.pid} died with return code {proc.returncode}")
                return False

        time.sleep(10)

    print(f"ERROR: Timeout waiting for workers after {timeout_seconds}s")
    return False


def beam_search_with_tracing(
    config: BeamSearchConfig,
    gen_config: LMCallingConfig,
    problem_inst: dict,
    lm_calls: list,
    rm_call,
    tracer: ExecutionTracer,
) -> TreeSearchSolutionOutput:
    """Run beam search with execution tracing."""
    task = Task(task_name=config.task_name)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "beam_size": config.beam_size,
            "cot_prompt": config.cot_prompt,
            "stop_str": config.stop_str,
            "sep": config.sep,
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
            "is_few_shot": config.is_few_shot,
            "add_step_prompt": config.add_step_prompt,
            "direct_io": config.direct_io,
            "double_line_break": config.double_line_break,
            "model_names": config.model_names,
        },
        math_problems=[{
            "question": problem_inst["question"],
            "answer": problem_inst.get("extracted_groundtruth",
                                       task.extract_groundtruth(problem_inst["answer"])),
        }],
        llm_gen_fns=lm_calls,
        rm_call=rm_call,
        update_legal_action=False,
    )

    search_tree = SearchTree(cfg={
        "model_names": config.model_names,
        "direct_io": config.direct_io,
        "max_actions": config.tree_max_width
    })

    traj_list = search_tree.beam_search(env, config.beam_size, config.tree_max_depth, rm_call)

    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        reward_history=[t["reward_history"] for t in traj_list],
        token_history=[t["token_history"] for t in traj_list],
        prob_history=[t["prob_history"] for t in traj_list],
        model_history=[t["model_history"] for t in traj_list],
    )


def main():
    parser = ArgumentParser(description="Profile execution graph for test-time compute")

    # Model configuration
    parser.add_argument("--LM", type=str, default=DEFAULT_POLICY_MODEL_PATH,
                        help="Language model path")
    parser.add_argument("--RM", type=str, default=DEFAULT_VALUE_MODEL_PATH,
                        help="Reward model path")
    parser.add_argument("--host_addr", type=str, default=DEFAULT_HOST_ADDR,
                        help="Host address for servers")
    parser.add_argument("--controller_port", type=int, default=DEFAULT_CONTROLLER_PORT,
                        help="FastChat controller port")
    parser.add_argument("--worker_base_port", type=int, default=DEFAULT_WORKER_BASE_PORT,
                        help="Base port for workers")

    # Method configuration
    parser.add_argument("--method", type=str, default="beam_search",
                        choices=["beam_search", "best_of_n", "cot"],
                        help="TTS method to profile")
    parser.add_argument("--num_sequence", type=int, default=1,
                        help="Number of sequences/beams")
    parser.add_argument("--tree_max_depth", type=int, default=40,
                        help="Maximum tree depth")
    parser.add_argument("--tree_max_width", type=int, default=4,
                        help="Maximum tree width")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens per generation")
    parser.add_argument("--double_line_break", type=int, default=1,
                        help="Double line break mode")

    # Profiling configuration
    parser.add_argument("--num_problems", type=int, default=3,
                        help="Number of problems to profile")
    parser.add_argument("--task_name", type=str, default="MATH",
                        help="Task/dataset name")
    parser.add_argument("--output_dir", type=str, default="./profiles",
                        help="Directory for output files")
    parser.add_argument("--show_ascii_timeline", action="store_true",
                        help="Print ASCII timeline to console")
    parser.add_argument("--problem_indices", type=str, default=None,
                        help="Comma-separated list of problem indices to run")
    parser.add_argument("--skip_server_start", action="store_true",
                        help="Skip starting servers (use existing ones)")
    parser.add_argument("--controller_addr", type=str, default=None,
                        help="Use existing controller address (implies --skip_server_start)")

    args = parser.parse_args()

    # If controller_addr is provided, skip server start
    if args.controller_addr:
        args.skip_server_start = True

    # Set up model-specific configurations
    model_key = 'default'
    if 'llama-3' in args.LM.lower() or 'llama3' in args.LM.lower():
        model_key = 'llama'
    elif 'qwen' in args.LM.lower():
        model_key = 'qwen'

    args.cot_prompt = cot_prompt_dict.get(model_key, cot_prompt_dict['default'])
    if model_key == 'llama':
        args.cot_prompt = cot_prompt_dict['llama_official']
    args.llm_step_tag = llm_step_tag_dict.get(model_key, llm_step_tag_dict['default'])
    args.sep = sep_dict.get(model_key, sep_dict['default'])
    args.stop_str = stop_str_dict.get(model_key, stop_str_dict['default'])

    if args.double_line_break == 1:
        args.sep = ["\n\n"]

    # Adjust for method
    if args.method == "best_of_n":
        args.tree_max_depth = 1
        direct_io = 1
    elif args.method == "cot":
        args.tree_max_depth = 1
        args.num_sequence = 1
        args.tree_max_width = 1
        direct_io = 2
    else:
        direct_io = 0

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXECUTION GRAPH PROFILER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"LM: {args.LM}")
    print(f"RM: {args.RM}")
    print(f"Method: {args.method}")
    print(f"Num problems: {args.num_problems}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Setup working directory
    os.chdir(src_path)
    os.environ["PYTHONPATH"] = str(src_path)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["LOGDIR"] = str(src_path / "logs_fastchat")

    # Setup cleanup handlers
    setup_cleanup_handlers()

    # Create log directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"/scratch/msa6093/tts_logs/profile_{run_id}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Start servers if needed
    if not args.skip_server_start:
        print("\n=== Starting model servers ===")

        controller_proc = start_controller(args.host_addr, args.controller_port, log_dir)
        time.sleep(5)

        rm_proc = start_reward_model_worker(
            args.RM, args.host_addr, args.controller_port,
            args.worker_base_port, log_dir
        )
        time.sleep(2)

        lm_proc = start_language_model_worker(
            args.LM, args.host_addr, args.controller_port,
            args.worker_base_port + 1, log_dir
        )

        if not wait_for_workers(
            args.host_addr, args.controller_port,
            args.LM, args.RM, timeout_seconds=300
        ):
            print("ERROR: Failed to start workers. Check logs in:", log_dir)
            cleanup()
            return 1

        controller_addr = f"http://{args.host_addr}:{args.controller_port}"
    else:
        controller_addr = args.controller_addr or f"http://{args.host_addr}:{args.controller_port}"
        print(f"\nUsing existing controller at: {controller_addr}")

    # Initialize the tracer
    tracer = ExecutionTracer.get_instance()

    # Create traced LM callers
    print("\nInitializing traced LM caller...")
    lm_calls = [
        TracedVLLMRemoteCaller(
            model_name=args.LM,
            model_path=args.LM,
            controller_addr=controller_addr,
            llm_step_tag=args.llm_step_tag,
            apply_chat_template=True,
            multi_gpu=False,
            serve_type="fastchat",
            double_line_break=args.double_line_break,
            tracer=tracer,
        )
    ]

    # Create traced RM caller
    print("Initializing traced RM caller...")
    if "dummy" in args.RM.lower():
        rm_config = RemoteRewardModelConfig(
            prm_step_tag="ки\n",
            format_str="{question} {answer}",
            model_name=args.RM,
            controller_addr=controller_addr,
            step_tag_id=None,
            returned_token_ids=None,
            rm_serve_type="fastchat",
            multi_gpu=False,
        )
        rm_call_base = DummyRewardModelCaller(rm_config)
        rm_call = partial(rm_call_base, model_names=[args.LM])
        rm_call = wrap_existing_rm_call(rm_call)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.RM, trust_remote_code=True)
        step_tag_id, returned_token_ids = get_prm_special_tokens(args.RM, tokenizer)

        if 'pqm' in args.RM.lower():
            prm_format_str = "{question}\n{answer}"
        else:
            prm_format_str = "{question} {answer}"

        rm_config = RemoteRewardModelConfig(
            prm_step_tag="ки\n",
            format_str=prm_format_str,
            model_name=args.RM,
            controller_addr=controller_addr,
            step_tag_id=step_tag_id,
            returned_token_ids=returned_token_ids,
            rm_serve_type="fastchat",
            multi_gpu=False,
        )
        rm_call_base = TracedRMRemoteCaller(rm_config, tokenizer=tokenizer, tracer=tracer)
        rm_call = partial(rm_call_base, model_names=[args.LM])

    # Create method configuration
    gen_config = LMCallingConfig(
        n=args.num_sequence,
        temperature=args.temperature,
        top_k=-1,
        top_p=1.0,
        max_new_tokens=args.max_new_tokens,
    )

    method_config = BeamSearchConfig(
        task_name=args.task_name,
        tree_max_depth=args.tree_max_depth,
        tree_max_width=args.tree_max_width,
        beam_size=args.num_sequence,
        model_names=[args.LM],
        is_few_shot=False,
        add_step_prompt=True,
        cot_prompt=args.cot_prompt,
        stop_str=args.stop_str if args.method == "beam_search" else None,
        sep=args.sep,
        direct_io=direct_io,
        double_line_break=args.double_line_break,
    )

    # Load test dataset
    print(f"\nLoading {args.task_name} test dataset...")
    _, test_ds = get_env_datasets(args.task_name)

    # Select problems to profile
    if args.problem_indices:
        indices = [int(i) for i in args.problem_indices.split(",")]
    else:
        indices = list(range(min(args.num_problems, len(test_ds))))

    problems = [test_ds[i] for i in indices]
    print(f"Selected {len(problems)} problems: indices {indices}")

    # Start tracing
    print("\n" + "-" * 60)
    print("Starting execution trace...")
    print("-" * 60)

    tracer.start_trace()
    total_start = time.time()

    for i, problem in enumerate(problems):
        print(f"\n[Problem {i+1}/{len(problems)}] Index: {indices[i]}")
        print(f"  Question: {problem['question'][:80]}...")

        problem_start = time.time()

        try:
            result = beam_search_with_tracing(
                config=method_config,
                gen_config=gen_config,
                problem_inst=problem,
                lm_calls=lm_calls,
                rm_call=rm_call,
                tracer=tracer,
            )
            problem_time = time.time() - problem_start
            print(f"  Completed in {problem_time:.2f}s, {len(result.solutions)} solutions")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start

    # End tracing
    tracer.end_trace()

    print("\n" + "-" * 60)
    print(f"Total profiling time: {total_time:.2f}s")
    print("-" * 60)

    # Compute stall times
    print("\nComputing stall times...")
    tracer.compute_stall_times()

    # Print summary
    tracer.print_summary()

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export Chrome trace
    chrome_trace_path = output_dir / f"chrome_trace_{timestamp}.json"
    print(f"\nExporting Chrome trace to: {chrome_trace_path}")
    export_chrome_trace(tracer, str(chrome_trace_path))

    # Export structured JSON
    json_path = output_dir / f"execution_graph_{timestamp}.json"
    print(f"Exporting structured JSON to: {json_path}")
    export_structured_json(tracer, str(json_path))

    # Export CSV
    csv_path = output_dir / f"timeline_{timestamp}.csv"
    print(f"Exporting timeline CSV to: {csv_path}")
    export_timeline_csv(tracer, str(csv_path))

    # Save config
    config_path = output_dir / f"config_{timestamp}.json"
    config_data = {
        "LM": args.LM,
        "RM": args.RM,
        "method": args.method,
        "num_problems": len(problems),
        "problem_indices": indices,
        "tree_max_depth": args.tree_max_depth,
        "tree_max_width": args.tree_max_width,
        "num_sequence": args.num_sequence,
        "temperature": args.temperature,
        "total_time_seconds": total_time,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Optional ASCII timeline
    if args.show_ascii_timeline:
        print_timeline_ascii(tracer)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print(f"\nTo visualize the execution graph:")
    print(f"  1. Open https://ui.perfetto.dev/ or chrome://tracing")
    print(f"  2. Load: {chrome_trace_path}")
    print(f"\nFor programmatic analysis, see: {json_path}")
    print(f"\nServer logs available at: {log_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
