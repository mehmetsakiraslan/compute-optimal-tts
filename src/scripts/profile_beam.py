#!/usr/bin/env python3
"""Profiling script for beam search inference using NVIDIA Nsight Systems.

This script:
1. Starts the FastChat controller
2. Starts reward model and language model workers
3. Waits for workers to register with the controller
4. Runs beam search evaluation with Nsight Systems profiling
5. Cleans up all processes on exit

Produces .nsys-rep files that can be analyzed in Nsight Systems UI to investigate:
- Variable parallelism (batch size changes from 128 at depth 0 to 16 at depths 1+)
- Branch dependencies (blocking RM calls, sequential depth loops)
"""

import subprocess
import os
import sys
import time
import signal
import atexit
import argparse
import requests
from datetime import datetime
from pathlib import Path

# === Configuration ===
POLICY_MODEL_PATH = "/scratch/msa6093/models/Qwen2.5-1.5B-Instruct"
VALUE_MODEL_PATH = "/scratch/msa6093/models/math-shepherd-mistral-7b-prm"
HOST_ADDR = "0.0.0.0"
CONTROLLER_PORT = 10014
WORKER_BASE_PORT = 10081

# Beam search parameters - reduced for profiling
BEAM_CONFIG = {
    "method": "beam_search",
    "task_name": "MATH",
    "temperature": 0.7,
    "max_new_tokens": 2048,
    "tree_max_depth": 40,
    "tree_max_width": 4,
    "num_sequence": 1,
    "num_worker": 1,  # Single worker for clearer profiling
    "batch_size": 50,  # Reduced batch size for profiling
    "max_time": 3,
    "double_line_break": 1,
    "local": 0,
    "question_parallel_num": 0,
}

# vLLM worker parameters
VLLM_CONFIG = {
    "gpu_memory_utilization": 0.88,
    "max_model_length": 8192,
    "swap_space": 16,
}

# Global list to track background processes for cleanup
_background_processes = []


def cleanup():
    """Terminate all background processes.

    Uses SIGINT first to allow nsys to gracefully finish and write profile data,
    then falls back to SIGTERM and SIGKILL if needed.
    """
    print("\n=== Cleaning up background processes ===")

    # First pass: send SIGINT to allow graceful shutdown (important for nsys)
    for proc in _background_processes:
        if proc.poll() is None:
            print(f"Sending SIGINT to process {proc.pid} (allowing nsys to finalize)...")
            try:
                proc.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass

    # Wait for nsys to generate report (can take time for large profiles)
    print("Waiting for processes to finalize (up to 30s for nsys report generation)...")
    time.sleep(5)  # Give nsys time to start finalizing

    # Second pass: check and terminate any remaining processes
    for proc in _background_processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=25)  # Wait up to 25 more seconds
                print(f"Process {proc.pid} finished gracefully")
            except subprocess.TimeoutExpired:
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


def start_controller(log_dir: Path) -> subprocess.Popen:
    """Start the FastChat controller."""
    print(f"Starting FastChat controller on port {CONTROLLER_PORT}...")

    log_file = log_dir / "controller.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "fastchat.serve.controller",
                "--port", str(CONTROLLER_PORT),
                "--host", HOST_ADDR,
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    _background_processes.append(proc)
    print(f"Controller started (PID: {proc.pid}), logging to {log_file}")
    return proc


def start_reward_model_worker(log_dir: Path) -> subprocess.Popen:
    """Start the reward model worker."""
    worker_port = WORKER_BASE_PORT
    print(f"Starting reward model worker on port {worker_port}...")
    print(f"  Model: {VALUE_MODEL_PATH}")

    log_file = log_dir / "reward_worker.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "reason.llm_service.workers.reward_model_worker",
                "--model-path", VALUE_MODEL_PATH,
                "--controller-address", f"http://{HOST_ADDR}:{CONTROLLER_PORT}",
                "--host", HOST_ADDR,
                "--port", str(worker_port),
                "--worker-address", f"http://{HOST_ADDR}:{worker_port}",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    _background_processes.append(proc)
    print(f"Reward model worker started (PID: {proc.pid}), logging to {log_file}")
    return proc


def start_language_model_worker(log_dir: Path, profile_dir: Path = None) -> subprocess.Popen:
    """Start the language model worker using vLLM.

    Args:
        log_dir: Directory for worker logs
        profile_dir: If provided, wrap worker with nsys profiling
    """
    worker_port = WORKER_BASE_PORT + 1
    print(f"Starting language model worker on port {worker_port}...")
    print(f"  Model: {POLICY_MODEL_PATH}")

    # Base worker command
    worker_cmd = [
        sys.executable, "-m", "reason.llm_service.workers.vllm_worker",
        "--model-path", POLICY_MODEL_PATH,
        "--controller-address", f"http://{HOST_ADDR}:{CONTROLLER_PORT}",
        "--host", HOST_ADDR,
        "--port", str(worker_port),
        "--worker-address", f"http://{HOST_ADDR}:{worker_port}",
        "--gpu_memory_utilization", str(VLLM_CONFIG["gpu_memory_utilization"]),
        "--max_model_length", str(VLLM_CONFIG["max_model_length"]),
        "--swap_space", str(VLLM_CONFIG["swap_space"]),
    ]

    # Wrap with nsys if profiling is enabled
    if profile_dir:
        profile_output = profile_dir / "vllm_worker_profile"
        print(f"  Profiling worker with nsys -> {profile_output}.nsys-rep")
        cmd = [
            "nsys", "profile",
            "-t", "cuda,nvtx,osrt,cudnn,cublas",
            "-o", str(profile_output),
            "-f", "true",
        ] + worker_cmd
    else:
        cmd = worker_cmd

    log_file = log_dir / "lm_worker.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    _background_processes.append(proc)
    print(f"Language model worker started (PID: {proc.pid}), logging to {log_file}")
    return proc


def wait_for_workers(timeout_seconds: int = 300) -> bool:
    """Wait for workers to register with the controller.

    Args:
        timeout_seconds: Maximum time to wait (default 5 minutes)

    Returns:
        True if both workers registered, False on timeout
    """
    print(f"\nWaiting for workers to register (timeout: {timeout_seconds}s)...")
    controller_url = f"http://127.0.0.1:{CONTROLLER_PORT}/list_models"

    start_time = time.time()
    policy_model_name = os.path.basename(POLICY_MODEL_PATH)
    value_model_name = os.path.basename(VALUE_MODEL_PATH)

    while time.time() - start_time < timeout_seconds:
        try:
            # FastChat controller uses POST for /list_models
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

        # Check if any worker process has died
        for proc in _background_processes:
            if proc.poll() is not None:
                print(f"ERROR: Process {proc.pid} died with return code {proc.returncode}")
                return False

        time.sleep(10)

    print(f"ERROR: Timeout waiting for workers after {timeout_seconds}s")
    return False


def run_profiled_evaluation(save_dir: Path, profile_dir: Path, num_problems: int) -> int:
    """Run the beam search evaluation.

    Note: GPU profiling is done on the vLLM worker process, not here.
    The worker is wrapped with nsys when started.

    Args:
        save_dir: Directory to save evaluation results
        profile_dir: Directory to save profiling output (for reference)
        num_problems: Number of problems to evaluate (for shorter profiling)

    Returns:
        Return code from the evaluation process
    """
    print("\n=== Running beam search evaluation ===")
    print(f"(GPU profiling is active on vLLM worker -> {profile_dir}/vllm_worker_profile.nsys-rep)")

    controller_addr = f"http://{HOST_ADDR}:{CONTROLLER_PORT}"

    # Build the evaluation command - run directly without nsys wrapper
    # GPU profiling happens on the vLLM worker process
    eval_cmd = [
        sys.executable, "reason/evaluation/evaluate.py",
        "--LM", POLICY_MODEL_PATH,
        "--RM", VALUE_MODEL_PATH,
        "--task_name", BEAM_CONFIG["task_name"],
        "--temperature", str(BEAM_CONFIG["temperature"]),
        "--max_new_tokens", str(BEAM_CONFIG["max_new_tokens"]),
        "--num_sequence", str(BEAM_CONFIG["num_sequence"]),
        "--tree_max_width", str(BEAM_CONFIG["tree_max_width"]),
        "--tree_max_depth", str(BEAM_CONFIG["tree_max_depth"]),
        "--save_dir", str(save_dir),
        "--method", BEAM_CONFIG["method"],
        "--num_worker", str(BEAM_CONFIG["num_worker"]),
        "--controller_addr", controller_addr,
        "--add_step_prompt",
        "--question_parallel_num", str(BEAM_CONFIG["question_parallel_num"]),
        "--double_line_break", str(BEAM_CONFIG["double_line_break"]),
        "--batch_size", str(num_problems),  # Use num_problems as batch_size
        "--max_time", str(BEAM_CONFIG["max_time"]),
        "--local", str(BEAM_CONFIG["local"]),
    ]

    print(f"Command: {' '.join(eval_cmd[:10])}...")
    print()

    result = subprocess.run(eval_cmd, env=os.environ.copy())
    return result.returncode


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile beam search inference with Nsight Systems"
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=5,
        help="Number of problems to profile (default: 5)"
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default=None,
        help="Directory for profile output (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip model warmup before profiling"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Beam Search Profiling Script (Nsight Systems)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Setup paths
    tts_source = Path("/scratch/msa6093/compute-optimal-tts/src")
    os.chdir(tts_source)

    # Create log and profile directories
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"/scratch/msa6093/tts_logs/profile_{run_id}")
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.profile_dir:
        profile_dir = Path(args.profile_dir)
    else:
        profile_dir = Path(f"/scratch/msa6093/tts_profiles/{run_id}")
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ["PYTHONPATH"] = str(tts_source)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["LOGDIR"] = str(tts_source / "logs_fastchat")

    print(f"\nConfiguration:")
    print(f"  Working directory: {tts_source}")
    print(f"  Log directory: {log_dir}")
    print(f"  Profile directory: {profile_dir}")
    print(f"  Number of problems: {args.num_problems}")
    print(f"  Policy model: {POLICY_MODEL_PATH}")
    print(f"  Value model: {VALUE_MODEL_PATH}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Setup cleanup handlers
    setup_cleanup_handlers()

    # Start services
    print("\n=== Starting model servers ===")

    controller_proc = start_controller(log_dir)
    time.sleep(5)  # Wait for controller to initialize

    rm_proc = start_reward_model_worker(log_dir)
    time.sleep(2)

    # Start LM worker with nsys profiling to capture GPU activity
    lm_proc = start_language_model_worker(log_dir, profile_dir=profile_dir)

    # Wait for workers to register
    if not wait_for_workers(timeout_seconds=300):
        print("ERROR: Failed to start workers. Check logs in:", log_dir)
        cleanup()
        return 1

    # Run profiled evaluation - use profile-specific output directory to avoid conflicts
    # with existing completed runs
    save_dir = tts_source / "output" / f"profile_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    return_code = run_profiled_evaluation(save_dir, profile_dir, args.num_problems)

    print(f"\n=== Evaluation completed with return code: {return_code} ===")
    print(f"GPU profile output: {profile_dir}/vllm_worker_profile.nsys-rep")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTo analyze GPU results, open the vllm_worker_profile.nsys-rep file in Nsight Systems UI")

    # Cleanup is handled by atexit
    return return_code


if __name__ == "__main__":
    sys.exit(main())
