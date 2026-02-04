#!/usr/bin/env python3
"""Self-contained beam search inference script for PBS jobs.

This script:
1. Starts the FastChat controller
2. Starts reward model and language model workers
3. Waits for workers to register with the controller
4. Runs beam search evaluation
5. Cleans up all processes on exit
"""

import subprocess
import os
import sys
import time
import signal
import atexit
import requests
from datetime import datetime
from pathlib import Path

# === Configuration ===
POLICY_MODEL_PATH = "/scratch/msa6093/models/Qwen2.5-1.5B-Instruct"
VALUE_MODEL_PATH = "/scratch/msa6093/models/math-shepherd-mistral-7b-prm"
HOST_ADDR = "0.0.0.0"
CONTROLLER_PORT = 10014
WORKER_BASE_PORT = 10081

# Beam search parameters
BEAM_CONFIG = {
    "method": "beam_search",
    "task_name": "MATH",
    "temperature": 0.7,
    "max_new_tokens": 2048,
    "tree_max_depth": 40,
    "tree_max_width": 4,
    "num_sequence": 1,
    "num_worker": 12,
    "batch_size": 500,
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
    """Terminate all background processes."""
    print("\n=== Cleaning up background processes ===")
    for proc in _background_processes:
        if proc.poll() is None:  # Process is still running
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


def start_language_model_worker(log_dir: Path) -> subprocess.Popen:
    """Start the language model worker using vLLM."""
    worker_port = WORKER_BASE_PORT + 1
    print(f"Starting language model worker on port {worker_port}...")
    print(f"  Model: {POLICY_MODEL_PATH}")

    log_file = log_dir / "lm_worker.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "reason.llm_service.workers.vllm_worker",
                "--model-path", POLICY_MODEL_PATH,
                "--controller-address", f"http://{HOST_ADDR}:{CONTROLLER_PORT}",
                "--host", HOST_ADDR,
                "--port", str(worker_port),
                "--worker-address", f"http://{HOST_ADDR}:{worker_port}",
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


def run_evaluation(save_dir: Path) -> int:
    """Run the beam search evaluation.

    Returns:
        Return code from the evaluation process
    """
    print("\n=== Running beam search evaluation ===")

    controller_addr = f"http://{HOST_ADDR}:{CONTROLLER_PORT}"

    cmd = [
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
        "--batch_size", str(BEAM_CONFIG["batch_size"]),
        "--max_time", str(BEAM_CONFIG["max_time"]),
        "--local", str(BEAM_CONFIG["local"]),
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode


def main():
    """Main entry point."""
    print("=" * 60)
    print("Beam Search Inference Script")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Setup paths
    tts_source = Path("/scratch/msa6093/compute-optimal-tts/src")
    os.chdir(tts_source)

    # Create log directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"/scratch/msa6093/tts_logs/{run_id}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ["PYTHONPATH"] = str(tts_source)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["LOGDIR"] = str(tts_source / "logs_fastchat")

    print(f"\nConfiguration:")
    print(f"  Working directory: {tts_source}")
    print(f"  Log directory: {log_dir}")
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

    lm_proc = start_language_model_worker(log_dir)

    # Wait for workers to register
    if not wait_for_workers(timeout_seconds=300):
        print("ERROR: Failed to start workers. Check logs in:", log_dir)
        cleanup()
        return 1

    # Run evaluation
    save_dir = tts_source / "output"
    save_dir.mkdir(parents=True, exist_ok=True)

    return_code = run_evaluation(save_dir)

    print(f"\n=== Evaluation completed with return code: {return_code} ===")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Cleanup is handled by atexit
    return return_code


if __name__ == "__main__":
    sys.exit(main())
