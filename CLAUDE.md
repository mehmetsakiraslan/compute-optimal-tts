# CLAUDE.md

## Project Overview
Research implementation of compute-optimal test-time scaling (TTS) for LLMs on mathematical reasoning tasks. Demonstrates that smaller 1B LLMs can surpass 405B LLMs through intelligent test-time scaling.

## Build & Installation
```bash
cd src
conda create -n tts python=3.10
conda activate tts
pip install -r requirements.txt
```

## Commands

### Model Serving (FastChat + vLLM)
```bash
# Start model servers (run in separate terminals)
bash src/scripts/serve_gpu1.sh  # Controller
bash src/scripts/serve_gpu2.sh  # Language model worker
bash src/scripts/serve_gpu3_1-2.sh  # Reward model worker
```

### Evaluation
```bash
bash src/scripts/run.sh
```
Key flags in run.sh:
- `--method`: TTS method (best_of_n, beam_search, dvts)
- `--LM`, `--RM`: Model endpoints
- `--num_sequence`: Number of samples/beams
- `--task`: Dataset (MATH, etc.)

## Architecture

### Entry Point
`src/reason/evaluation/evaluate.py` - Main evaluation script

### Core Modules
- `src/reason/evaluation/` - Evaluation orchestration, TTS method implementations
  - `evaluator.py` - Base evaluator class
  - `bon_evaluator.py` - Best-of-N sampling
  - `beam_evaluator.py` - Beam search with PRM guidance
- `src/reason/inference/` - LM/RM inference via vLLM
  - `lm_call.py` - Language model API calls
  - `rm_call.py` - Reward model scoring
- `src/reason/guided_search/` - Tree search algorithms
  - `tree.py` - Search tree data structures
  - `beam_search.py` - Beam search implementation
- `src/reason/reranking/` - Answer aggregation strategies
  - `vote.py` - Majority voting, weighted voting
- `src/envs/MATH/` - Math task environment
  - `env.py` - Environment wrapper
  - `grader.py` - Answer verification

### LLM Service Layer
`src/reason/llm_service/` - FastChat-based model serving
- `workers/` - vLLM worker implementations
- `workers/skywork_o1_prm_inference/` - Process reward model

## Key Files for Common Tasks

| Task | File |
|------|------|
| Add new TTS method | `src/reason/evaluation/` (create new evaluator) |
| Modify search algorithm | `src/reason/guided_search/beam_search.py` |
| Change answer aggregation | `src/reason/reranking/vote.py` |
| Add new math dataset | `src/envs/MATH/` |
| Adjust model inference | `src/reason/inference/lm_call.py`, `rm_call.py` |
