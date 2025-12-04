# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AgentFlux is a framework for optimizing LLM tool-calling through **DualTune**, a dual-stage finetuning approach that separates tool classification from argument generation. The system consists of three main components:

1. **DualTune Finetuning Pipeline** (`finetune/`) - Generates synthetic training data and trains specialized models
2. **AgentFlux Inference System** (`inference/agentflux/`) - FastAPI proxy that routes requests through finetuned models
3. **Rena Core Orchestration** (`orchestration_framework/`) - Rust-based evaluation framework with Python MCP runtime

## Key Architecture Principles

### DualTune Two-Stage Approach

The core innovation is **decoupled optimization**:
- **Stage 1**: A lightweight **classifier model** selects which tool to use (voting across 10 samples at temperature=1.0)
- **Stage 2**: Per-tool **adapter models** generate precise arguments for the selected tool

This separation allows each model to specialize rather than forcing a single model to handle both routing and argument generation.

### Tool Categories

The system is organized by **categories** (e.g., `filesys`, `monday`, `notion`). Each category contains:
- Multiple related tools (e.g., filesys has `read_file`, `write_file`, `list_directory`, etc.)
- One shared classifier model for all tools in the category
- Individual adapter models for each tool

When adding support for new tools, you typically work within a category context.

## Common Commands

### Finetuning Pipeline

```bash
cd finetune

# Full pipeline for a category (query gen → trajectory collection → data cleaning → training)
bash scripts/finetune.sh <category>

# Train only the classifier
bash scripts/finetune_classifier.sh <category> [batch_size] [grad_accum] [epochs]
# Default: batch_size=4, grad_accum=4, epochs=4

# Train only the tool adapters
bash scripts/finetune_tool_adaptors.sh <category> [batch_size] [grad_accum] [epochs]

# Generate synthetic queries (requires OPENAI_API_KEY)
python gen_queries.py --category <category> --output_path <path>

# Collect trajectories from baseline model (requires OPENAI_API_KEY and WORKSPACE env var)
WORKSPACE=/path/to/workspace python gen_trajs.py \
  --category <category> \
  --model gpt-5-mini \
  --input_queries <queries.txt> \
  --output_trajs <trajs.jsonl>

# Prepare data (clean, validate, split into train/eval/test)
python data_prepare.py --category <category>

# Generate tool-specific chat templates
python gen_tool_template.py --category <category> --model_folder <base_model_path>
```

**Training outputs**:
- Models: `finetune/<category>/results/finetune_output/`
- Epoch 2 checkpoints (used for serving): `finetune/<category>/results/finetune_serve/`
- Logs: `finetune/<category>/results/log/`
- Data: `finetune/<category>/results/trajectories/`

### Inference System

```bash
cd inference/agentflux

# Start vLLM server with LoRA adapters (edit vllm.sh to configure category and paths)
bash scripts/vllm.sh &

# Start AgentFlux proxy (defaults to port 9015)
bash scripts/proxy.sh [category]
# Default category: filesys

# The proxy exposes: http://localhost:9015/v1/chat/completions
```

The proxy is OpenAI-compatible - send requests using the OpenAI SDK with `base_url="http://localhost:9015/v1"`.

### Evaluation

```bash
cd orchestration_framework/evaluation

# Full evaluation pipeline
bash scripts/evaluate.sh <category>

# Individual steps:
# 1. Generate test queries
python gen_queries.py --category <category>

# 2. Run with AgentFlux
python run_agentflux.py <category> \
  --classifier config/<category>/classifier.json \
  --tool_adapters config/<category>/tool_adapters.json \
  --query <category>/queries/fuzzing_queries.txt \
  --output <category>/eval-results/trajs.jsonl

# 3. Run baseline comparison
python run_baseline.py <category> \
  --query <category>/queries/fuzzing_queries.txt \
  --output <category>/baseline-results/trajs.jsonl

# 4. Judge results
python <category>/judge.py \
  --trajs <category>/eval-results/trajs.jsonl \
  --output <category>/judge-results/judged.jsonl

# 5. Calculate scores
python score.py --llm_judge_path <category>/judge-results/judged.jsonl
```

### Rena Core (Rust Orchestration)

```bash
cd orchestration_framework/rena-core

# Setup (requires Python 3.11, Rust, Docker)
make setup

# Build only browserd (Rust)
make setup_browserd

# Build only runtime (Python)
make setup_runtime

# Run tests
make test

# Cleanup Docker images
make cleanup
```

## Configuration Files

Each category requires these files in `inference/agentflux/config/<category>/`:

- **`tool_list.json`**: MCP tool definitions in OpenAI format (array of tool objects)
- **`classifier.json`**: Classifier model configuration
  ```json
  {
    "model": "filesys-classifier",
    "port": 8001,
    "tools": ["read_file", "write_file", ...]
  }
  ```
- **`tool_adapters.json`**: Per-tool adapter configurations
  ```json
  {
    "read_file": {"model": "read_file-adapter", "port": 8002},
    "write_file": {"model": "write_file-adapter", "port": 8003}
  }
  ```

For finetuning, each category needs in `finetune/<category>/`:

- **`query_generation_template.txt`**: Prompt for generating synthetic queries
- **`base_models/<model_name>/classifier.jinja`**: Chat template for classifier
- **`base_models/<model_name>/<category>/<tool>.jinja`**: Per-tool chat templates

For evaluation:

- **`orchestration_framework/evaluation/<category>/judge.py`**: Category-specific judging logic
- **`orchestration_framework/evaluation/<category>/judge_sys_prompt.txt`**: LLM judge prompt

## Code Structure Details

### Finetuning Pipeline (`finetune/`)

**Data Flow**: Query Gen → Trajectory Collection → Data Cleaning → Model Training

**Key Files**:
- `gen_queries.py`: Uses GPT-5 to generate diverse queries per category
- `gen_trajs.py`: Collects full tool-calling conversations from baseline model
- `data_prepare.py`: **Critical validation logic**:
  - Validates required vs optional parameters against tool schemas
  - Type checks all arguments (string, number, boolean, array, object)
  - Rejects unexpected arguments
  - Filters conversations > 32,768 tokens
  - Uses binary search to remove problematic entries
  - Splits into 80/10/10 train/eval/test
- `gen_tool_template.py`: Generates Jinja2 templates for each tool using base model's tokenizer
- `unsloth-cli-split.py`: Main training script using Unsloth + LoRA
  - Base model: Qwen2.5-7B-Instruct (32K context)
  - LoRA: rank=32, alpha=64, dropout=0.1
  - Targets: Q/K/V/O projections + Gate/Up/Down projections
  - Optimizer: AdamW-8bit with cosine scheduling
  - Learning rate: 5e-6

**Training Scripts**:
- `scripts/finetune_classifier.sh`: Trains one classifier per category
  - Output: `<category>/results/finetune_output/classifier/`
  - Serves from epoch 2: `<category>/results/finetune_serve/classifier/`
- `scripts/finetune_tool_adaptors.sh`: Loops through tools, trains one adapter per tool
  - Each tool gets custom chat template from `base_models/<model>/<category>/<tool>.jinja`
  - Outputs: `<category>/results/finetune_output/tool_adaptors/<tool>/`

### Inference System (`inference/agentflux/`)

**Request Flow**: Proxy → Classifier (10-sample voting) → Tool Adapter (single generation)

**Key Files**:
- `agentflux/proxy.py`: FastAPI server (port 8030 by default)
  - Endpoint: `/v1/chat/completions`
  - Substitutes tool list from config before processing
  - Handles tool name substitution (e.g., `get_users_by_name` → `list_users_and_teams`)
- `agentflux/classifier.py`:
  - `FinetunedClassifier`: Generates n=10 completions at temp=1.0, extracts tools from `<tool_call>` tags, votes by frequency
  - `GPTClassifier`: Baseline using GPT API
- `agentflux/tool_adaptor.py`:
  - `FinetunedToolAdaptor`: Uses `tool_choice` to force specific tool, generates arguments
  - Implements caching: tracks up to 5 unique calls, blocks after 10 identical calls (loop prevention)
  - Cache clears when "summarize" tool is called
  - Retry logic: max 3 attempts
  - `GPTToolAdaptor`: Baseline using GPT API

**Configuration**:
- `scripts/vllm.sh`: Starts vLLM server with multiple LoRA adapters
  - Uses `--enable-lora` and `--lora-modules` to load classifier + all tool adapters
  - Default port: 8010
  - Edit `CUDA_VISIBLE_DEVICES` to select GPU
- `scripts/proxy.sh`: Starts proxy with category config
  - Loads classifier.json, tool_adapters.json, tool_list.json

### Orchestration Framework (`orchestration_framework/`)

**Architecture**: Rust process manager (browserd) + Python MCP runtime

**Rena Core** (`rena-core/`):
- `rena-browserd/` (Rust):
  - `browserd/`: Core library for Docker/process management
  - `browserd-cli/`: CLI for running queries and managing apps
  - `browserd-eval/`: Evaluation harness
  - Uses: tokio (async), bollard (Docker), tonic (gRPC), tracing (logging)
- `rena-runtime/` (Python):
  - MCP protocol implementation
  - Tool invocation and response handling
  - Trajectory logging (JSONL format)
  - Container lifecycle management

**Evaluation** (`evaluation/`):
- `gen_queries.py`: Generates test queries for a category
- `run_agentflux.py`: Runs queries through AgentFlux proxy, logs trajectories
- `run_baseline.py`: Runs queries through baseline model for comparison
- `<category>/judge.py`: Category-specific validation (checks file operations, API calls, etc.)
- `score.py`: Aggregates judgments and calculates success metrics

**Judge Implementation Pattern**: Each judge validates that the tool calls in trajectories match expected behavior for the query. For example, `filesys/judge.py` verifies file paths are correct, operations match intent, etc.

## Working with Tool Categories

### Adding a New Category

1. **Create config structure**:
   ```bash
   mkdir -p inference/agentflux/config/new_category
   # Add tool_list.json, classifier.json, tool_adapters.json
   ```

2. **Create finetuning structure**:
   ```bash
   mkdir -p finetune/new_category
   # Add query_generation_template.txt
   ```

3. **Generate training data**:
   ```bash
   cd finetune
   # Edit gen_queries.py if needed for category-specific query generation
   bash scripts/finetune.sh new_category
   ```

4. **Create evaluation infrastructure**:
   ```bash
   mkdir -p orchestration_framework/evaluation/new_category
   # Implement judge.py with category-specific validation logic
   # Add judge_sys_prompt.txt
   ```

5. **Update vllm.sh**: Add LoRA module paths for new category's classifier and tool adapters

### Modifying Training Hyperparameters

All training uses these fixed hyperparameters (in `finetune_classifier.sh` and `finetune_tool_adaptors.sh`):
- Learning rate: 5e-6 (fixed)
- Scheduler: cosine
- LoRA: rank=32, alpha=64, dropout=0.1
- Optimizer: adamw_8bit
- Max sequence length: 32,768 tokens

Configurable via script arguments:
- Batch size (default: 4)
- Gradient accumulation steps (default: 4)
- Number of epochs (default: 4)
- Base model (default: unsloth/Qwen2.5-7B-Instruct)

The scripts automatically calculate `eval_steps` based on dataset size to ensure ~5 evaluations per epoch.

### Important: Epoch 2 Checkpoints

Both training scripts copy the **epoch 2 checkpoint** to `results/finetune_serve/` for serving:
```bash
# From finetune_classifier.sh
cp -r "$output_dir/checkpoint-$(($steps_per_epoch * 2))/"* "$epoch_2_dir/"
```
This is the checkpoint actually used by vLLM. If you want to serve a different checkpoint, update these paths.

## Environment Variables

- **`OPENAI_API_KEY`**: Required for query/trajectory generation and baseline evaluation
- **`WORKSPACE`**: Required for filesys category trajectory generation (path to test workspace)
- **`CUDA_VISIBLE_DEVICES`**: Set in `vllm.sh` to select GPU

## Development Notes

### Data Validation is Critical

The `data_prepare.py` script performs rigorous validation because invalid training data will cause training failures or model quality issues:
- Always validates arguments against tool schemas
- Uses binary search to isolate problematic entries when Dataset creation fails
- Filters by token length to stay within 32K context window

If you modify tool schemas, ensure `data_prepare.py` validation logic matches.

### Classifier Voting Strategy

The classifier generates 10 completions to achieve robustness against inconsistent outputs. The tool name with the most votes wins. Falls back to "summarize" if no tools detected. This multi-sample approach is more reliable than single-shot classification with smaller models.

### Tool Adapter Caching

The tool adapter tracks function call history to prevent infinite loops:
- Stores up to 5 unique calls per request
- Blocks after 10 identical calls (uses SHA256 hash)
- Clears cache when "summarize" is called

This is implemented in `tool_adaptor.py:FinetunedToolAdaptor.adapt()`.

### Chat Templates

Each tool gets a custom Jinja2 template that formats the conversation for that specific tool. These templates are auto-generated by `gen_tool_template.py` based on the base model's tokenizer and the tool's schema. The templates are stored in `finetune/base_models/<model>/<category>/<tool>.jinja`.

### Port Allocation

Default ports (configurable):
- vLLM server: 8010
- AgentFlux proxy: 9015
- Classifier model: 8001 (in classifier.json)
- Tool adapters: 8002+ (in tool_adapters.json, one per tool)

Make sure these don't conflict with other services.

## Testing and Evaluation

### Running a Single Test

```bash
# Using browserd-cli (after make setup)
cd orchestration_framework/rena-core/rena-browserd
cargo run --bin browserd-cli -- run-query \
  --query "Read the contents of README.md" \
  --category filesys
```

### Comparing AgentFlux vs Baseline

1. Run both evaluation scripts with same queries:
   ```bash
   python orchestration_framework/evaluation/run_agentflux.py filesys ...
   python orchestration_framework/evaluation/run_baseline.py filesys ...
   ```

2. Judge both trajectories with same judge script:
   ```bash
   python orchestration_framework/evaluation/filesys/judge.py --trajs <trajs.jsonl> --output <judged.jsonl>
   ```

3. Compare scores:
   ```bash
   python orchestration_framework/evaluation/score.py --llm_judge_path <judged.jsonl>
   ```

The evaluation framework logs detailed trajectories in JSONL format for debugging.
