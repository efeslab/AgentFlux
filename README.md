# AgentFlux: Framework for Privacy-Preserving On-Device Agentic Systems

<div align="center">

**A framework for optimizing Large Language Model tool-calling through dual-stage finetuning and intelligent routing**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

[Features](#-key-features) •
[Architecture](#-architecture) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •

</div>

---

## 🎯 Overview

AgentFlux is a novel framework that significantly improves LLM tool-calling performance through a **two-stage optimization approach**: specialized classifier models for tool selection, and individual tool adapter models for precise argument generation. This dual-optimization strategy achieves superior accuracy while maintaining cost-efficiency compared to traditional monolithic approaches.

### Why AgentFlux?

Modern LLM applications increasingly rely on tool-calling capabilities to interact with external APIs, databases, and services. However, traditional approaches face several challenges:

- **Inefficient Tool Selection**: Large models waste compute on simple routing decisions
- **Suboptimal Argument Generation**: Generic models struggle with tool-specific parameter formatting
- **High Latency & Cost**: Every request requires full model inference
- **Poor Scalability**: Adding new tools degrades performance across all tools

AgentFlux solves these problems by **separating concerns**: we build a decoupled fine-tuning framework called DualTune that creates a lightweight classifier rapidly selects the appropriate tool, then a specialized adapter generates precise arguments. 

---

## 🚀 Key Features

### 🎓 DualTune: Automated Decoupled Finetuning Pipeline

- **Synthetic Data Generation**: Automatically generate high-quality training data using GPT-5
- **Intelligent Data Validation**: Comprehensive argument validation and trajectory cleaning
- **Dual-Model Training**: Simultaneous training of classifier and tool-specific adapters
- **Built on Unsloth**: Leverages state-of-the-art LoRA optimization

### ⚡ AgentFlux Inference Framework

- **Smart Routing**: FastAPI-based proxy with intelligent request classification
- **Tool Specialization**: Per-tool finetuned models for optimal argument generation

### 🔬 Comprehensive Evaluation Suite

- **Built on Rena Core**: Production-grade orchestration framework (Rust + Python)
- **Multi-Category Benchmarks**: Evaluate across filesys, Monday.com, Notion, and custom MCP tools
- **Automated Judging**: LLM-based evaluation with ground truth comparison
- **Detailed Metrics**: Track accuracy, latency, and cost metrics across the pipeline

---

## 📐 Architecture

AgentFlux consists of three integrated components working in harmony:

```
┌─────────────────────────────────────────────────────────────────┐
│                         AgentFlux System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────┐   │
│  │    DualTune      |   |                  |   |            |   |
|  |   Finetuning     │   │   AgentFlux      │   │    Rena    │   │
│  │    Pipeline      │──▶│   Inference      │──▶│    Core    │   │
│  │                  │   │                  │   │  (Eval)    │   │
│  └──────────────────┘   └──────────────────┘   └────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1. DualTune Finetuning Pipeline

Generates optimized models through a multi-stage process:

```
Query Generation → Trajectory Collection → Data Cleaning → Model Training
    (GPT-5)           (GPT-5)        (Validation)    (Unsloth LoRA)
```

**Components**:
- **Query Generator**: Creates diverse, realistic user queries per tool category
- **Trajectory Collector**: Records complete tool-calling conversations with a frontier model (GPT-5)
- **Data Processor**: Validates arguments, checks types, splits train/eval/test sets
- **Model Trainer**: Finetunes both classifier and per-tool adapters using LoRA

**Technical Specifications**:
- Training Method: LoRA (rank=32, alpha=64, dropout=0.1)
- Optimization: RSLoRA with AdamW-8bit optimizer
- Context Length: 32,768 tokens
- Learning Rate: 5e-6 with cosine scheduling

### 2. AgentFlux Inference Framework

Production-ready inference system with intelligent routing:

```
User Request → Proxy Server → Classifier → Tool Adapter → API Response
                   ↓              ↓             ↓
              [Tool Cap]    [Tool Select]  [Args Gen]
```

**Request Flow**:
1. **Proxy Layer** (`proxy.py`): FastAPI server receives chat completion requests
2. **Classification** (`classifier.py`): Lightweight model predicts tool from context (n=10 samples, temperature=1.0)
3. **Adaptation** (`tool_adaptor.py`): Specialized model generates precise arguments
4. **Execution**: Formatted tool call sent to target API

**Key Innovations**:
- **Multi-Sample Classification**: Aggregates predictions from 10 samples for robustness
- **Tool-Specific Chat Templates**: Custom Jinja2 templates optimize each tool's behavior

### 3. Rena Core Orchestration Framework

Production-grade evaluation infrastructure:

```
┌─────────────────┐         ┌──────────────────┐
│  rena-browserd  │◄───────▶│  rena-runtime    │
│   (Rust Core)   │  gRPC   │  (Python MCP)    │
│                 │         │                  │
│ Process Manager │         │  Tool Execution  │
│ Docker Control  │         │  MCP Protocol    │
└─────────────────┘         └──────────────────┘
```

**Components**:
- **rena-browserd** (Rust): Manages MCP server lifecycle, Docker containers, process orchestration
- **rena-runtime** (Python): Executes MCP protocol, handles tool invocations, logs trajectories
- **browserd-cli**: Command-line interface for running queries and managing apps
- **browserd-eval**: Evaluation harness for benchmarking tool-calling performance

**Evaluation Pipeline**:
1. Query Generation → Category-specific test queries
2. Trajectory Generation → Run queries through AgentFlux or baseline
3. Automated Judging → Compare outputs against ground truth
4. Score Calculation → Aggregate accuracy and success metrics

---

## 🏗️ Repository Structure

```
AgentFlux/
├── finetune/                           # 🎓 Model Finetuning Pipeline
│   ├── unsloth-cli-split.py            #    Main training script (Unsloth + LoRA)
│   ├── data_prepare.py                 #    Data validation & train/eval/test splitting
│   ├── gen_queries.py                  #    Synthetic query generation (GPT-5)
│   ├── gen_trajs.py                    #    Trajectory collection from baseline
│   ├── gen_tool_template.py            #    Generate Jinja2 templates per tool
│   ├── classifier.jinja                #    Chat template for classifier model
│   ├── tool_template.jinja             #    Chat template for tool adapters
│   └── results/                        #    Training outputs, logs, model checkpoints
│
├── inference/agentflux/                # ⚡ AgentFlux Inference System
│   ├── agentflux/
│   │   ├── proxy.py                    #    FastAPI proxy server (port 8030)
│   │   ├── classifier.py               #    Finetuned & GPT classifier implementations
│   │   ├── tool_adaptor.py             #    Finetuned & GPT tool adapter implementations
│   │   └── utils/logging_setup.py      #    Logging configuration
│   └── pyproject.toml                  #    Package configuration
│
├── orchestration-framework/            # 🔬 Evaluation Infrastructure
│   ├── rena-core/                      #    Core orchestration framework
│   │   ├── rena-browserd/              #    Rust: Process & Docker management
│   │   │   ├── browserd/               #       Core library
│   │   │   ├── browserd-cli/           #       CLI interface
│   │   │   └── browserd-eval/          #       Evaluation harness
│   │   └── rena-runtime/               #    Python: MCP protocol execution
│   ├── evaluation/                     #    Evaluation scripts
│   │   ├── run_agentflux.py            #    Run with AgentFlux proxy
│   │   ├── run_baseline.py             #    Run with baseline GPT
│   │   ├── gen_queries.py              #    Generate test queries
│   │   ├── score.py                    #    Calculate final metrics
│   │   ├── filesys/judge.py            #    Filesystem category judge
│   │   ├── monday/judge.py             #    Monday.com category judge
│   │   └── notion/judge.py             #    Notion category judge
│   └── __init__.py                     #    Helper functions
│
└── bash/                               # 🔧 Automation Scripts
    ├── finetune.sh                     #    Complete finetuning pipeline
    ├── finetune_classifier.sh          #    Train classifier only
    ├── finetune_tool_adaptors.sh       #    Train tool adapters only
    └── evaluate.sh                     #    Complete evaluation pipeline
```

---

## 🎬 Quick Start

### Prerequisites

#### System Requirements
- **Python**: 3.8 or higher
- **Rust**: 1.70 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **Docker**: Latest stable version (for Rena Core)

#### API Keys
- **OpenAI API Key**: Required for query/trajectory generation and baseline evaluation
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AgentFlux.git
cd AgentFlux
```

#### 2. Install DualTune Finetuning Dependencies

The finetuning pipeline requires [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA training:

```bash
# Install Unsloth (recommended: use conda/mamba)
conda create -n dualtune python=3.10
conda activate dualtune

# Install Unsloth with CUDA support
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional dependencies
pip install transformers datasets trl accelerate peft
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Install AgentFlux Inference System

```bash
cd inference/agentflux
pip install -e .
cd ../..

# Install inference dependencies
pip install fastapi uvicorn httpx aioconsole
```

#### 4. Setup Rena Core (Optional - for evaluation)

```bash
cd orchestration-framework/rena-core

# Install Rust dependencies and build
make setup

# Install Python runtime dependencies
cd rena-runtime
pip install -e .
cd ../../..
```

### Usage

#### 🎓 Finetuning Your Models

Run the complete finetuning pipeline for a category (e.g., `filesys`, `monday`, `notion`):

```bash
# Full pipeline: query gen → trajectory collection → training
bash bash/finetune.sh filesys
```

This will:
1. Generate 1000+ synthetic queries using GPT-5
2. Collect tool-calling trajectories from baseline model
3. Clean and validate data (argument checking, type validation)
4. Split into train/eval/test sets (80/10/10)
5. Train classifier model → `finetune/results/finetune_output/filesys/classifier/`
6. Train per-tool adapters → `finetune/results/finetune_output/filesys/tool_adaptors/{tool_name}/`

**Customization**:
```bash
# Custom hyperparameters: category, batch_size, grad_accumulation, epochs
bash bash/finetune_classifier.sh filesys 8 2 3
bash bash/finetune_tool_adaptors.sh filesys 8 2 3
```

**Training Outputs**:
- Model checkpoints: `finetune/results/finetune_output/{category}/`
- Training logs: `finetune/results/log/{category}/`
- Processed data: `finetune/results/trajectories/{category}/`

#### ⚡ Running AgentFlux Inference

Deploy your finetuned models via the AgentFlux proxy server:

```bash
# Start classifier model server (port 8001)
vllm serve finetune/results/finetune_output/filesys/classifier/ \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Start tool adapter servers (one per tool)
vllm serve finetune/results/finetune_output/filesys/tool_adaptors/read_file/ \
  --port 8002 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Start AgentFlux proxy (requires config files)
python -c "
from agentflux import start_proxy
from agentflux.proxy import ProxyConfig

config = ProxyConfig(
    port=8030,
    tool_list='config/filesys/tool_list.json',
    classifier_name_or_path='config/filesys/classifier.json',
    tool_adaptor_name_or_path='config/filesys/tool_adapters.json'
)
start_proxy(config)
"
```

Now send requests to `http://localhost:8030/v1/chat/completions` using OpenAI SDK format!

**Example Request**:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8030/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="agentflux",  # Model name doesn't matter, proxy handles routing
    messages=[
        {"role": "user", "content": "Read the contents of README.md"}
    ],
    tools=[...]  # Your MCP tool definitions
)

print(response.choices[0].message.tool_calls)
```

#### 🔬 Evaluation

Benchmark your finetuned models against baseline:

```bash
# Set workspace for filesys category
export WORKSPACE=/path/to/test/workspace

# Run complete evaluation pipeline
bash bash/evaluate.sh filesys
```

This executes:
1. Query generation for test set
2. Trajectory generation using AgentFlux proxy
3. Automated judging against ground truth
4. Score calculation and metrics reporting

**Output**:
- Trajectories: `orchestration-framework/evaluation/{category}/eval-results/`
- Judgments: `orchestration-framework/evaluation/{category}/judge-results/`
- Final scores printed to console

**Manual Evaluation Steps**:
```bash
# 1. Generate test queries
python orchestration-framework/evaluation/gen_queries.py --category filesys

# 2. Run with AgentFlux
python orchestration-framework/evaluation/run_agentflux.py filesys \
  --classifier config/filesys/classifier.json \
  --tool_adapters config/filesys/tool_adapters.json \
  --query orchestration-framework/evaluation/filesys/queries/fuzzing_queries.txt \
  --output orchestration-framework/evaluation/filesys/eval-results/trajs.jsonl

# 3. Judge results
python orchestration-framework/evaluation/filesys/judge.py \
  --trajs orchestration-framework/evaluation/filesys/eval-results/trajs.jsonl \
  --output orchestration-framework/evaluation/filesys/judge-results/judged.jsonl

# 4. Calculate scores
python orchestration-framework/evaluation/score.py \
  --llm_judge_path orchestration-framework/evaluation/filesys/judge-results/judged.jsonl
```

---

## 📚 Documentation

### Configuration Files

Each tool category requires configuration in `config/{category}/`:

- **`tool_list.json`**: MCP tool definitions (OpenAI format)
- **`classifier.json`**: Classifier model endpoint and configuration
  ```json
  {
    "model": "filesys-classifier",
    "port": 8001,
    "tools": ["read_file", "write_file", "list_directory", ...]
  }
  ```
- **`tool_adapters.json`**: Per-tool adapter configurations
  ```json
  {
    "read_file": {"model": "read_file-adapter", "port": 8002},
    "write_file": {"model": "write_file-adapter", "port": 8003},
    ...
  }
  ```
- **`query_generation_template.txt`**: Prompt template for generating training queries
- **`judge_sys_prompt.txt`**: System prompt for evaluation judging

### Adding a New Tool Category

1. **Create Configuration**:
   ```bash
   mkdir -p config/my_category
   # Add tool_list.json, templates, etc.
   ```

2. **Generate Training Data**:
   ```bash
   bash bash/finetune.sh my_category
   ```

3. **Create Judge Script**:
   ```python
   # orchestration-framework/evaluation/my_category/judge.py
   # Implement category-specific validation logic
   ```

4. **Run Evaluation**:
   ```bash
   bash bash/evaluate.sh my_category
   ```

### Advanced Configuration

**Finetuning Hyperparameters** (`bash/finetune_classifier.sh`, `bash/finetune_tool_adaptors.sh`):
- `batch_size`: Per-device batch size (default: 4)
- `accumulate_step`: Gradient accumulation steps (default: 4)
- `num_train_epochs`: Training epochs (default: 4)
- Learning rate: 5e-6 (fixed in script)
- Scheduler: Cosine annealing

**Data Validation** (`finetune/data_prepare.py`):
- Validates required vs optional parameters
- Type checking (string, number, boolean, array, object)
- Rejects unexpected arguments
- Filters conversations exceeding 32,768 tokens
- Binary search removal of problematic entries

**Caching Behavior** (`inference/agentflux/tool_adaptor.py`):
- Tracks up to 5 unique function calls per request
- Blocks after 10 identical calls (prevents loops)
- Clears cache when "summarize" tool is called
- Uses SHA256 hashing for call deduplication

---

## 🛠️ Technical Details

### Finetuning Stack

- **Base Model**: Qwen2.5-7B-Instruct (32K context)
- **Framework**: Unsloth (optimized Hugging Face Transformers)
- **Method**: LoRA (Low-Rank Adaptation)
  - Rank: 32
  - Alpha: 64 (scaling factor)
  - Dropout: 0.1
  - Target modules: Q, K, V, O, Gate, Up, Down projections
- **Optimizer**: AdamW-8bit (memory efficient)
- **Scheduler**: Cosine annealing with warmup
- **Training**: RSLoRA enabled, gradient checkpointing (Unsloth mode)

### AgentFlux Architecture

**Proxy Server** (`proxy.py`):
- FastAPI application on port 8030
- OpenAI-compatible `/v1/chat/completions` endpoint
- Automatic tool list substitution
- Error handling and logging

**Classification Strategy** (`classifier.py`):
- Generates 10 completions with temperature=1.0
- Extracts tool names from `<tool_call>` tags
- Votes by frequency (most common tool wins)
- Falls back to "summarize" if no tools detected

**Adaptation Strategy** (`tool_adaptor.py`):
- Single tool selection via `tool_choice` parameter
- Custom chat templates per tool
- Retry logic (max 3 attempts)
- Response validation and error handling

### Rena Core Implementation

**browserd** (Rust):
- Docker API integration (bollard)
- Async process management (tokio)
- gRPC server for runtime communication
- Structured logging (tracing)

**rena-runtime** (Python):
- MCP protocol implementation (mcp SDK)
- Tool invocation and response handling
- Trajectory logging (JSONL format)
- Container lifecycle management

---

## 🤝 Contributing

We welcome contributions! Here are some areas where you can help:

- **New Tool Categories**: Add support for additional MCP tool sets (GitHub, Slack, Google Drive, etc.)
- **Evaluation Metrics**: Implement new judging criteria and success metrics
- **Model Architectures**: Experiment with different base models and training techniques
- **Optimization**: Improve inference speed, memory usage, or training efficiency
- **Documentation**: Enhance guides, add tutorials, create example notebooks

**To Contribute**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows existing style conventions
- Tests pass (if applicable)
- Documentation is updated
- Commit messages are descriptive

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/efeslab/AgentFlux/issues)
- **Discussions**: [GitHub Discussions](https://github.com/efeslab/AgentFlux/discussions)
- **Email**: rohankad@cs.washington.edu

---

## 🌟 Citation

If you use AgentFlux in your research, please cite:

```bibtex
@article{kadekodi2025dualtune,
  title={DualTune: Decoupled Fine-Tuning for On-Device Agentic Systems},
  author={Kadekodi, Rohan and Jin, Zhan and Kamahori, Keisuke and Gu, Yile and Khatiri, Sean and Bayindirli, Noah H and Gorbunov, Sergey and Kasikci, Baris},
  journal={arXiv preprint arXiv:2510.00229},
  year={2025}
}
```

---

<div align="center">

**⭐ Star us on GitHub if AgentFlux helps your project! ⭐**

</div>
