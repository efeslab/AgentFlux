#!/bin/bash

set -euo pipefail

PYTHON_VERSION="python3.11"
USE_CASES_PATH="${USE_CASES_PATH:-examples/use-cases}"
RESULTS_DIR="${USE_CASES_PATH}/.generated/evals/results"
VENV_DIR=".build/venv"
PROTO_DIR="./proto"
RUNTIME_PROTO_DIR="./rena-runtime/rena_runtime/proto"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_dependency() {
    local cmd=$1
    local help_msg=$2
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo -e "${RED}Error: ${cmd} is not installed.${NC}"
        echo "$help_msg"
        exit 1
    fi
}

success_msg() {
    echo -e "${GREEN}✓ $1${NC}"
}

error_msg() {
    echo -e "${RED}✗ $1${NC}"
}

process_msg() {
    echo -e "${YELLOW}$1${NC}"
}

if [[ $# -ne 1 ]] || [[ "$1" != "run" ]]; then
    echo "Usage: ./eval.sh run"
    exit 1
fi

echo "Checking dependencies..."
check_dependency "$PYTHON_VERSION" "Please install Python 3.11 (e.g. via pyenv)"
check_dependency "docker" "Please install Docker from https://docker.com"

if ! docker ps >/dev/null 2>&1; then
    error_msg "Docker daemon is not running"
    echo "Please start Docker Desktop or the Docker daemon"
    exit 1
fi

check_dependency "cargo" "Please install Rust from https://rustup.rs"
success_msg "All dependencies satisfied"

echo "Running evaluations..."
mkdir -p "$RESULTS_DIR"

if [[ ! -f .env ]]; then
    error_msg ".env file not found. API key env vars are mandatory for run_eval step."
    exit 1
fi

source .env

inference_type=$(grep -A 3 "\[inference_engine\]" "$USE_CASES_PATH/browserd.toml" | grep "^type" | cut -d"=" -f2 | tr -d " " | tr -d '"')
inference_model=$(grep -A 3 "\[inference_engine\]" "$USE_CASES_PATH/browserd.toml" | grep "^model" | cut -d"=" -f2 | tr -d " " | tr -d '"')
max_tokens=$(grep -A 3 "\[inference_engine\]" "$USE_CASES_PATH/browserd.toml" | grep "^max_tokens" | cut -d"=" -f2 | tr -d " ")

output_filename="${inference_type}-${inference_model}-${max_tokens}_results.json"

if cd rena-browserd && \
    cargo run --bin browserd-cli --release -- \
        --config "../$USE_CASES_PATH/browserd.toml" \
        eval "../$USE_CASES_PATH/eval.toml" \
        --format json \
        --output "../$RESULTS_DIR/$output_filename" && \
    cd ..; then
    success_msg "Evaluation complete"
    success_msg "  Results saved to: $RESULTS_DIR/$output_filename"
else
    error_msg "Evaluation failed"
fi
