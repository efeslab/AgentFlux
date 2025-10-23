SHELL := /bin/bash

.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

PYTHON_VERSION := python3.11
VENV_DIR := .build/venv
PROTO_DIR := ./proto
RUNTIME_PROTO_DIR := ./rena-runtime/rena_runtime/proto

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SED_INPLACE := sed -i ''
else
    SED_INPLACE := sed -i
endif

RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

define check_dependency
	@command -v $(1) >/dev/null 2>&1 || { \
		echo -e "$(RED)Error: $(1) is not installed.$(NC)"; \
		echo "$(2)"; \
		exit 1; \
	}
endef

define success_msg
	@echo -e "$(GREEN)✓ $(1)$(NC)"
endef

define error_msg
	@echo -e "$(RED)✗ $(1)$(NC)"
endef

define process_msg
	@echo -e "$(YELLOW)$(1)$(NC)"
endef

.PHONY: help check setup setup_runtime setup_browserd cleanup test

help:
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check:
	@echo "Checking dependencies..."
	$(call check_dependency,$(PYTHON_VERSION),Please install Python 3.11 (e.g. via pyenv))
	$(call check_dependency,docker,Please install Docker from https://docker.com)
	@docker ps >/dev/null 2>&1 || { \
		$(call error_msg,Docker daemon is not running); \
		echo "Please start Docker Desktop or the Docker daemon"; \
		exit 1; \
	}
	$(call check_dependency,cargo,Please install Rust from https://rustup.rs)
	$(call success_msg,All dependencies satisfied)

setup_runtime: check
	@echo "Setting up rena-runtime..."
	@mkdir -p $(VENV_DIR) $(RUNTIME_PROTO_DIR)
	@echo "Creating virtual environment..."
	@$(PYTHON_VERSION) -m venv $(VENV_DIR)
	@source $(VENV_DIR)/bin/activate && \
		pip install --quiet --upgrade pip && \
		pip install --quiet grpcio-tools==1.70.0
	@echo "Generating protobuf files..."
	@source $(VENV_DIR)/bin/activate && \
		python -m grpc_tools.protoc \
			-I $(PROTO_DIR) \
			--python_out=$(RUNTIME_PROTO_DIR) \
			--pyi_out=$(RUNTIME_PROTO_DIR) \
			--grpc_python_out=$(RUNTIME_PROTO_DIR) \
			$(PROTO_DIR)/common.proto \
			$(PROTO_DIR)/browserd.proto
	@echo "Fixing protobuf imports..."
	@$(SED_INPLACE) 's/import browserd_pb2 as browserd__pb2/import rena_runtime.proto.browserd_pb2 as browserd__pb2/g' \
		$(RUNTIME_PROTO_DIR)/browserd_pb2_grpc.py
	@$(SED_INPLACE) 's/import common_pb2 as common__pb2/import rena_runtime.proto.common_pb2 as common__pb2/g' \
		$(RUNTIME_PROTO_DIR)/browserd_pb2.py
	$(call success_msg,rena-runtime setup complete)

setup_browserd: check
	@echo "Building rena-browserd..."
	@cd rena-browserd && cargo build --release
	$(call success_msg,rena-browserd setup complete)

setup: setup_runtime setup_browserd
	$(call success_msg,Setup complete!)

cleanup:
	@echo "Cleaning up..."
	@docker image rm -f rena_python_runtime rena_nodejs_runtime 2>/dev/null || true
	@docker image prune -f
	@rm -rf $(VENV_DIR)
	$(call success_msg,Cleanup complete)

test:
	@echo "Running tests..."
	@cd rena-browserd && cargo test
	$(call success_msg,Tests complete)
