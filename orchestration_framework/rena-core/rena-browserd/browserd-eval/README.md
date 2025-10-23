# Browserd Eval

An evaluation framework for evaluating browserd use cases with support for LLM-generated trace explanations.

## Configuration

One need to provide two config files:

- `browserd.toml`: contains the browserd config (e.g., which mcp servers to run). Please refer to [browserd-cli README](../browserd-cli/README.md) for more details.
- `eval.toml`: contains the evaluation config (e.g., which queries to run and how to evaluate the results)

Below is the eval config file format with explanations for each entry:

```toml
name=[name of the user flow]
description=[description of the user flow]
runs=[default number of times to run each eval (optional, defaults to 1)]

[[evaluators]]
type=[type of the evaluator (e.g., basic_match, llm)]
id=[id of the evaluator]
base_prompt=[base prompt for the evaluator]

[evaluators.inference_engine]
type=[provider of the inference engine (e.g., anthropic, ollama, openai)]
model=[model name to run as evaluator]
max_tokens=[max tokens to generate]
api_key=[api key for anthropic inference engine]

[[evals]]
query=[query to run]
ideal=[ideal response]
evaluator=[id of the evaluator]
runs=[number of times to run this specific eval (optional, overrides global runs)]
```

**Note on Multiple Runs**: When `runs` > 1 is specified (either globally or per-eval), all runs execute in parallel for improved performance. The system aggregates results including success rates and latency statistics (mean, min, max, standard deviation).

## Example Config

```toml
name = "hello-world-greeting"
description = "a simple evaluation of hello-world greeting app with match and llm evaluator"
runs = 5  # Run all evals 5 times by default

[[evaluators]]
type = "basic_match"
id = "basic-match"

[[evaluators]]
type = "llm"
id = "basic-llm"
base_prompt = "You are an expert evaluator for AI agent responses. Your task is to evaluate if the actual response matches the expected response for a given query."

[evaluators.inference_engine]
type = "anthropic"
model = "claude-sonnet-4-20250514"
max_tokens = 512
api_key = "<your-anthropic-api-key>"

[[evals]]
query = "say hello to Bob"
ideal = "Hello, Bob!"
evaluator = "basic-match"
# Uses global runs = 5

[[evals]]
query = "say hello to Bob"
ideal = "at lease one call of say_hello tool of hello_world mcp server"
evaluator = "basic-llm"
runs = 10  # Override: run this eval 10 times
```

## Trace Explanations

The evaluation framework can generate human-readable explanations of execution traces using LLMs. This helps understand what happened during each run and identify patterns across multiple runs.

See [TRACE_EXPLANATIONS.md](./TRACE_EXPLANATIONS.md) for detailed documentation on configuring and using trace explanations.

## Usage

Use the `browserd-cli` with the `eval` subcommand (from `rena-browserd` directory):

```
cargo run --bin browserd-cli --release -- --config <path-to-browserd-config> eval <path-to-eval-config> --format <json|csv> --output <path-to-output-file>
```

## Use Cases Evaluation

Prerequisites:

- create a `.env` file in the root directory (following the `.example.env` file)
- update `<your-whoami-here>` for `cli` and `filesystem` mcp apps within `./examples/use-cases/browserd.toml`
- update mcp apps environment variables within `./examples/use-cases/browserd.toml`: `BRAVE_API_KEY`, `NOTION_API_KEY`, `MONDAY_API_KEY`
- update `HF_TOKEN` within HuggingFace-related evals (i.e., `download-gated-llama-model`, `download-public-bert-model`) within `./examples/use-cases/eval.toml`

Note: `NOTION_API_KEY`, `MONDAY_API_KEY`, `HF_TOKEN` need to be specific ones for tests to run, ask @gray-floyd or @nbayindirli for them.

To run all evals you can either directly use `browserd-cli` or use `eval.sh` script that automatically derive a results file name given your config file:

- `browserd-cli` (from `rena-browserd` directory)

    ```bash
    cargo run --bin browserd-cli --release -- --config ../examples/use-cases/hello-world-greeting/browserd.toml eval ../examples/use-cases/hello-world-greeting/eval.toml --format json --output ../examples/use-cases/.generated/evals/results/use_cases_eval_results.json
    ```

- `eval.sh` (from root directory)

    ```bash
    ./eval.sh run
    ```

To see the details of each eval, see [examples/use-cases/README.md](../../examples/use-cases/README.md).

Note: to run evaluation with `basic_orchestrator(model=lfm2-1.2B)` you need to run llama-server locally first (since ollama does not support the model yet):

```bash
<path-to-llama-server-binary>/llama-server --jinja -fa -hf LiquidAI/LFM2-1.2B-GGUF
```

Make sure download the latest release binary [here](https://github.com/ggml-org/llama.cpp/releases/tag/b5922) since instalation with brew fails for `LFM2` models. Since it spawns an openai compatible server, one can use such inference_engine config within broserd config:

```toml
[inference_engine]
type = "openai"
model = "lfm2-1.2b"
max_tokens = 2048
url = "http://127.0.0.1:8080"
```