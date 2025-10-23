# Browserd CLI

A command line interface to interact with the Browserd library and run evaluations.

## Configuration

One needs to create a config.toml file in `rena-browserd` directory. Below is the config file format with explanations for each entry:

```toml
container_comm_host=[container_comm_host defines the gRPC server host for apps (running inside containers) to connect to (manual browserd-app gRPC transport)]
container_comm_port=[container_comm_port defines the gRPC server port for apps to connect to]
runtime_path=[local path to the runtime code]

[inference_engine]
type=[provider of the inference engine, e.g. ollama, anthropic, openai]
model=[model name to run inference]
max_tokens=[max tokens to generate]
api_key=[only applicable to anthropic inference engines]

[discovery]
type=[how to discover apps, currently only static available, which runs all specified apps below]

[[discovery.apps]]
name=[name of the app]
path=[(optional) path to the app (e.g., local file path, Github URL, or Rena registry URL)]
agent_protocol=[protocol used by the app (e.g., mcp)]
runtime=[runtime type (e.g., python, nodejs)]
command=[command to run the app]
args=[arguments to pass to the app]
env=[(optional) environment variables to pass to the app]
scripts=[(optional) scripts to run before running the app]

(optional) [registry]
type=[type of the registry, only needed for Rena registry (e.g., rena)]
url=[URL of the registry]
```

### Example Config

An example config:

```toml
container_comm_host = "127.0.0.1"
container_comm_port = 50051
runtime_path = "../rena-runtime"

[inference_engine]
type = "ollama"
model = "qwen3:8b"
max_tokens = 4096

[discovery]
type = "static"

[[discovery.apps]]
name = "repomix"
agent_protocol = "mcp"
runtime = "nodejs"
command = "npx"
args = ["-y", "repomix", "--mcp"]

[[discovery.apps]]
name = "obsidian"
path = "https://github.com/MarkusPfundstein/mcp-obsidian/tree/main"
agent_protocol = "mcp"
runtime = "python"
command = "uvx"
args = ["--from=.", "mcp-obsidian"]
env = { "OBSIDIAN_API_KEY" = "<your-obsidian-api-key>", "OBSIDIAN_HOST" = "host.docker.internal" }
scripts = ["pip install ."]

[[discovery.apps]]
name = "filesystem"
agent_protocol = "mcp"
runtime = "nodejs"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]

[[discovery.apps]]
name = "context7"
agent_protocol = "mcp"
runtime = "nodejs"
command = "npx"
args = ["-y", "@upstash/context7-mcp"]

[[discovery.apps]]
name = "brave-search"
agent_protocol = "mcp"
runtime = "nodejs"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-brave-search"]
env = {"BRAVE_API_KEY" = "<your_api_key>"}

[[discovery.apps]]
name = "browser-use"
agent_protocol = "mcp"
runtime = "python"
command = "uvx"
args = ["mcp-server-browser-use@latest"]
scripts = ["pip install playwright", "python -m playwright install-deps", "python -m playwright install"]

[[discovery.apps]]
name = "git"
agent_protocol = "mcp"
runtime = "python"
command = "uvx"
args = ["mcp-server-git"]
scripts = ["apt-get update", "apt-get install -y git"]
```

## Usage

Prerequisites:
- Rena-core setup (following [rena-core README](../../README.md))
- Docker

One can run the CLI in three modes:
- Query: run a specific query with pre-defined config
- Eval: run evaluation suites to test MCP functionality
- Read-Eval-Print-Loop (REPL): interactively query browserd

### Query

To run a specific query, use the following command (from `rena-browserd` directory):

```bash
cargo run --bin browserd-cli --release -- query <query-input>
```

By default when providing multiple `query-input`s, they will be run concurrently:

```bash
cargo run --bin browserd-cli --release -- query <query-input-1> <query-input-2> ... <query-input-n>
```

Or if the `query-input` is the same, and the goal is to test concurrency:

```bash
cargo run --bin browserd-cli --release -- query --parallel <num-concurrent-queries> <query-input>
```

### Evaluation

Please refer to [browserd-eval README](../browserd-eval/README.md) for more details.

### (TODO) REPL

To run the REPL, use the following command (from `rena-browserd` directory):

```bash
cargo run --bin browserd-cli --release -- repl
```

## Config Detail

By default, pypi and npmjs registry are used to fetch and setup the app automatically (through `uvx` and `npx`). However, one can provide additional registries with custom setup scripts:
- Github:
  ```toml
  [[discovery.apps]]
  name = "time"
  path = "https://github.com/modelcontextprotocol/servers/tree/main/src/time"
  agent_protocol = "mcp"
  runtime = "python"
  command = "python"
  args = ["-m", "mcp_server_time", "--local-timezone=America/New_York"]
  scripts = ["pip install ."]
  ```
- Local:
  ```toml
  [[discovery.apps]]
  name = "hello_world"
  path = "file:///users/grayfloyd/Desktop/rena-browser/rena-example-apps/hello_world"
  agent_protocol = "mcp"
  runtime = "python"
  command = "python"
  args = ["-m", "mcp_server_hello_world"]
  scripts = ["pip install ."]
  ```
- Rena Registry:
  ```toml
  [[discovery.apps]]
  name = "hello_world"
  path = "http://127.0.0.1:50052/hello_world"
  agent_protocol = "mcp"
  runtime = "python"
  runtime = "python"
  args = ["-m", "mcp_server_hello_world"]
  scripts = ["pip install ."]

  [registry]
  type = "rena"
  url = "http://127.0.0.1:50052"
  ```
  Note: make sure the registry is already running (following [rena-browser README](https://github.com/rena-labs-ai/rena-browser?tab=readme-ov-file#quick-setup)), and the app is pushed to the registry (using [rena-cli](https://github.com/rena-labs-ai/rena-browser/tree/main/rena-cli))


Note: scripts will run within the specified app as base path.
