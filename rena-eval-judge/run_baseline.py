import os
import sys
import socket
import subprocess
from pathlib import Path
from string import Template
from typing import List
import argparse
from rena_proxy import start_proxy
import tempfile
import json
import hashlib


EVAL_ROOT = Path(__file__).resolve().parent.parent

def get_comm_ports(n: int = 2) -> List[int]:
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        ports.append(port)
    return ports

def read_last_line(path: Path) -> str:
    lines = path.read_text().splitlines()
    if lines:
        return lines[-1]
    return ""

def pop_last_line(path: Path):
    lines = path.read_text().splitlines()
    if lines:
        last_line = lines.pop()
        path.write_text("\n".join(lines) + "\n")
        return last_line
    return None


def run_query(rena_port: int, model: str, url: str, tool_name: str, query: str) -> str:
    
    # get temp config file content
    template_path = EVAL_ROOT / tool_name / "browserd_template_baseline.txt"
    with open(template_path, "r") as f:
        template = Template(f.read())
    # config = template.substitute(RENA_PORT=rena_port, MODEL=model, URL=url)
    mapping = {
        "RENA_PORT": rena_port,
        "MODEL": model,
        "URL": url,
        "WORKSPACE": os.environ.get("WORKSPACE", ""),  # 默认空字符串
    }
    config = template.safe_substitute(mapping)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".toml") as f:
        f.write(config.encode())
        f.flush()
        config_path = Path(f.name).resolve()   # get temp config file path

        with tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl") as traj_f:
            traj_path = Path(traj_f.name).resolve()   # get temp traj file path

            # Project directory (already absolute)
            BROWSERD_DIR = Path("/home/zhan/rena/rena-core/rena-browserd").resolve()
            os.chdir(BROWSERD_DIR)

            # Log path
            env = os.environ.copy()
            env["RENA_LOG_NAME"] = str(Path(traj_path).resolve())

            print(f"config path: {config_path}")
            print(f"query: {query}")
            print(f"jsonl traj path: {traj_path.resolve()}")

            tries = 0
            while True:
                try:
                    subprocess.run(
                        [
                            "cargo",
                            "run",
                            "--bin",
                            "browserd-cli",
                            "--release",
                            "--",
                            "--config",
                            str(config_path),
                            "query",
                            query,
                        ],
                        check=True,
                        env=env,
                    )
                    last_line = read_last_line(traj_path)
                    if json.loads(last_line)["messages"][-1]["content"] == "inference engine error":
                        raise Exception("inference engine error")
                    break  # success, exit the retry loop
                except Exception as e:
                    print(f"Encountered error: {e}, retrying...{tries+1}/3")
                    tries += 1
                    if tries >=3:
                        # keep error state and not revert
                        break
            
            final_last_line = read_last_line(traj_path)
            return final_last_line



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tool_name", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("url", type=str)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    rena_port, _ = get_comm_ports(2)
    queries_path = Path(args.query)
    traj_path = Path(args.output)
    print(f"traj_path: {traj_path}")
    print(f"queries_path: {queries_path}")

    # count the lines of traj_path
    if args.start:
        start = args.start
    else:
        if not traj_path.exists():
            start = 0
        else:
            with open(traj_path, "r") as f:
                start = len(f.readlines())
    with open(queries_path, "r") as f:
        end = len(f.readlines()) if args.end is None else args.end

    queries = queries_path.read_text().splitlines()[start:end]
    with open(traj_path, "a") as f:
        for query in queries:
            traj = run_query(rena_port, args.model_name, args.url, args.tool_name, query)
            f.write(traj + "\n")
            f.flush()
