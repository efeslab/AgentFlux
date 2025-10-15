import os
import sys
import socket
import subprocess
from pathlib import Path
from string import Template
from typing import List
import argparse
import tempfile
import json
import hashlib

ROOT_DIR = Path(__file__).parent.parent

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

def run_query_baseline(rena_port: int, model: str, url: str, tool_name: str, query: str) -> str:
    # get temp config file content
    template_path = ROOT_DIR / "config" / tool_name / "browserd_template_baseline.txt"
    with open(template_path, "r") as f:
        template = Template(f.read())
    mapping = {
        "RENA_PORT": rena_port,
        "MODEL": model,
        "URL": url,
        "WORKSPACE": os.environ.get("WORKSPACE", ""),
    }
    config = template.safe_substitute(mapping)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".toml") as f:
        f.write(config.encode())
        f.flush()
        config_path = Path(f.name).resolve()   # get temp config file path

        with tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl") as traj_f:
            traj_path = Path(traj_f.name).resolve()   # get temp traj file path

            # Project directory (already absolute)
            BROWSERD_DIR = ROOT_DIR / "rena-core-proxy" / "rena-core" / "rena-browserd"
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

def run_query_proxy(rena_port: int, proxy_port: int, tool_name: str, query: str) -> str:
    # get temp config file content
    template_path = ROOT_DIR / "config" / tool_name / "browserd_template.txt"
    with open(template_path, "r") as f:
        template = Template(f.read())
    mapping = {
        "RENA_PORT": rena_port,
        "PROXY_PORT": proxy_port,
        "WORKSPACE": os.environ.get("WORKSPACE", ""),
    }
    config = template.safe_substitute(mapping)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".toml") as f:
        f.write(config.encode())
        f.flush()
        config_path = Path(f.name).resolve()   # get temp config file path

        with tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl") as traj_f:
            traj_path = Path(traj_f.name).resolve()   # get temp traj file path

            # Project directory (already absolute)
            BROWSERD_DIR = ROOT_DIR / "rena-core-proxy" / "rena-core" / "rena-browserd"
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
                    if tries >=1:
                        # keep error state and not revert
                        break
            
            final_last_line = read_last_line(traj_path)
            return final_last_line

def run_baseline(tool_name, model, url, queries_path, output_path, start=None, end=None):
    rena_port, _ = get_comm_ports(2)
    queries_path = Path(queries_path)
    traj_path = Path(output_path)
    print(f"traj_path: {traj_path}")
    print(f"queries_path: {queries_path}")

    # count the lines of traj_path
    if start is None:
        if not traj_path.exists():
            start = 0
        else:
            with open(traj_path, "r") as f:
                start = len(f.readlines())
    if end is None:
        with open(queries_path, "r") as f:
            end = len(f.readlines())

    queries = queries_path.read_text().splitlines()[start:end]
    with open(traj_path, "a") as f:
        for query in queries:
            traj = run_query_baseline(rena_port, model, url, tool_name, query)
            f.write(traj + "\n")
            f.flush()

def run_proxy(tool_name, queries_path, output_path, classifier_config_path=None, tool_adapters_config_path=None, start=None, end=None):
    rena_port, proxy_port = get_comm_ports(2)
    queries_path = Path(queries_path)
    traj_path = Path(output_path)
    print(f"traj_path: {traj_path}")
    print(f"queries_path: {queries_path}")
    print(f"classifier_config_path: {classifier_config_path}")
    print(f"tool_adapters_config_path: {tool_adapters_config_path}")

    # count the lines of traj_path
    if start is None:
        if not traj_path.exists():
            start = 0
        else:
            with open(traj_path, "r") as f:
                start = len(f.readlines())
    if end is None:
        with open(queries_path, "r") as f:
            end = len(f.readlines())

    with start_proxy(
        port=proxy_port,
        classifier_config_path=classifier_config_path,
        tool_adapters_config_path=tool_adapters_config_path,
    ) as proxier:
        queries = queries_path.read_text().splitlines()[start:end]
        with open(traj_path, "a") as f:
            for query in queries:
                traj = run_query_proxy(rena_port, proxy_port, tool_name, query)
                f.write(traj + "\n")
                f.flush()
