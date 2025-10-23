# input: results/queries/{category}/all_queries.txt
# output: results/trajectories/{category}/all_trajectories.jsonl
import sys
import argparse
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())
from orchestration_framework import run_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Category name for MCP Tools")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--url", type=str, default="https://api.openai.com")
    parser.add_argument("--input_queries", type=str, required=True, help="Path to the input queries file")
    parser.add_argument("--output_trajs", type=str, required=True, help="Path to save the output trajectories in jsonl format")
    args = parser.parse_args()

    run_baseline(
        category=args.category,
        model=args.model,
        url=args.url,
        queries_path=args.input_queries,
        output_path=args.output_trajs
    )
