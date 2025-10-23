import argparse
from orchestration_framework import run_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("url", type=str)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    run_baseline(
        category=args.category,
        model=args.model,
        url=args.url,
        queries_path=args.query,
        output_path=args.output,
        start=args.start,
        end=args.end,
    )
