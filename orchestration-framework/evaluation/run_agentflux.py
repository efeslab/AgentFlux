import argparse
from orchestration_framework import run_agentflux


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category", type=str)
    parser.add_argument("--classifier", type=int, default=None)
    parser.add_argument("--tool_adapters", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_agentflux(
        category=args.category,
        queries_path=args.query,
        output_path=args.output,
        classifier_config_path=args.classifier,
        tool_adapters_config_path=args.tool_adapters,
        start=args.start,
        end=args.end,
    )
