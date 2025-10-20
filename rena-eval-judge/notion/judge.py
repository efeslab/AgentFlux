import argparse
import json
from openai import AsyncOpenAI
import asyncio
import os


async def async_judge(client: AsyncOpenAI, gt: str, traj: str, tool_description: str) -> str:
    
    def get_prompt(ground_truth: str, query: str, traj: str, tool_description: str) -> str:
        input_a = ground_truth
        input_b = tool_description
        input_c = query
        input_d = traj
        user_pt = f"Input A: {input_a}\n" + \
                f"Input B: {input_b}\n" + \
                f"Input C: {input_c}\n" + \
                f"Input D: {input_d}\n"
        return user_pt

    async def async_call_json(client: AsyncOpenAI, user_pt: str) -> str:
        sys_pt = Path(CUR_DIR.parent / "config" / "notion" / "judge_sys_prompt.txt").read_text()
        resp = await client.chat.completions.create(
            model=os.environ.get("JUDGE_MODEL", "gpt-5"),
            reasoning_effort="low",
            prompt_cache_key="please-cache-please",
            messages=[
                {"role": "system", "content": sys_pt},
                {"role": "user", "content": user_pt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "eval_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "Reasoning_ToolCoverage": {"type": "string"},
                            "Score_ToolCoverage": {"type": "integer"}
                        },
                        "required": [
                            "Reasoning_ToolCoverage",
                            "Score_ToolCoverage"
                        ],
                        "additionalProperties": False
                    },
                    "strict": True
                },
            },
        )
        # If your SDK exposes structured parsing, you can use:
        # return resp.choices[0].message.parsed
        print(f"Response received: {resp}")
        return resp.choices[0].message.content  # JSON string

    def dumps_scores_first(d, **kw):
        # Keep insertion order: Python 3.7+ preserves it for plain dicts
        scores_first = {k: d[k] for k in d if k.startswith("Score_")}
        rest = {k: d[k] for k in d if not k.startswith("Score_")}
        ordered = {**scores_first, **rest}
        # ensure we don't undo the order with alphabetical sorting
        kw.setdefault("sort_keys", False)
        return json.dumps(ordered, **kw)

    traj_json = json.loads(traj)
    query = traj_json["messages"][0]["content"]
    user_pt = get_prompt(gt, query, traj, tool_description)
    result = await async_call_json(client, user_pt)
    result = dumps_scores_first(json.loads(result))
    return result

async def async_nosum_judge(client: AsyncOpenAI, gt: str, traj: str, tool_description: str) -> str:
    try:
        traj_json = json.loads(traj)
        messages = traj_json["messages"]
        if messages[-1]["content"] == "inference engine error":
            return json.dumps({
                "Reasoning_ToolOutput": "inference engine error",
                "Score_ToolOutput": 0,
            })
        # traj = await async_gptsum(client, traj)
        traj_json["messages"] = traj_json["messages"][:-1]  # remove the last summary
        traj = json.dumps(traj_json)
        result = await async_judge(client, gt, traj, tool_description)
        return result
    except Exception as e:
        print(f"Error during async_call: {e}")
        return json.dumps({
            "Reasoning_ToolOutput": "Error occurred",
            "Score_ToolOutput": 0,
        })

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajs", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    tool_description = json.load(Path(CUR_DIR.parent / "config" / "notion" / "tool_list.json").open())
    gt = Path(CUR_DIR.parent / "config" / "notion" / "ground_truth.txt").read_text()
    trajs = [line.strip() for line in open(args.trajs).readlines()]
    if args.end is None:
        trajs = trajs[args.start:]
    else:
        trajs = trajs[args.start:args.end]
    max_concurrent = args.max_concurrent
    if args.max_concurrent is None:
        max_concurrent = len(trajs)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    coroutine_list = []
    with open(args.output, "w") as f:
        for i in range(0, len(trajs), max_concurrent):
            batch = trajs[i:i+max_concurrent]
            for traj in batch:
                coroutine_list.append(async_nosum_judge(client, gt, traj, tool_description))
            batch_results = await asyncio.gather(*coroutine_list)
            for res in batch_results:
                f.write(json.dumps(json.loads(res)) + "\n")
            f.flush()
            coroutine_list = []

if __name__ == "__main__":
    asyncio.run(main())
