# mcp-bench generation
import argparse
import json
from openai import AsyncOpenAI
import asyncio
from string import Template
from pathlib import Path

CUR_DIR = Path(__file__).parent

async def async_call(client: AsyncOpenAI, prompt: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
    except Exception as e:
        print(f"Error during API call: {e}")
    return resp.choices[0].message.content


async def async_detailed_generation(client: AsyncOpenAI, tool_description: str, fs_stat: str) -> str:
    task_generation_prompt = Template(TASK_GENERATION_PROMPT_TEMPLATE).substitute(TOOL_DESCRIPTIONS=tool_description, FS_STATUS=fs_stat)
    return await async_call(client, task_generation_prompt)


async def async_fuzzing_generation(client: AsyncOpenAI, detailed_tasks: str) -> str:
    fuzzing_prompt = Template(FUZZING_PROMPT_TEMPLATE).substitute(DETAILED_TASKS=detailed_tasks)
    return await async_call(client, fuzzing_prompt)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    tool_description = Path(CUR_DIR.parent / "config" / args.category / "tool_list.json").read_text()
    status = Path(CUR_DIR.parent / "config" / args.category / "status.txt").read_text()
    output_path = CUR_DIR / args.category / "queries" / "fuzzing_queries.txt"

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    detailed_queries = await async_detailed_generation(client, tool_description, status)
    fuzzing_queries = await async_fuzzing_generation(client, detailed_queries)
    # TODO: how to ensure that the fuzzing queries has exactly 50 queries?

    with open(output_path, "w") as f:
        f.write(fuzzing_queries)

if __name__ == "__main__":
    asyncio.run(main())
