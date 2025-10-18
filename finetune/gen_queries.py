# generate 1000 queries for each of the tool in tool_list

import os
import json
import argparse
from pathlib import Path
from string import Template
import asyncio
from typing import Coroutine
from openai import AsyncOpenAI

async def async_call(client: AsyncOpenAI, user_request: str) -> str:
    resp = await client.responses.create(
        model="gpt-5",
        input=user_request,
    )
    return resp.output_text

async def gen_queries_per_tool(client: AsyncOpenAI, tool_name: str, global_tools: str, prompt_template: Template) -> str:
    prompt = prompt_template.substitute(TOOL_NAME=tool_name, GLOBAL_TOOLS=global_tools)
    response = await async_call(client, prompt)
    return response.strip()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated queries")
    args = parser.parse_args()

    tool_list_path = Path("config") / args.category / "tool_list.json"
    prompt_template_path = Path("config") / args.category / "query_generation_template.txt"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tool_list = json.load(open(tool_list_path, "r"))
    prompt_template = Template(open(prompt_template_path, "r").read())
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    tasks: list[Coroutine] = []
    for tool in tool_list:
        name = tool["function"]["name"]
        global_tools = json.dumps(tool_list)
        tasks.append(gen_queries_per_tool(client, name, global_tools, prompt_template))
    results = await asyncio.gather(*tasks)

    with open(output_path, "w") as f:
        for query in results:
            f.write(query + "\n")

if __name__ == "__main__":
    asyncio.run(main())

