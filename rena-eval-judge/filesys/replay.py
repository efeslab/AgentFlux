import os
import shutil
import json
import tempfile
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import argparse


class Replay:
    def __init__(self, dir_path: str):
        """Copy the dir to a tmp dir"""
        if not os.path.isdir(dir_path):
            raise ValueError(f"Provided path '{dir_path}' is not a valid directory")

        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp(prefix="replay_")

        # Destination path inside the tmp directory
        dest_path = os.path.join(self.tmp_dir, os.path.basename(dir_path))

        # Copy the directory tree
        shutil.copytree(dir_path, dest_path)

        self.dir = dest_path

    def get_stat(self) -> str:
        """Return all files and their sizes in the directory"""
        stats = []
        for root, _, files in os.walk(self.dir):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                rel_path = os.path.relpath(file_path, self.dir)
                stats.append(f"{rel_path} - {size} bytes")
        ret = "\n".join(stats)
        # print(f"stat: {ret}")
        return ret

    async def _call_tool(self, message: dict, session: ClientSession):
        assert message["role"] == "assistant"
        assert "tool_calls" in message
        for tool_call in message["tool_calls"]:
            name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]

            # If arguments is a string, parse it
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            # Replace any path references
            def replace_workspace(obj):
                if isinstance(obj, str):
                    return obj.replace("/workspace", self.dir)
                elif isinstance(obj, dict):
                    return {k: replace_workspace(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_workspace(x) for x in obj]
                return obj

            arguments = replace_workspace(arguments)

            try:
                result = await session.call_tool(
                    name=name,
                    arguments=arguments  # ✅ pass dict, not string
                )
                print(f"Called tool '{name}' with arguments {arguments}. Result: {result.content[0].text if result.content else 'No content'}")
            except Exception as e:
                print(f"Error calling tool '{name}' {arguments}: {e}")

    async def replay(self, messages: list):
        print(f"Initial stat: \n{self.get_stat()}")
        initial_stat = self.get_stat()

        # Set up server parameters
        server_params = StdioServerParameters(
            command="npx",
            args=["@modelcontextprotocol/server-filesystem", self.dir]  # current directory
        )
        
        # Connect and test
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()
                print("✓ Connected to filesystem MCP server")

                for message in messages:
                    if message["role"] == "assistant" and "tool_calls" in message:
                        await self._call_tool(message, session)

        # print(f"Ending stat: \n{self.get_stat()}")
        ending_stat = self.get_stat()
        return f"Initial stat:\n{initial_stat}\n\nEnding stat:\n{ending_stat}"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Directory to replay")
    parser.add_argument("trajs", type=str, help="JSON file with trajectories")
    args = parser.parse_args()
    
    with open(args.trajs, "r") as f:
        trajs = [json.loads(line) for line in f.readlines()]
    
    # replay = Replay(args.dir)
    for traj in trajs:
        replay = Replay(args.dir)
        await replay.replay(traj["messages"])

if __name__ == "__main__":
    asyncio.run(main())
