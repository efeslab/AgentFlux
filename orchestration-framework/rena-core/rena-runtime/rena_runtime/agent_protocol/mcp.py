from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack
import logging
import subprocess

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

from rena_runtime.agent_protocol.base import BaseAgentProtocol
from rena_runtime.data_types import ToolParam, ToolResultBlockParam


logger = logging.getLogger("rena_runtime")


class MCPAgent(BaseAgentProtocol):
    server_path: Optional[str]
    _exit_stack: AsyncExitStack
    _session: Optional[ClientSession]

    def __init__(self, server_path: Optional[str] = None):
        self.server_path = server_path
        self._exit_stack = AsyncExitStack()
        self._session = None

    async def run_scripts(self, scripts: List[str]) -> None:
        for script in scripts:
            try:
                _run_subprocess(cmd=script.split(" "), cwd=self.server_path)
            except Exception as e:
                raise ValueError(
                    f"agent config script failed. script: {script}, error: {e}"
                ) from e

    # TODO(sean): add SSE support
    async def run(
        self, cmd: str, args: List[str], env: Optional[Dict[str, str]] = None
    ) -> None:
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=cmd, args=args, cwd=self.server_path, env=env
                    )
                )
            )

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await self._session.initialize()
        except Exception as e:
            raise ValueError(f"mcp_server run failed. error: {e}") from e

    async def list_tools(self) -> List[ToolParam]:
        if not self._session:
            raise ValueError("mcp_agent is not running.")

        tools = await self._session.list_tools()

        return [
            ToolParam(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
            )
            for tool in tools.tools
        ]

    async def call_tool(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> ToolResultBlockParam:
        if not self._session:
            raise ValueError("mcp_agent is not running.")

        tool_result = await self._session.call_tool(
            name=tool_name, arguments=tool_input
        )

        return ToolResultBlockParam(
            type="tool_result",
            content=" ".join(c.text for c in tool_result.content),
            is_error=tool_result.isError,
        )


def _run_subprocess(cmd: List[str], cwd: Optional[str]) -> None:
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"subprocess failed. error: {e.stderr}") from e
    except Exception as e:
        raise ValueError(f"subprocess failed. error: {e}") from e
