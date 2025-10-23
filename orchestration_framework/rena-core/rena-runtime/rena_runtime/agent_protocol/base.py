from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from rena_runtime.data_types import ToolParam, ToolResultBlockParam


class BaseAgentProtocol(ABC):
    """Base class for Agent Protocols (e.g., mcp)."""

    @abstractmethod
    async def run_scripts(self, scripts: List[str]) -> None:
        """Run agent config scripts before running the agent server."""

    @abstractmethod
    async def run(
        self, cmd: str, args: List[str], env: Optional[Dict[str, str]] = None
    ) -> None:
        """Run the agent server."""

    @abstractmethod
    async def list_tools(self) -> List[ToolParam]:
        """List available tools."""

    @abstractmethod
    async def call_tool(self, tool_name: str, tool_input: str) -> ToolResultBlockParam:
        """Call a specific tool."""
