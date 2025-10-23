from typing import Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource


class HelloWorldServer:
    def say_hello(self, name: str) -> str:
        """Says hello to the given name"""
        return f"Hello, {name}!"


async def serve() -> None:
    server = Server("mcp-hello-world")
    hello_world_server = HelloWorldServer()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="say_hello",
                description="Say hello to the given name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name to greet",
                        },
                    },
                    "required": ["name"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Handle tool calls for hello_world queries."""
        try:
            match name:
                case "say_hello":
                    if "name" not in arguments:
                        raise ValueError("Missing 'name' argument")

                    result = hello_world_server.say_hello(arguments.get("name"))
                case _:
                    raise ValueError(f"Unknown tool: {name}")

            return [TextContent(type="text", text=result)]

        except Exception as e:
            raise ValueError(f"Error processing mcp-server-hello-world query: {str(e)}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)
