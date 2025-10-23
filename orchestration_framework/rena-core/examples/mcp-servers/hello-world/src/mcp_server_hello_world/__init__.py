from .server import serve


def main():
    """Hello World Time Server - Hello World functionality for MCP"""
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
