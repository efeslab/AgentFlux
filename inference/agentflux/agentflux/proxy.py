import os
import json
import copy
import threading
import argparse
from dataclasses import dataclass
from math import inf
from typing import Optional
import fastapi
import httpx
import traceback
import uvicorn
from pathlib import Path

from .classifier import get_classifier
from .tool_adaptor import get_tool_adaptor
from .utils.logging_setup import setup_logging

import logging
logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).parent.parent

@dataclass
class ProxyConfig:
    port: int = 8030
    tool_list: str = ""
    classifier_name_or_path: str = "gpt"  # path to classifier config or name
    tool_adaptor_name_or_path: str = "gpt"  # path to tool adaptor config or name

class Proxier:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = fastapi.FastAPI()
        self._register_routes()

        self.tool_list = json.load(open(self.config.tool_list, "r"))
        assert isinstance(self.tool_list, list), "tool_list should be a list of tools"
        self.classifier = get_classifier(config.classifier_name_or_path)
        self.tool_adaptor = get_tool_adaptor(config.tool_adaptor_name_or_path)

        # server相关
        self._server = None
        self._thread = None

    def __enter__(self):
        """启动uvicorn server (后台线程)"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)

        def run_server():
            # run 会阻塞，所以放在线程里
            import asyncio
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        logger.info(f"Proxier started at http://0.0.0.0:{self.config.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭server"""
        if self._server and self._server.started:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Proxier stopped.")
    
    async def classify(self, req_payload: dict) -> str:
        return await self.classifier.classify(req_payload)
    
    async def tool_adaption(self, req_payload: dict, tool_name: str) -> httpx.Response:
        return await self.tool_adaptor.adapt(req_payload, tool_name)
    
    def _register_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: fastapi.Request):
            try:
                req_payload = await request.json()
                
                # Used for substitue tool names. Note that it should be done before tool_cap
                req_payload["tools"] = self.tool_list
                
                logger.info(f"Received request messages: {json.dumps(req_payload['messages'])}")
                tool_name = await self.classify(req_payload)
                logger.info(f"Classified tool name: {tool_name}")
                response = await self.tool_adaption(req_payload, tool_name)

                # used for substitute tool names
                Substitute_tool_list = {
                    "get_users_by_name": "list_users_and_teams"
                }
                response_json = response.json()
                for tool_call in response_json["choices"][0]["message"].get("tool_calls", []):
                    if tool_call["function"]["name"] in Substitute_tool_list:
                        tool_call["function"]["name"] = Substitute_tool_list[tool_call["function"]["name"]]
                response = httpx.Response(response.status_code, json=response_json)

                logger.info(f"Tool adaptation response: {json.dumps(response.json())}")
                return fastapi.responses.JSONResponse(
                    status_code=response.status_code,
                    content=response.json()
                )
            except Exception as e:
                logger.error(f"Error when processing request: {json.dumps(req_payload)}")
                logger.error(f"Exception occurred: {str(e)}")
                traceback.print_exc()
                return fastapi.responses.JSONResponse(
                    status_code=500, content={"error": str(e)}
                )
       
def start_proxy(
    port: int, 
    category: str, 
    tool_list: str,
    classifier: Optional[str] = None, 
    tool_adapters: Optional[str] = None,
) -> Proxier:
    classifier_name_or_path = classifier if classifier else "gpt"
    tool_adaptor_name_or_path = tool_adapters if tool_adapters else "gpt"

    log_file_path = str(PACKAGE_ROOT / "logs" / category / f"{int(classifier is not None)}_{int(tool_adapters is not None)}.log")

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    print(f"log_file_path: {log_file_path}")
    setup_logging(str(log_file_path))

    config = ProxyConfig(
        port=port,
        tool_list=tool_list,
        classifier_name_or_path=classifier_name_or_path,
        tool_adaptor_name_or_path=tool_adaptor_name_or_path,
    )
    return Proxier(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str)
    parser.add_argument("--classifier", type=str, default=None)
    parser.add_argument("--tool_adapters", type=str, default=None)
    parser.add_argument("--tool_list", type=str, required=True)
    parser.add_argument("--port", type=int, default=8030, help="Port to run the proxy server on.")
    args = parser.parse_args()

    with start_proxy(
        port=args.port,
        category=args.category,
        tool_list=args.tool_list,
        classifier=args.classifier,
        tool_adapters=args.tool_adapters,
    ) as proxier:
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Shutting down proxy server...")
