import httpx
import copy
import json
from collections import defaultdict
from math import inf
import os

import logging
logger = logging.getLogger(__name__)

class Classifier:

    async def classify(self, req_payload: dict) -> str:
        """req_payload: after tool cap
        return: tool_adapter name"""
        raise NotImplementedError
    
    def extract_tools_from_response(self, res: dict) -> list[str]:
        raise NotImplementedError
    
    def get_most_occurance_tool_name(self, res: dict) -> str:
        tool_calls = self.extract_tools_from_response(res)
        counts = defaultdict(int)
        for tool in tool_calls:
            counts[tool] += 1
        return max(counts, key=counts.get, default="summarize")

class FinetunedClassifier(Classifier):
    
    def __init__(self, model: str, port: int, tools: list[str]):
        self.model = model
        self.port = port
        self.tools = tools

        self.url = f"http://localhost:{port}/v1/chat/completions"

    @staticmethod
    def from_json_file(file_path: str) -> "FinetunedClassifier":
        with open(file_path, "r") as f:
            config = json.load(f)
        return FinetunedClassifier(
            model=config["model"],
            port=config["port"],
            tools=config["tools"]
        )

    def extract_tools_from_response(self, res: dict) -> list[str]:
        tool_names = []
        for choice in res["choices"]:
            content = choice["message"]["content"]
            try:
                tool_name = json.loads(content[len("<tool_call>"):-len("</tool_call>")])["name"]
            except Exception as e:
                logger.error(f"Failed to parse tool name from content: {content}, error: {e}")
                continue
            for tool in self.tools:
                # BUG: if there are tools like "API-get-user" and "API-get-users", it will be matched twice
                if tool == tool_name:
                    tool_names.append(tool)
                    break
        logger.info(f"Extracted tool names: {tool_names}")
        if len(tool_names) == 0:
            raise ValueError("No tool name extracted")
        return tool_names

    async def classify(self, req_payload: dict) -> str:
        # req_copy = req_payload.copy()
        req_copy = copy.deepcopy(req_payload)
        req_copy["model"] = self.model
        req_copy["n"] = 10
        req_copy["temperature"] = 1.0
        req_copy.pop("stream", None)
        req_copy.pop("stream_options", None)
        async with httpx.AsyncClient(timeout=1200000.0) as client:
            response = await client.post(self.url, json=req_copy)
        if response.status_code != 200:
            logger.error(f"FinetunedClassifier Failed: {json.dumps(response.json())}")
            raise ValueError("FinetunedClassifier Failed")

        # logger.info(f"FinetunedClassifier response: {response}")
        print(f"response: {response}")
        return self.get_most_occurance_tool_name(response.json())

class GPTClassifier(Classifier):

    def extract_tools_from_response(self, res: dict) -> list[str]:
        message = res["choices"][0]["message"]
        if message.get("tool_calls"):
            return [message["tool_calls"][0]["function"]["name"]]
        else:
            return ["summarize"]

    async def classify(self, req_payload: dict) -> str:
        req_copy = copy.deepcopy(req_payload)
        req_copy.pop("max_tokens", None)
        req_copy.pop("_workflow_patterns", None)
        req_copy.pop("tool_selection_guidelines", None)
        req_copy.pop("stream", None)
        req_copy.pop("stream_options", None)
        req_copy["model"] = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        api_key = os.environ.get("OPENAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"https://api.openai.com/v1/chat/completions"
        async with httpx.AsyncClient(timeout=12000000.0, headers=headers) as client:
            response = await client.post(url, json=req_copy)
        if response.status_code != 200:
            logger.error(f"GPTClassifier Failed: {json.dumps(response.json())}")
            raise ValueError("GPTClassifier Failed")

        return self.get_most_occurance_tool_name(response.json())

def get_classifier(classifier_name_or_path) -> Classifier:
    """classifier_name_or_path:
     - gpt
     - $PATH_TO_FINETUNED_CLASSIFIER_CONFIG_PATH
     """
    if classifier_name_or_path == "gpt":
        return GPTClassifier()
    else:
        return FinetunedClassifier.from_json_file(classifier_name_or_path)
