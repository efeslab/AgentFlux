import os
import json
import httpx
import copy
import aioconsole
import hashlib

import logging
logger = logging.getLogger(__name__)

def dict_hash(d: dict) -> str:
    # sort_keys=True 保证相同内容但 key 顺序不同的 dict 也能得到一致 hash
    dhash = hashlib.sha256(
        json.dumps(d, sort_keys=True).encode()
    ).hexdigest()
    return dhash

class ToolAdaptor:
    def __init__(self):
        self.cache = {}  # 10 slots for each req_hash

    async def adapt(self, req_payload: dict, tool_name: str) -> httpx.Response:
        """req_payload: after tool cap"""
        raise NotImplementedError
    
    def update_cache(self, req_hash: str, res_json: dict):
        """if same function with same arguments appear 10 times, then return false. else return true"""
        if req_hash not in self.cache:
            self.cache[req_hash] = {}
        if "tool_calls" not in res_json["choices"][0]["message"] or len(res_json["choices"][0]["message"]["tool_calls"]) == 0:  # summarize
            self.cache.pop(req_hash)
            return True
        function = res_json["choices"][0]["message"]["tool_calls"][0]["function"]
        ha = dict_hash(function)
        if ha not in self.cache[req_hash]:
            # pop the item with most counter
            if len(self.cache[req_hash]) >= 5:
                max_ha = sorted(self.cache[req_hash].items(), key=lambda x: x[1], reverse=True)[0][0]
                self.cache[req_hash].pop(max_ha)
            self.cache[req_hash][ha] = 1
        else:
            self.cache[req_hash][ha] += 1

        return self.cache[req_hash][ha] <= 10

class GPTToolAdaptor(ToolAdaptor):
    def __init__(self):
        super().__init__()

    async def adapt(self, req_payload: dict, tool_name: str) -> httpx.Response:
        req_copy = copy.deepcopy(req_payload)
        messages = req_copy.get("messages", [])
        tools = req_copy.get("tools", [])
        if tool_name == "summarize":
            messages.append({
                "role": "user",
                "content": f"You must summarize the content."
            })
            tools = []
        else:
            pass
        #     messages.append({
        #         "role": "user",
        #         "content": f"You must use the {tool_name} tool."
        #     })
            for tool in tools:
                if tool["function"]["name"] == tool_name:
                    tools = [tool]
                    # req_copy["tool_choice"] = "auto"
                    req_copy["tool_choice"] = {
                        "type": "function",
                        "function": {
                            "name": tool_name
                        }
                    }
                    break
        url = f"https://api.openai.com/v1/chat/completions"
        req_copy.pop("max_tokens", None)
        req_copy.pop("_workflow_patterns", None)
        req_copy.pop("tool_selection_guidelines", None)
        req_copy.pop("stream", None)
        req_copy.pop("stream_options", None)
        req_copy["model"] = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        req_copy["messages"] = messages
        req_copy["tools"] = tools
        api_key = os.environ.get("OPENAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 3
        tries = 0
        while True:
            tries += 1
            if tries > max_retries:
                logger.error("GPToolAdaptor: Max retries exceeded")
                raise RuntimeError("Max retries exceeded")
            
            logger.info(f"GPTTollAdaptor req_copy: {json.dumps(req_copy)}")
            
            async with httpx.AsyncClient(timeout=1200000.0, headers=headers) as client:
                response = await client.post(url, json=req_copy)
            try:
                if tool_name != "summarize":
                    response_tool = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
                    assert response_tool == tool_name, f"Expected tool name {tool_name}, but got {response_tool}."
            except Exception as e:
                logger.error(f"GPTToolAdaptor: Error occurred. Tries {tries}/{max_retries}. Response: {response.text}")
            else:
                break
        
        req_hash = dict_hash(req_payload["messages"][0]["content"])
        is_valid = self.update_cache(req_hash, response.json())
        if not is_valid:
            self.cache[req_hash] = {}
            logger.error("GPToolAdaptor: 10 same function with same arguments are been called. Drop this traj")
            raise RuntimeError("10 same function with same arguments are been called. Drop this traj")
            
        return response

class DebugToolAdaptor(ToolAdaptor):
    async def adapt(self, req_payload: dict, tool_name: str) -> httpx.Response:

        input_tool_name = await aioconsole.ainput("> tool_name: ")
        input_arguments = await aioconsole.ainput("> arguments(JSON): ")


        response = httpx.Response(200, json={
            "id": "xxx",  # just random id
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "xxx",  # just random id
                        "type": "function",
                        "function": {
                            "name": input_tool_name,
                            "arguments": input_arguments
                        }
                    }],
                }
            }]
        })
        return response

class FinetunedToolAdaptorStrict(ToolAdaptor):

    def __init__(self, adaptor_dict: dict):
        self.adaptor_dict = adaptor_dict
        super().__init__()

    async def call_gpt(self, req_payload: dict) -> httpx.Response:
        req_copy = copy.deepcopy(req_payload)
        url = f"https://api.openai.com/v1/chat/completions"
        req_copy.pop("max_tokens", None)
        req_copy.pop("_workflow_patterns", None)
        req_copy.pop("tool_selection_guidelines", None)
        req_copy["model"] = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        api_key = os.environ.get("OPENAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=1200000.0, headers=headers) as client:
            response = await client.post(url, json=req_copy)
        return response

    async def adapt(self, req_payload: dict, tool_name: str) -> httpx.Response:
        req_copy = copy.deepcopy(req_payload)
        req_copy.pop("stream", None)
        req_copy.pop("stream_options", None)
        # if tool_name == "summarize":  # call gpt directly for summarization
        #     req_copy["messages"].append({
        #         "role": "user",
        #         "content": f"You must summarize the content."
        #     })
        #     req_copy["tools"] = []
        #     response = await self.call_gpt(req_copy)
        # else:
        model = self.adaptor_dict[tool_name]["model"]
        port = self.adaptor_dict[tool_name]["port"]
        if model is None:  # sometimes the tool doesn't have a specific model because they don't have arguments
            response = httpx.Response(200, json={
                "id": "xxx",  # just random id
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "xxx",  # just random id
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": "{}"
                            }
                        }],
                    }
                }]
            })
        else:
            req_copy["model"] = model
            url = f"http://localhost:{port}/v1/chat/completions"
            if tool_name == "summarize":
                req_copy["tools"] = []
            async with httpx.AsyncClient(timeout=1200000.0) as client:
                response = await client.post(url, json=req_copy)

            # NOTE: vllm may generate id that is too long to feed to openai
            # ref: https://github.com/camel-ai/camel/issues/2215
            response_json = response.json()
            response_json["id"] = "xxx"
            try:
                if response_json["choices"][0]["message"].get("tool_calls"):
                    # response_json["choices"][0]["message"]["tool_calls"][0]["id"] = "xxx"
                    for i in range(len(response_json["choices"][0]["message"]["tool_calls"])):
                        response_json["choices"][0]["message"]["tool_calls"][i]["id"] = "xxx"
                        args = response_json["choices"][0]["message"]["tool_calls"][i]["function"]["arguments"]
                        while not isinstance(args, dict):
                            args = json.loads(args)
                        response_json["choices"][0]["message"]["tool_calls"][i]["function"]["arguments"] = json.dumps(args)
            except Exception as e:
                logger.error(f"FinetunedToolAdaptorStrict: Failed to process response json: {response_json}, error: {e}")
                raise e
            response = httpx.Response(
                response.status_code,
                json=response_json
            )

        req_hash = dict_hash(req_payload["messages"][0]["content"])
        is_valid = self.update_cache(req_hash, response.json())
        if not is_valid:
            self.cache = {}
            logger.error("GPToolAdaptor: 10 same function with same arguments are been called. Drop this traj")
            raise RuntimeError("10 same function with same arguments are been called. Drop this traj")

        return response


def get_tool_adaptor(adaptor_name_or_path) -> ToolAdaptor:
    """adaptor_name_or_path:
     - gpt
     - path to tool adaptor config
    """
    if adaptor_name_or_path == "gpt":
        return GPTToolAdaptor()
    elif adaptor_name_or_path == "debug":
        return DebugToolAdaptor()
    else:
        with open(adaptor_name_or_path, "r") as f:
            config = json.load(f)
        # return FinetunedToolAdaptor(config)
        return FinetunedToolAdaptorStrict(config)
