# input: trajectory_path, output_dir
# outcome: for classifier: create cleaned train, eval, test splits and save them to output_dir
# for tool adapters: for each tool adapter, create a directory in output_dir and save the corresponding cleaned train, eval, test splits there

import os
import json
import random
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset

ROOT_DIR = Path(__file__).parent.parent

def clean_dataset(trajs: list[dict], tokenizer) -> list[dict]:
    """
    Clean the dataset by:
     - Verifying tool call arguments against tool specifications
     - Checking length of input_ids after applying chat template
     - Remove problematic entries using binary search method
    """
    
    def check_arguments(arguments: dict, properties: dict) -> bool:
        for key, prop in properties.items():
            if prop.get("required", False) and key not in arguments:
                return False
            # If key exists, evalidate type
            if key in arguments:
                expected_type = prop.get("type")
                if expected_type:
                    if expected_type == "string" and not isinstance(arguments[key], str):
                        return False
                    elif expected_type == "number" and not isinstance(arguments[key], (int, float)):
                        return False
                    elif expected_type == "boolean" and not isinstance(arguments[key], bool):
                        return False
                    elif expected_type == "array" and not isinstance(arguments[key], list):
                        return False
                    elif expected_type == "object" and not isinstance(arguments[key], dict):
                        return False
        # Ensure no unexpected keys are present
        for arg_key in arguments:
            if arg_key not in properties:
                return False
        return True

    def binary_search_clean(data_json):
        def binary_search(data_json):
            try:
                _ = Dataset.from_list(data_json)
            except Exception as e:
                pass
            else:
                return None
            left = 0
            right = len(data_json)  # define the problematic idx (which case [:idx] to fail) is in [left, right]
            while left < right:
                mid = (left + right) // 2
                try:
                    _ = Dataset.from_list(data_json[:mid])
                except Exception as e:
                    right = mid
                else:
                    left = mid + 1
            
            return left

        while True:
            idx = binary_search(data_json)
            if idx is None:
                break
            print(f"Found problematic index: {idx}, removing it")
            data_json.pop(idx - 1)
        
        return data_json

    tool_properties = {}
    for tool in trajs[0].get("tools", []):
        tool_properties[tool["function"]["name"]] = tool["function"]["parameters"]["properties"]

    cleaned_trajs = []
    for line_idx, line_json in enumerate(trajs):
        arg_check = True
        for i, message in enumerate(line_json.get("messages", [])):
            if message["role"] == "assistant" and "tool_calls" in message:
                for j, tool_call in enumerate(message["tool_calls"]):
                    arguments = json.loads(tool_call["function"]["arguments"])
                    properties = tool_properties[tool_call["function"]["name"]]
                    arg_check = arg_check and check_arguments(arguments, properties)
                    line_json["messages"][i]["tool_calls"][j]["function"]["arguments"] = arguments
        if arg_check is False:
            print(f"line {line_idx} has wrong arguments. pass this line")
            continue
        input_ids = tokenizer.apply_chat_template(
            conversation=line_json["messages"],
            tools=line_json["tools"],
        )
        if len(input_ids) > 32768:
            print(f"input_ids has {len(input_ids)} which is larger than arg.max_seq_length. Drop it")
            continue
        cleaned_trajs.append(line_json)

    cleaned_trajs = binary_search_clean(cleaned_trajs)

    return cleaned_trajs

def split_tools(cleaned_trajs: list[dict]) -> dict[str, list[dict]]:
    """
    For each tool, create a split of the dataset containing that tool's calls.    
    """

    def has_tool_call(messages: list[dict], tool_name: str) -> bool:
        for message in messages:
            if message["role"] == "assistant" and "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if tool_call["function"]["name"] == tool_name:
                        return True
        return False
    tool_names = []
    for tool in cleaned_trajs[0].get("tools", []):
        tool_names.append(tool["function"]["name"])

    tool_splits = {tool_name: [] for tool_name in tool_names}
    for tool_name in tool_names:
        for line_json in cleaned_trajs:
            if has_tool_call(line_json["messages"], tool_name):
                tool_splits[tool_name].append(line_json)

    return tool_splits

def write_to_file(trajs: list[dict], file_path: Path):
    dir_path = file_path.parent
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w") as f:
        for line in trajs:
            f.write(json.dumps(line) + "\n")

def split_train_eval_test(cleaned_trajs: list[dict], train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1, seed=42) -> tuple[list[dict], list[dict], list[dict]]:
    random.seed(seed)
    random.shuffle(cleaned_trajs)
    n = len(cleaned_trajs)
    train_end = int(n * train_ratio)
    eval_end = int(n * (train_ratio + eval_ratio))
    train_set = cleaned_trajs[:train_end]
    eval_set = cleaned_trajs[train_end:eval_end]
    test_set = cleaned_trajs[eval_end:]
    return train_set, eval_set, test_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Category name for MCP Tools")
    args = parser.parse_args()

    traj_path = ROOT_DIR / "finetune" / args.category / "results" / "trajectories" / "all_trajectories.jsonl"

    trajs = [json.loads(line) for line in open(traj_path, "r")]
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")

    cleaned_trajs = clean_dataset(trajs, tokenizer)
    print(f"Cleaned dataset size: {len(cleaned_trajs)}")

    cleaned_trains, cleaned_evals, cleaned_tests = split_train_eval_test(cleaned_trajs)
    print(f"Train size: {len(cleaned_trains)}, eval size: {len(cleaned_evals)}, Test size: {len(cleaned_tests)}")

    # train
    train_dir = ROOT_DIR / "finetune" / args.category / "results" / "trajectories" / "train"
    write_to_file(cleaned_trains, train_dir / "cleaned_train.jsonl")    # used for train classifier
    split_tools_train = split_tools(cleaned_trains)
    tool_adaptors_path = train_dir / "tool_adaptors"
    for tool_name, tool_trajs in split_tools_train.items():    # used for train tool adaptors
        write_to_file(tool_trajs, tool_adaptors_path / f"{tool_name}.jsonl")
        print(f"Saved {len(tool_trajs)} training examples for tool {tool_name} to {tool_adaptors_path / f'{tool_name}.jsonl'}")

    # eval
    eval_dir = ROOT_DIR / "finetune" / args.category / "results" / "trajectories" / "eval"
    write_to_file(cleaned_evals, eval_dir / "cleaned_eval.jsonl")
    split_tools_eval = split_tools(cleaned_evals)
    tool_adaptors_path = eval_dir / "tool_adaptors"
    for tool_name, tool_trajs in split_tools_eval.items():
        write_to_file(tool_trajs, tool_adaptors_path / f"{tool_name}.jsonl")
        print(f"Saved {len(tool_trajs)} eval examples for tool {tool_name} to {tool_adaptors_path / f'{tool_name}.jsonl'}")

    # test
    test_dir = ROOT_DIR / "finetune" / args.category / "results" / "trajectories" / "test"
    write_to_file(cleaned_tests, test_dir / "cleaned_test.jsonl")
    split_tools_test = split_tools(cleaned_tests)
    tool_adaptors_path = test_dir / "tool_adaptors"
    for tool_name, tool_trajs in split_tools_test.items():
        write_to_file(tool_trajs, tool_adaptors_path / f"{tool_name}.jsonl")
        print(f"Saved {len(tool_trajs)} test examples for tool {tool_name} to {tool_adaptors_path / f'{tool_name}.jsonl'}")
