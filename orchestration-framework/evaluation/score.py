import asyncio
from typing import Coroutine
# from openai import AsyncOpenAI
import argparse
import json
import os
# import httpx

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated trajectories using LLM judge.")
    parser.add_argument("--llm_judge_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.llm_judge_path, "r") as f:
        llm_judge_output = [json.loads(line) for line in f]

    results = []
    results_task_fulfillment = []
    overall_success_num = 0
    fulfillment_num = 0
    tool_choice_num = 0
    tool_output_num = 0
    for idx, item in enumerate(llm_judge_output):
        if item["Reasoning_ToolChoice"] == "Error occurred" and \
              item["Reasoning_ToolOutput"] == "Error occurred" and \
                item["Reasoning_GroundedAnswer"] == "Error occurred" and \
                    item["Reasoning_TaskSuccess"] == "Error occurred":
            print(f"Skipping item with all zero scores")
            continue
        Score_ToolChoice = float(item.get("Score_ToolChoice", 0))
        Score_ToolOutput = float(item.get("Score_ToolOutput", 0))
        Score_GroundedAnswer = float(item.get("Score_GroundedAnswer", 0))
        Score_TaskSuccess = float(item.get("Score_TaskSuccess", 0))
        average_score = (Score_ToolChoice + Score_ToolOutput + Score_GroundedAnswer + Score_TaskSuccess) / 4.0

        results.append(average_score)
        results_task_fulfillment.append(Score_TaskSuccess)
        if average_score >= 8:
            overall_success_num += 1
        if Score_TaskSuccess >= 8:
            # print(idx)
            fulfillment_num += 1
        if Score_ToolChoice >= 8:
            tool_choice_num += 1
        if Score_ToolOutput >= 8:
            tool_output_num += 1

    print("Avg score\tAvg task success\tOverall Successfull\tFulfillment\tTool Choice\tTool Output")
    print(f"{(sum(results) / len(results) / 10 * 100 if results else 0):.3f}\t"
        f"{(sum(results_task_fulfillment) / len(results_task_fulfillment) / 10 * 100 if results_task_fulfillment else 0):.3f}\t"
        f"{overall_success_num}/{len(results)}={(overall_success_num / len(results) * 100 if results else 0):.3f}\t"
        f"{fulfillment_num}/{len(results)}={(fulfillment_num / len(results) * 100 if results else 0):.3f}\t"
        f"{tool_choice_num}/{len(results)}={(tool_choice_num / len(results) * 100 if results else 0):.3f}\t"
        f"{tool_output_num}/{len(results)}={(tool_output_num / len(results) * 100 if results else 0):.3f}")

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
