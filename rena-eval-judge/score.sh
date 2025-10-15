#!/bin/bash

path=(
  # /home/zhan/rena/rena-eval-judge/notion/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_gpt-oss-120b.jsonl
  # /home/zhan/rena/rena-eval-judge/notion/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_kimi-k2-instruct.jsonl
  # /home/zhan/rena/rena-eval-judge/notion/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_deepseek-V3.1.jsonl
  # /home/zhan/rena/rena-eval-judge/notion/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_glm-4.5.jsonl




  # /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_ToolACE-2.5-8B.jsonl
  # /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/baseline_mcpbench/nosum-gpt5-judge/rena_traj_ToolACE-2.5-8B.jsonl

  # /home/zhan/rena/rena-eval-judge/monday/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_ToolACE-2.5-8B.jsonl
  # /home/zhan/rena/rena-eval-judge/monday/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_xLAM-2-8b-fc-r.jsonl
  # /home/zhan/rena/rena-eval-judge/monday/judge-results/9-25/baseline_mcpbench/nosum-gpt5-judge/rena_traj_ToolACE-2.5-8B.jsonl

  # /home/zhan/rena/rena-eval-judge/notion/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_ToolACE-2.5-8B.jsonl
  # /home/zhan/rena/rena-eval-judge/notion/judge-results/9-25/baseline_rena/nosum-gpt5-judge/rena_traj_xLAM-2-8b-fc-r.jsonl
  # /home/zhan/rena/rena-eval-judge/notion/judge-results/9-25/baseline_mcpbench/nosum-gpt5-judge/rena_traj_ToolACE-2.5-8B.jsonl


  /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/detailed_baseline/rena_traj_GPT-5-mini_judge.jsonl
  /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/detailed_baseline/rena_traj_Qwen-2.5-7B_judge.jsonl
  /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/detailed_baseline/rena_traj_Qwen-3-8B_judge.jsonl
  /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/detailed_baseline/rena_traj_Qwen-3-8B_No_Reasoning_judge.jsonl
  /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/detailed_baseline/rena_traj_ToolAce-8B_judge.jsonl
  /home/zhan/rena/rena-eval-judge/filesys/judge-results/9-25/detailed_baseline/rena_traj_xLAM-2-8B_judge.jsonl
)

for p in "${path[@]}"; do
loc=$(wc -l < $p)
echo ""
echo $p $loc
python python/llm_gt_judge_score_01loss.py --llm_judge_path $p
done