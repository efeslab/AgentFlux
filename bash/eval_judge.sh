#!/bin/bash

category=${1:-filesys}

# query generation
python rena-eval-judge/gen_queries.py \
  --category $category

# trajectory generation (evaluation)
python rena-eval-judge/run_proxy.py \
  $category \
  --classifier config/$category/classifier.json \
  --tool_adapters config/$category/tool_adapters.json \
  --query rena-eval-judge/$category/queries/fuzzing_queries.txt \
  --output rena-eval-judge/$category/eval-results/trajs_11.jsonl

# judgement
python rena-eval-judge/$category/judge.py \
  --trajs rena-eval-judge/$category/eval-results/trajs_11.jsonl \
  --output rena-eval-judge/$category/judge-results/judged_11.jsonl

# scoring
python rena-eval-judge/score.py \
  --llm_judge_path rena-eval-judge/$category/judge-results/judged_11.jsonl
