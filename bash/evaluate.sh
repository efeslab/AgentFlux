#!/bin/bash

category=${1:-filesys}

# query generation
python orchestration-framework/evaluation/gen_queries.py \
  --category $category

# trajectory generation (evaluation)
python orchestration-framework/evaluation/run_agentflux.py \
  $category \
  --classifier config/$category/classifier.json \
  --tool_adapters config/$category/tool_adapters.json \
  --query orchestration-framework/evaluation/$category/queries/fuzzing_queries.txt \
  --output orchestration-framework/evaluation/$category/eval-results/trajs_11.jsonl

# judgement
python orchestration-framework/evaluation/$category/judge.py \
  --trajs orchestration-framework/evaluation/$category/eval-results/trajs_11.jsonl \
  --output orchestration-framework/evaluation/$category/judge-results/judged_11.jsonl

# scoring
python orchestration-framework/evaluation/score.py \
  --llm_judge_path orchestration-framework/evaluation/$category/judge-results/judged_11.jsonl
