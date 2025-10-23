#!/bin/bash

category=${1:-filesys}

results_dir=$category/results
queries_dir=$results_dir/queries
trajectories_dir=$results_dir/trajectories
mkdir -p $queries_dir
mkdir -p $trajectories_dir

# # generate queries
# python gen_queries.py \
#   --category $category \
#   --output_path $queries_dir/all_queries.txt

# # generate trajectories
# workspace="/m-coriander/coriander/zhan/AgentFlux/workspace"
# WORKSPACE=$workspace python gen_trajs.py \
#   --category $category \
#   --model gpt-5-mini \
#   --url https://api.openai.com \
#   --input_queries $queries_dir/all_queries.txt \
#   --output_trajs $trajectories_dir/all_trajectories.jsonl

# prepare data for finetuning
# this will split train, eval, test sets for finetuning classifier and each tools
python data_prepare.py \
  --category $category

# generate tool template that used for finetuning tool adapters
python gen_tool_template.py \
  --category $category \
  --model_folder ./base_models/Qwen2.5-7B-Instruct
exit 0
# finetune classifier
bash scripts/finetune_classifier.sh $category

# finetune each tool adapter
bash scripts/finetune_tool_adaptors.sh $category
