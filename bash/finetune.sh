#!/bin/bash

category=${1:-filesys}

# generate queries
python finetune/gen_queries.py \
  --category $category \
  --output_path finetune/results/queries/$category/all_queries.txt

# generate trajectories
python finetune/gen_trajs.py \
  --category $category \
  --model gpt-5-mini \
  --url https://api.openai.com \
  --input_queries finetune/results/queries/$category/all_queries.txt \
  --output_trajs finetune/results/trajectories/$category/all_trajectories.jsonl

# prepare data for finetuning
# this will split train, eval, test sets for finetuning classifier and each tools
python finetune/data_prepare.py \
  --category $category

# generate tool template that used for finetuning tool adapters
python finetune/gen_tool_template.py \
  --category $category

# finetune classifier
bash bash/finetune_classifier.sh $category

# finetune each tool adapter
bash bash/finetune_tool_adaptors.sh $category
