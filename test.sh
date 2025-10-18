#!/bin/bash

dir="/m-coriander/coriander/zhan/DualTune/rena-finetune/results/trajectories/filesys/train/tool_adaptors"

for file in "$dir"/*.jsonl; do
  tool_name=$(basename "$file" .jsonl)
  echo $tool_name
done