#!/bin/bash

category=${1:-filesys}
batch_size=${2:-4}
accumulate_step=${3:-4}
num_train_epochs=${4:-4}

train_dir="./$category/results/trajectories/train/tool_adaptors"
for file in "$train_dir"/*.jsonl; do
  tool_name=$(basename "$file" .jsonl)
  echo "Finetuning tool adapter for tool: $tool_name"

  train_json_file_path="./$category/results/trajectories/train/tool_adaptors/$tool_name.jsonl"
  eval_json_file_path="./$category/results/trajectories/eval/tool_adaptors/$tool_name.jsonl"
  chat_template_path="./base_models/Qwen2.5-7B-Instruct/$category/$tool_name.jinja"
  output_dir="./$category/results/finetune_output/tool_adaptors/$tool_name"
  log_file_path="./$category/results/log/tool_adaptors/$tool_name.log"

  mkdir -p "$output_dir"
  mkdir -p "$(dirname "$log_file_path")"

  num_samples=$(wc -l < $train_json_file_path)
  num_batches_per_epoch=$(( (num_samples + 4 - 1) / 4 ))
  steps_per_epoch=$(( (num_batches_per_epoch + 4 - 1) / 4 ))
  eval_steps=$(( (steps_per_epoch + 4 - 1) / 4 ))

  echo "Number of batches per epoch: $num_batches_per_epoch"
  echo "Steps per epoch: $steps_per_epoch"
  echo "Eval steps: $eval_steps"

  export PYTHONUNBUFFERED=1
  python unsloth-cli-split.py \
    --model_name unsloth/Qwen2.5-7B-Instruct \
    --max_seq_length 32768 \
    --r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --bias "none" \
    --use_gradient_checkpointing "unsloth" \
    --random_state 3407 \
    --use_rslora \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $accumulate_step \
    --warmup_steps 5 \
    --num_train_epochs $num_train_epochs \
    --learning_rate 5e-6 \
    --logging_steps 1 \
    --optim "adamw_8bit" \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --seed 3407 \
    --output_dir "$output_dir" \
    --report_to "none" \
    --eval_steps $eval_steps \
    --save_model \
    --save_path "$save_path" \
    --quantization "f16" \
    --train_json_file $train_json_file_path \
    --validation_json_file $eval_json_file_path \
    --chat_template_path "$chat_template_path" \
    2>&1 | tee "$log_file_path"
  
  # copy the content of second epoch's output to a separate folder for easy access
  epoch_2_dir="./$category/results/finetune_serve/$tool_name"
  mkdir -p "$epoch_2_dir"
  cp -r "$output_dir/checkpoint-$(($steps_per_epoch * 2))/"* "$epoch_2_dir/"

done