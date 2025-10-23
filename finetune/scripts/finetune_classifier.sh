#!/bin/bash

category=${1:-filesys}
batch_size=${2:-4}
accumulate_step=${3:-4}
num_train_epochs=${4:-4}

train_data_path="./$category/results/trajectories/train/cleaned_train.jsonl"
validation_data_path="./$category/results/trajectories/eval/cleaned_eval.jsonl"
chat_template_path="./base_models/Qwen2.5-7B-Instruct/classifier.jinja"
output_dir="./$category/results/finetune_output/classifier"
log_path="./$category/results/log/classifier.log"

mkdir -p "$output_dir"
mkdir -p "$(dirname "$log_path")"

num_samples=$(wc -l < $train_data_path)
num_batches_per_epoch=$(( (num_samples + batch_size - 1) / batch_size ))
steps_per_epoch=$(( (num_batches_per_epoch + accumulate_step - 1) / accumulate_step ))
eval_steps=$(( (steps_per_epoch + 5 - 1) / 5 ))    # 5 evals per epoch

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
 --output_dir $output_dir \
 --report_to "none" \
 --eval_steps $eval_steps \
 --save_model \
 --save_path $output_dir \
 --quantization "f16" \
 --train_json_file $train_data_path \
 --validation_json_file $validation_data_path \
 --chat_template_path $chat_template_path \
 --per_device_eval_batch_size 1 \
 2>&1 | tee $log_path

# copy the content of second epoch's output to a separate folder for easy access
epoch_2_dir="./$category/results/finetune_serve/classifier"
mkdir -p "$epoch_2_dir"
cp -r "$output_dir/checkpoint-$(($steps_per_epoch * 2))/"* "$epoch_2_dir/"
