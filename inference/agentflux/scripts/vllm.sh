#!/bin/bash

port=8010
export CUDA_VISIBLE_DEVICES=7

# filesys
finetune_serve_dir=${1:-"../finetune/filesys/results/finetune_serve/"}

filesys_classifier=$finetune_serve_dir/classifier
read_file="$finetune_serve_dir/read_file"
read_text_file="$finetune_serve_dir/read_text_file"
read_multiple_files="$finetune_serve_dir/read_multiple_files"
write_file="$finetune_serve_dir/write_file"
edit_file="$finetune_serve_dir/edit_file"
create_directory="$finetune_serve_dir/create_directory"
directory_tree="$finetune_serve_dir/directory_tree"
move_file="$finetune_serve_dir/move_file"
get_file_info="$finetune_serve_dir/get_file_info"
list_directory="$finetune_serve_dir/list_directory"
list_directory_with_sizes="$finetune_serve_dir/list_directory_with_sizes"
search_files="$finetune_serve_dir/search_files"

vllm serve \
 unsloth/Qwen2.5-7B-Instruct \
 --enforce-eager \
 --enable-auto-tool-choice \
 --tool-call-parser hermes \
 --enable-lora \
 --lora-modules \
 filesys_classifier=$filesys_classifier \
 read_file=$read_file \
 read_text_file=$read_text_file \
 read_multiple_files=$read_multiple_files \
 write_file=$write_file \
 edit_file=$edit_file \
 create_directory=$create_directory \
 directory_tree=$directory_tree \
 move_file=$move_file \
 get_file_info=$get_file_info \
 list_directory=$list_directory \
 list_directory_with_sizes=$list_directory_with_sizes \
 search_files=$search_files \
 --max-lora-rank 64 \
 --port $port