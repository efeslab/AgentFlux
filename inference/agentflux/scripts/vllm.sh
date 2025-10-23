#!/bin/bash

port=8010
export CUDA_VISIBLE_DEVICES=7

# filesys
filesys_classifier_path="/m-coriander/coriander/zhan/rena/unsloth/output/fs/fs-finetuning-adapter-classifier-8000-7b-unmask-75/checkpoint-1410"
read_file_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/read_file"
read_text_file_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/read_text_file"
read_multiple_files_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/read_multiple_files"
write_file_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/write_file"
edit_file_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/edit_file"
create_directory_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/create_directory"
directory_tree_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/directory_tree"
move_file_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/move_file"
get_file_info_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/get_file_info"
list_directory_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/list_directory"
list_directory_with_sizes="/m-coriander/coriander/zhan/rena/filesys/7b-tools/list_directory_with_sizes"
search_files_path="/m-coriander/coriander/zhan/rena/filesys/7b-tools/search_files"

vllm serve \
 unsloth/Qwen2.5-7B-Instruct \
 --enforce-eager \
 --enable-auto-tool-choice \
 --tool-call-parser hermes \
 --enable-lora \
 --lora-modules \
 filesys_classifier=$filesys_classifier_path \
 read_file=$read_file_path \
 read_text_file=$read_text_file_path \
 read_multiple_files=$read_multiple_files_path \
 write_file=$write_file_path \
 edit_file=$edit_file_path \
 create_directory=$create_directory_path \
 directory_tree=$directory_tree_path \
 move_file=$move_file_path \
 get_file_info=$get_file_info_path \
 list_directory=$list_directory_path \
 list_directory_with_sizes=$list_directory_with_sizes \
 search_files=$search_files_path \
 --max-lora-rank 64 \
 --port $port