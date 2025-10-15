#!/usr/bin/env python3

"""
🦥 Starter Script for Fine-Tuning FastLanguageModel with Unsloth

This script is designed as a starting point for fine-tuning your models using unsloth.
It includes configurable options for model loading, PEFT parameters, training arguments, 
and model saving/pushing functionalities.

You will likely want to customize this script to suit your specific use case 
and requirements.

Here are a few suggestions for customization:
    - Modify the dataset loading and preprocessing steps to match your data.
    - Customize the model saving and pushing configurations.

Usage: (most of the options have valid default values this is an extended example for demonstration purposes)
    python unsloth-cli.py --model_name "unsloth/llama-3-8b" --max_seq_length 8192 --dtype None --load_in_4bit \
    --r 64 --lora_alpha 32 --lora_dropout 0.1 --bias "none" --use_gradient_checkpointing "unsloth" \
    --random_state 3407 --use_rslora --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --warmup_steps 5 --max_steps 400 --learning_rate 2e-6 --logging_steps 1 --optim "adamw_8bit" \
    --weight_decay 0.005 --lr_scheduler_type "linear" --seed 3407 --output_dir "outputs" \
    --report_to "tensorboard" --save_model --save_path "model" --quantization_method "f16" \
    --push_model --hub_path "hf/model" --hub_token "your_hf_token"

To see a full list of configurable options, use:
    python unsloth-cli.py --help

Happy fine-tuning!
"""

import argparse
import os


def run(args):
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset, Dataset
    from transformers.utils import strtobool
    from transformers import AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    import json
    import random
    from datasets import Features, Value, Sequence
    from preprocess_dataset import substituted_prepare_dataset
    import logging
    logging.getLogger('hf-to-gguf').setLevel(logging.WARNING)


    features = Features({
        "model": Value("string"),

        # define structures of messages
        "messages": Sequence({
            "role": Value("string"),
            "content": Value("string"),
            "tool_call_id": Value("string"),
            "tool_calls": Sequence({
                "id": Value("string"),
                "type": Value("string"),
                "function": {
                    "name": Value("string"),
                    "arguments": Value("string")
                }
            }),
        }),

        # define structures of tools
        "tools": Sequence({
            "type": Value("string"),
            "function": {
                "name": Value("string"),
                "description": Value("string"),
                "parameters": Value("string"),
            }
        })
    })


    # Load model and tokenizer
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-3B-Instruct")
    with open(args.chat_template_path, "r") as f:
        tokenizer.chat_template = f.read()


    def load_dataset_from_file(file_path):
        # data_json = [json.loads(line) for line in open(file_path, "r")]
        data_json = []
        for line in open(file_path, "r"):
            data = json.loads(line)
            if data["messages"][-1]["content"].strip() == "inference engine error":
                print("Warning: Skipping a data point with inference engine error.")
                continue
            # messages
            for i, msg in enumerate(data["messages"]):
                if msg["role"] == "user":
                    assert isinstance(msg["content"], str)
                elif msg["role"] == "assistant":
                    if msg.get("tool_calls"):
                        for tool_call in msg["tool_calls"]:
                            assert isinstance(tool_call["id"], str)
                            assert isinstance(tool_call["type"], str)
                            assert isinstance(tool_call["function"]["name"], str)
                            # assert isinstance(tool_call["function"]["arguments"], str)
                    if msg.get("content"):
                        assert isinstance(msg["content"], str)
                elif msg["role"] == "tool":
                    assert isinstance(msg["content"], str)

            # tools
            for tool in data.get("tools", []):
                assert isinstance(tool["type"], str)
                assert isinstance(tool["function"]["name"], str)
                assert isinstance(tool["function"]["description"], str)
                # assert isinstance(tool["function"]["parameters"], str)
                tool["function"]["parameters"] = json.dumps(tool["function"]["parameters"])
                assert isinstance(tool["function"]["parameters"], str)
            data_json.append(data)
        random.shuffle(data_json)
        cleaned_dataset = Dataset.from_list(data_json)
        return cleaned_dataset
        # return data_json

    train_dataset = load_dataset_from_file(args.train_json_file)
    eval_dataset = load_dataset_from_file(args.validation_json_file)
    print(f"train_dataset nums: {len(train_dataset)}")
    print(f"eval_dataset nums: {len(eval_dataset)}")

    # Configure training arguments
    if args.num_train_epochs: 
        training_args = SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            # max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=args.output_dir,
            report_to=args.report_to,
            max_length=args.max_seq_length,
            dataset_num_proc=2,
            packing=False,

            num_train_epochs=args.num_train_epochs,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="epoch",
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            assistant_only_loss=True,
            eval_on_start=True
        )
    else:
        training_args = SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=args.output_dir,
            report_to=args.report_to,
            max_length=args.max_seq_length,
            dataset_num_proc=2,
            packing=False,
        )

    # training_args = training_args.set_dataloader(num_workers=2)
    # Initialize trainer
    SFTTrainer._prepare_dataset = substituted_prepare_dataset()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Train model
    trainer_stats = trainer.train()

    # Save model
    if args.save_model:
        # if args.quantization_method is a list, we will save the model for each quantization method
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(f"Saving model with quantization method: {quantization_method}")
                    model.save_pretrained_gguf(
                        args.save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=args.hub_path,
                            hub_token=args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                print(f"Saving model with quantization method: {args.quantization}")
                model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=quantization_method,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        print("Warning: The model is not saved!")


if __name__ == "__main__":

    # Define argument parser
    parser = argparse.ArgumentParser(description="🦥 Fine-tune your llm faster using unsloth!")

    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument('--model_name', type=str, default="unsloth/llama-3-8b", help="Model name to load")
    model_group.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action='store_true', help="Use 4bit quantization to reduce memory usage")
    # model_group.add_argument('--json_file', type=str, default="yahma/alpaca-cleaned", help="Huggingface dataset to use for training")
    model_group.add_argument('--train_json_file', type=str, required=True, help="Path to the JSON file containing training data")
    model_group.add_argument('--validation_json_file', type=str, required=True, help="Path to the JSON file containing validation data")

    lora_group = parser.add_argument_group("🧠 LoRA Options", "These options are used to configure the LoRA model.")
    lora_group.add_argument('--r', type=int, default=16, help="Rank for Lora model, default is 16.  (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_dropout', type=float, default=0.0, help="LoRA dropout rate, default is 0.0 which is optimized.")
    lora_group.add_argument('--bias', type=str, default="none", help="Bias setting for LoRA")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Use gradient checkpointing")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Random state for reproducibility, default is 3407.")
    lora_group.add_argument('--use_rslora', action='store_true', help="Use rank stabilized LoRA")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="Configuration for LoftQ")

   
    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training, default is 2.")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps, default is 4.")
    training_group.add_argument('--warmup_steps', type=int, default=5, help="Number of warmup steps, default is 5.")
    training_group.add_argument('--max_steps', type=int, default=400, help="Maximum number of training steps.")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate, default is 2e-4.")
    training_group.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizer type.")
    training_group.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay, default is 0.01.")
    training_group.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler type, default is 'linear'.")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility, default is 3407.")

    training_group.add_argument('--train_ratio', type=float, default=0.9, help="Train ratio, default is 0.9.")
    training_group.add_argument('--eval_steps', type=int, default=15, help="Eval steps, default is 15.")
    training_group.add_argument('--num_train_epochs', type=int, required=True, help="Number of training epochs")
    training_group.add_argument('--chat_template_path', type=str, default="chat_template.jinja2", help="Path to the chat template file")
    training_group.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size per device during evaluation, default is 1.")

    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="tensorboard",
        choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"],
        help="The list of integrations to report the results and logs to. Supported platforms are: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations installed, 'none' for no integrations.")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps, default is 1")

    # Saving and pushing arguments
    save_group = parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    save_group.add_argument('--save_model', action='store_true', help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model, default is 'merged_16bit'")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="model", help="Path to save the model")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+",
        help="Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf ")

    push_group = parser.add_argument_group('🚀 Push Model Options')
    push_group.add_argument('--push_model', action='store_true', help="Push the model to Hugging Face hub after training")
    push_group.add_argument('--push_gguf', action='store_true', help="Push the model as GGUF to Hugging Face hub after training")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Path on Hugging Face hub to push the model")
    push_group.add_argument('--hub_token', type=str, help="Token for pushing the model to Hugging Face hub")

    args = parser.parse_args()
    run(args)
