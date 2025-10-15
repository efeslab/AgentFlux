import warnings
import os
from typing import Any, Callable, Optional, TypeVar, Union
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from accelerate import PartialState
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import remove_none_values
from trl.data_utils import (
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    pack_dataset,
    truncate_dataset,
)
import random

search_file_response_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 1836, 10931, 16707, 151658, 151645, 271]
all_monday_api_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 541, 20737, 1292, 11697, 16707, 151658, 151645, 271]
get_board_info_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 455, 23919, 3109, 16707, 151658, 151645, 271]
get_graphql_schema_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 455, 14738, 1470, 25371, 16707, 151658, 151645, 271]
get_type_details_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 455, 1819, 13260, 16707, 151658, 151645, 271]
list_workspaces_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 1607, 11498, 44285, 16707, 151658, 151645, 271]
workspace_info_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 42909, 3109, 16707, 151658, 151645, 271]
summarize_token_ids = [151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 1242, 5612, 551, 16707, 151658, 151645, 271]


unmask_list = [
    (search_file_response_token_ids, float(os.environ.get("UNMASK_SEARCH_FILE_PROB", 0.0))),
    (all_monday_api_token_ids, float(os.environ.get("UNMASK_ALL_MONDAY_API_PROB", 0.0))),
    (get_board_info_token_ids, float(os.environ.get("UNMASK_GET_BOARD_INFO_PROB", 0.0))),
    (get_graphql_schema_token_ids, float(os.environ.get("UNMASK_GET_GRAPHQL_SCHEMA_PROB", 0.0))),
    (get_type_details_token_ids, float(os.environ.get("UNMASK_GET_TYPE_DETAILS_PROB", 0.0))),
    (list_workspaces_token_ids, float(os.environ.get("UNMASK_LIST_WORKSPACES_PROB", 0.0))),
    (workspace_info_token_ids, float(os.environ.get("UNMASK_WORKSPACE_INFO_PROB", 0.0))),
    (summarize_token_ids, float(os.environ.get("UNMASK_SUMMARIZE_PROB", 0.0))),
]

def apply_randomly_unmask(processed: dict, tokenizer):
    # unmask_ids = tokenizer(unmask_text, add_special_tokens=False)["input_ids"]

    for i, (unmask_ids, unmask_probability) in enumerate(unmask_list):

        input_ids = processed["input_ids"]

        # Ensure plain Python lists
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()

        m = len(unmask_ids)
        n = len(input_ids)

        matches = []
        # Scan linearly and allow overlapping matches
        for i in range(n - m + 1):
            if input_ids[i:i + m] == unmask_ids:
                matches.append((i, i + m))
        if unmask_probability > 0:
            print(f"{i} matches: {matches}")
        for start, end in matches:
            if random.random() < unmask_probability:
                processed["assistant_masks"][start: end] = [0] * (end - start)



def substituted_prepare_dataset():
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = next(iter(dataset)).keys()
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            prompt_ids = processing_class.apply_chat_template(
                                example["prompt"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_processed = processing_class.apply_chat_template(
                                example["prompt"] + example["completion"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            warnings.warn(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create a completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask

                    else:  # language modeling case
                        if is_conversational(example):
                            processed = processing_class.apply_chat_template(
                                example["messages"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                                raise RuntimeError(
                                    "You're using `assistant_only_loss=True`, but at least one example has no "
                                    "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                    "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                                    "check the template and ensure it's correctly configured to support assistant "
                                    "masking."
                                )
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}
                    
                    apply_randomly_unmask(output, processing_class)
                    
                    return output

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in dataset.column_names:
                    columns.append("completion_mask")
                if "assistant_masks" in dataset.column_names:
                    columns.append("assistant_masks")

                dataset = dataset.select_columns(columns)

                # Packing adds new column "seq_lengths" needed for document aware FlashAttention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                dataset = dataset.select_columns(
                    {"input_ids", "seq_lengths", "completion_mask"}.intersection(dataset.column_names)
                )

        return dataset
    
    return _prepare_dataset