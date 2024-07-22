import argparse
import gc
import csv
import os
from typing import List


import pandas as pd
import ray
import torch
from transformers import pipeline
from tqdm import tqdm
from vllm import LLM, SamplingParams


def initialize_arg_parser():
    """Initialize and return an argument parser for command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate model responses for input prompts."
    )
    parser.add_argument(
        "--model_name", required=True, type=str, help="The name of the model to process"
    )
    parser.add_argument(
        "--csv_file_path",
        default="/share/pi/nigam/scottyf/nbme_sample_items.csv",
        type=str,
        help="Path to the CSV input file",
    )
    parser.add_argument(
        "--output_path",
        default="/share/pi/nigam/scottyf/gpt4usmle_model_outputs.csv",
        type=str,
        help="Path to the CSV output file",
    )
    parser.add_argument(
        "--model_path_prefix",
        default="/share/pi/nigam/pretrained/",
        type=str,
        help="Prefix path to the model directories",
    )
    parser.add_argument(
        "--num_devices", default=1, type=int, help="Number of GPU devices to use"
    )
    parser.add_argument(
        "--backend",
        default="huggingface",
        type=str,
        help="Whether to use `vllm` or `huggingface` for inference",
    )
    return parser


def read_input_data(csv_file_path):
    """Read and return the input data from a CSV file."""
    return pd.read_csv(csv_file_path)


def format_item_to_prompt(row, backend):
    if "ItemText_Raw" in row:
        item_text = row["ItemText_Raw"]
    elif "ItemStem_Text" in row:
        item_text = f"{row['ItemStem_Text']}\n"
        for label in "ABCDEFGHIJ":
            answer_key = f"Answer__{label}"
            if pd.notna(row[answer_key]):
                item_text += f"({label}) {row[answer_key]}\n"
    else:
        raise ValueError("Expected 'ItemStem_Text' or 'ItemStem_Raw' in columns")
    prompt_prefix = """The following is a question from the US Medical Licensing Exam:

"""
    prompt_suffix = """
Output only the letter corresponding to the correct answer surrounded by parentheses and nothing else.
So if "Z" were the letter corresponding to the correct answer, then the model output would be as follows:

OUTPUT FORMAT:
(Z)

YOUR ANSWER:
(
"""
    if backend == "huggingface":
        return prompt_prefix + item_text + prompt_suffix
    else:
        return item_text


def generate_model_responses(
    model_path,
    prompts,
    num_devices,
    torch_dtype="bfloat16",
    backend="huggingface",
    verbose=1,
):
    """Generate responses from the model for given prompts and return them."""

    # Get GPU device characteristics
    devices: List[int] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"Visible CUDA devices: `{devices}`")

    # If we're using V100 machines, we cannot use 'bfloat16' because it's not supported
    # See https://github.com/vllm-project/vllm/issues/1144
    compute_capability = torch.cuda.get_device_capability()
    if torch_dtype == "bfloat16" and compute_capability[0] < 8:
        print("GPU does not support bfloat16, falling back to float16")
        torch_dtype = "float16"

    # TODO: https://github.com/vllm-project/vllm/issues/1116#issuecomment-1730162137
    # breakpoint()
    # ray.shutdown()
    # ray.init(num_gpus=torch.cuda.device_count())

    if backend == "vllm":
        print("Loading the vllm model...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=num_devices,
            dtype=torch_dtype,
            max_model_len=2048,  # TODO: This should be exracted from the model config, but not so high that GPUs can't support
        )
        sampling_params = SamplingParams(n=1, temperature=0.0, top_p=0.95, max_tokens=1)
        outputs = llm.generate(prompts, sampling_params)

        model_answers = [output.outputs[0].text for output in outputs]

        # Cleanup resources
        # gc.collect()
        # torch.cuda.empty_cache()

    elif backend == "huggingface":
        if num_devices > 1:
            import accelerate  # Throw an error if accelerate isn't available b/c otherwise multi-GPU won't work

        dtype_obj = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        print("Initializing huggingface pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": dtype_obj},
            device_map="auto",  # Requires `accelerate` from huggingface to support sharding model across devices
        )

        model_answers = []
        for p in tqdm(prompts):
            messages = [
                # System role not supported for Gemma or Mistral models
                # {"role": "system", "content": "You are a medical expert who answers US Medical License Exam questions correctly"},
                {"role": "user", "content": p}
            ]

            # In `tranformers>=4.42.0` chat templates are applied automatically
            # prompt = pipe.tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )

            terminators = [
                pipe.tokenizer.eos_token_id,
                # pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # The text generation pipeline in huggingface automatically adds the necessary chat tags
            # See https://huggingface.co/docs/transformers/main/en/chat_templating#is-there-an-automated-pipeline-for-chat
            # See https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts
            # See https://huggingface.co/docs/transformers/main/en/chat_templating#how-do-i-use-chat-templates
            # See https://github.com/huggingface/transformers/blob/6c1d0b069de22d7ed8aa83f733c25045eea0585d/src/transformers/pipelines/text_generation.py#L281
            outputs = pipe(
                messages, max_new_tokens=64, eos_token_id=terminators, do_sample=False
            )
            model_answer = outputs[0]["generated_text"][-1]["content"]
            if verbose:
                print(model_answer)
            model_answers.append(model_answer)

    else:
        raise ValueError(f"Backend {backend} not supported")

    return model_answers


def write_responses_to_file(
    output_path, item_nums, gt_answers, model_name, model_answers
):
    """Append generated responses to the output CSV file."""
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for item_num, gt_answer, model_answer in zip(
            item_nums, gt_answers, model_answers
        ):
            writer.writerow([item_num, gt_answer, model_name, model_answer])


def main():
    parser = initialize_arg_parser()
    args = parser.parse_args()

    df = read_input_data(args.csv_file_path)
    prompts = df.apply(
        lambda x: format_item_to_prompt(x, backend=args.backend), axis=1
    ).tolist()
    item_nums = df["ItemNum"].tolist()
    gt_answers = df["Answer_Key"].tolist()

    model_path = os.path.join(args.model_path_prefix, args.model_name)
    model_answers = generate_model_responses(
        model_path, prompts, item_nums, gt_answers, args.num_devices, args.backend
    )

    write_responses_to_file(
        args.output_path, item_nums, gt_answers, args.model_name, model_answers
    )
    print(f"Completed processing for model: {args.model_name}")


if __name__ == "__main__":
    main()
