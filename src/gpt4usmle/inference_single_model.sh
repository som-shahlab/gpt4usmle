#!/bin/bash

# Paths
# CSV_FILE_PATH="/share/pi/nigam/scottyf/nbme_sample_items.csv"
# OUTPUT_PATH="/share/pi/nigam/scottyf/gpt4usmle_model_outputs_20240719.csv"
CSV_FILE_PATH="/share/pi/nigam/scottyf/ai_generated_questions.csv"
OUTPUT_PATH="/share/pi/nigam/scottyf/ai_generated_questions_output_20240721_v2.csv"
MODEL_PATH_PREFIX="/share/pi/nigam/pretrained/"

# Model names
MODELS=(
    # "gemma-2-27b-it"  # Doesn't generate any output text with huggingface TGI
    "Mixtral-8x22B-Instruct-v0.1"
    "Qwen2-72B-Instruct"
    "Phi-3-medium-4k-instruct"
    "BioMistral-7B"  # Too difficult to parse
    "Mistral-7B-Instruct-v0.3"
    "Llama3-OpenBioLLM-70B"  # Base model is Llama-3-70B-Instruct
    "Llama3-Med42-70B"  # Base model is LLaMA-3 70B
    "Meta-Llama-3-70B-Instruct"
    "med42-70b"  # Base model is LLaMA-2 70B
    # "ClinicalCamel-70B"  # Format is too difficult to parse
    "llama-2-70b-chat_huggingface"
)

# Initialize the output file with column headers
echo "item_num,gt_answer,model,model_answer" > "$OUTPUT_PATH"

export NCCL_P2P_DISABLE=1  # Needed in order for ray to initialize properly on Carina?
export VLLM_ATTENTION_BACKEND=FLASHINFER  # Required for Gemma-2?

# Process each model
for MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODEL"
    
    # Default number of devices
    # You need either 4 A100's or 8 V100's to run the 70B parameter models
    NUM_DEVICES=4
    # NUM_DEVICES=1
    # Adjust NUM_DEVICES for models with "70b" or "70B" in their name
    # if [[ "$MODEL" =~ 70[bB] ]]; then
    #     NUM_DEVICES=4
    # fi
    
    python inference_single_model.py --model_name "$MODEL" --csv_file_path "$CSV_FILE_PATH" --output_path "$OUTPUT_PATH" --model_path_prefix "$MODEL_PATH_PREFIX" --num_devices "$NUM_DEVICES"
done

echo "All models have been processed."

# Other models to include: 
# tiiuae/falcon-40b-instruct
# google/gemma-7b
# databricks/dbrx-instruct
# meta-llama/Meta-Llama-3-70B-Instruct
# meta-llama/Meta-Llama-3-8B-Instruct
# lmsys/vicuna-13b-v1.3
# allenai/OLMo-7B-Instruct
# "Qwen1.5-72B-Chat"
# "gemma-2-27b-it"
# "Mixtral-8x7B-Instruct-v0.1"
# "Mistral-7B-Instruct-v0.2"
# "Mistral-7B-Instruct-v0.1"
# "llama-2-7b-chat_huggingface"
# "llama-2-13b-chat_huggingface"
# "AlpaCare-llama2-7b"
# "AlpaCare-llama2-13b"
# "meditron-7b"
# "meditron-70b"
# "zephyr-7b-beta"  # Base model is Mistral-7B-v0.1
# "Yi-1.5.34B"