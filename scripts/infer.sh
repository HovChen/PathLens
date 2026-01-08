#!/bin/bash

MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
VIT_PATH="openai/clip-vit-large-patch14-336"
HLORA_PATH="path/to/hlora_weights.bin"
VOCAB_PROJ_PATH="path/to/vocab_proj_weights.bin"
IMG_PATH=<path/to/imgs>

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/infer.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --vocab_proj_path "$VOCAB_PROJ_PATH" \
    --question "What is the condition of the interstitial in the image?" \
    --img_path "examples/demo.png" \
    --infer
