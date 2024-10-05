#!/bin/bash

GPU="0"
OUTPUT_FILE="pair_for_mpo.json"

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./pair_generation.py \
    --output_file="${OUTPUT_FILE}" \
    --batch_size=1 \
    --model="models/sft/checkpoint-135" \
    --method="mpo" \