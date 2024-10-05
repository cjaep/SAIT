#!/bin/bash

GPU="0"
OUTPUT_FILE="mpo"
PAIRED_FILE="pair_for_mpo.json"

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./generation.py \
    --output_dir="models/${OUTPUT_FILE}" \
    --paired_file="pairs/${PAIRED_FILE}" \
    --epochs=1 \
    --beta=0.5 \
