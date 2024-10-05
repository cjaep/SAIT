#!/bin/bash

GPU="0"
MODEL="mpo"

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python ./generation.py \
    --output_file="results/${MODEL}.json" \
    --batch_size=16 \
    --model="models/${MODEL}" \