#!/bin/bash

CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-7 python sft.py