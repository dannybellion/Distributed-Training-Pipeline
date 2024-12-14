#!/bin/bash

# Run training on a single GPU
CUDA_VISIBLE_DEVICES=0 python src/train.py --config configs/default_config.yaml
