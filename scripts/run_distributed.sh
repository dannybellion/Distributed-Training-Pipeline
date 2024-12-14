#!/bin/bash

# Number of GPUs to use
NUM_GPUS=4

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    src/train.py \
    --config configs/default_config.yaml
