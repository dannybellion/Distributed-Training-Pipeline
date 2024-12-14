#!/bin/bash

# Example distributed training script
# Modify according to your specific distributed setup

NUM_NODES=2
NUM_GPUS_PER_NODE=4
NODE_RANK=$1  # Pass this as argument when running the script

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr="master_node_hostname" \
    --master_port=29500 \
    src/train.py
