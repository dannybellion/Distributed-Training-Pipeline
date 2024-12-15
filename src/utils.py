import random
import numpy as np
import torch
import os
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, tokenizer, output_dir, epoch, accelerator):
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch}"))
        tokenizer.save_pretrained(os.path.join(output_dir, f"epoch_{epoch}"))