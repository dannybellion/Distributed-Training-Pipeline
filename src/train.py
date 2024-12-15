import os
import math
import yaml
import torch
import logging
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from utils import set_seed, save_checkpoint, load_config
from dataset import TextDataset
from model import load_model_and_tokenizer

def main():
    # Load config
    config = load_config("configs/default_config.yaml")
    set_seed(42)

    accelerator = Accelerator(mixed_precision="fp16")
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    model, tokenizer = load_model_and_tokenizer(config["model_name"])

    train_dataset = TextDataset(config["train_file"], tokenizer, config["max_seq_length"])
    val_dataset = TextDataset(config["val_file"], tokenizer, config["max_seq_length"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    total_steps = (len(train_loader) * config["num_epochs"]) // config["gradient_accumulation_steps"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(0.1*total_steps), 
                                                num_training_steps=total_steps)

    # Prepare for distributed
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    # Training loop
    global_step = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss / config["gradient_accumulation_steps"]
            accelerator.backward(loss)

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config["log_interval"] == 0 and accelerator.is_local_main_process:
                    logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item()*config['gradient_accumulation_steps']:.4f}")

        # Validation after each epoch
        val_loss = evaluate(model, val_loader, accelerator)
        if accelerator.is_local_main_process:
            logger.info(f"Validation Loss after epoch {epoch}: {val_loss:.4f}")
            save_checkpoint(model, tokenizer, config["output_dir"], epoch, accelerator)

def evaluate(model, loader, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(**batch, labels=batch["input_ids"])
            losses.append(accelerator.gather(outputs.loss).mean().item())
    return sum(losses)/len(losses)

if __name__ == "__main__":
    main()