import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict
import json

from dataset import get_dataloaders
from model import CustomModel
from utils import load_config

def evaluate_model(
    checkpoint_path: str,
    config_path: str = "configs/default_config.yaml"
) -> Dict:
    """
    Evaluate a trained model on the test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load model from checkpoint
    model = CustomModel.load_from_checkpoint(
        checkpoint_path,
        config=config
    )
    
    # Get test dataloader
    _, _, test_loader = get_dataloaders(config)
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1
    )
    
    # Run evaluation
    results = trainer.test(model, test_loader)
    
    return results[0]

def main():
    results = evaluate_model("checkpoints/best_model.ckpt")
    
    # Save results
    output_path = Path("evaluation_results.json")
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
