import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import yaml
from pathlib import Path

from dataset import get_dataloaders
from model import CustomModel
from utils import load_config

def main():
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Initialize data
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Initialize model
    model = CustomModel(config)
    
    # Initialize logger
    logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb']['entity']
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu",
        devices=config['distributed']['gpus_per_node'],
        num_nodes=config['distributed']['num_nodes'],
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath='checkpoints',
                filename='{epoch}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss'
            )
        ]
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    main()
