import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
import yaml

class CustomDataset(Dataset):
    """Custom dataset class for the ML project."""
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Root directory of the dataset
            split: One of 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.split = split
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load data samples."""
        # TODO: Implement data loading logic
        pass
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        # TODO: Implement sample loading logic
        pass

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test dataloaders."""
    train_dataset = CustomDataset(config['data_dir'], "train")
    val_dataset = CustomDataset(config['data_dir'], "val")
    test_dataset = CustomDataset(config['data_dir'], "test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    return train_loader, val_loader, test_loader
