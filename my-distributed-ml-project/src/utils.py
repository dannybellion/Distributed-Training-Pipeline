import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict, save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def setup_logging(config: Dict) -> None:
    """Setup logging configuration."""
    # TODO: Implement logging setup
    pass

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
