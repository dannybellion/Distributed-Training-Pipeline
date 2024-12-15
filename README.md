# Distributed Training Pipeline

A scalable distributed training pipeline for transformer-based language models using PyTorch and Hugging Face.

## Features

- Multi-GPU training with Hugging Face Accelerate
- Configurable hyperparameters via YAML
- Mixed precision training support
- Checkpoint saving and loading
- Progress logging and validation metrics

## Code Structure

### Core Components

- `src/train.py`: Main training loop with distributed training setup using Accelerate
- `src/model.py`: Model loading utilities using Hugging Face transformers
- `src/dataset.py`: Custom dataset class for text data with padding collation
- `src/utils.py`: Helper functions for reproducibility and checkpointing

### Key Implementation Details

**Dataset Handling**
```python
class TextDataset:
    # Efficient text dataset with tokenization and padding
    # Supports both masked LM and causal LM tasks
```

**Training Loop**
```python
def main():
    # Distributed setup with Accelerate
    # Gradient accumulation and mixed precision
    # Regular validation and checkpointing
```

**Model Management**
```python
def load_model_and_tokenizer():
    # Flexible model loading from Hugging Face hub
    # Supports various model architectures
```

## Directory Structure

```
.
├── config/         # YAML configuration files
├── data/          # Training and validation data
├── notebooks/     # Analysis notebooks
├── scripts/       # Training scripts
└── src/           # Core implementation
```

## License

MIT
