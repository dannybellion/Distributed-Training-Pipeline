# Distributed Training Pipeline

A distributed training pipeline for transformer language models using PyTorch, Hugging Face, and Accelerate for efficient multi-GPU training.

## Features

- Multi-GPU training with Accelerate
- Mixed precision training
- Configurable via YAML
- Automatic checkpointing
- Training progress logging
- Validation metrics tracking

## Code Structure

The implementation is organized into several core modules:

### Training Pipeline (`src/train.py`)
- Main training loop with distributed setup
- Gradient accumulation and mixed precision
- Regular validation and checkpointing
- Progress logging with tqdm

### Dataset Handling (`src/dataset.py`)
```python
class TextDataset:
    """Efficient text dataset with dynamic padding
    Handles both masked and causal language modeling"""
```
- Custom collation for batching
- Automatic tokenization and padding
- Configurable sequence length

### Model Management (`src/model.py`)
- Loads models from Hugging Face hub
- Supports various architectures (BERT, GPT, etc)
- Handles tokenizer configuration

### Utilities (`src/utils.py`)
- Reproducibility (seed setting)
- Config loading/parsing
- Checkpoint management
- Logging setup

## Directory Structure

```
.
├── config/         # Training configurations
├── data/          # Dataset files
├── notebooks/     # Analysis notebooks
├── scripts/       # Training scripts
└── src/           # Core implementation
    ├── train.py
    ├── dataset.py
    ├── model.py
    └── utils.py
```

## Key Features

**Distributed Training**
- Accelerate for multi-GPU coordination
- Automatic batch size scaling
- Gradient synchronization

**Data Pipeline**
- Efficient data loading
- Dynamic padding
- Configurable preprocessing

**Training Loop**
- Mixed precision support
- Gradient accumulation
- Regular validation
- Progress tracking
- Checkpointing

## License

MIT
