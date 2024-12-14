# Distributed Machine Learning Project

This project implements a distributed machine learning training pipeline using PyTorch and the Transformers library for distributed training of language models.

## Features
- Distributed training support across multiple GPUs/machines
- Configuration-based experiment management
- Integrated logging and monitoring with Weights & Biases
- Performance profiling tools
- Model interpretation notebooks

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/my-distributed-ml-project.git
cd my-distributed-ml-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your experiment in `configs/default_config.yaml`

4. Download and preprocess data:
```bash
cd data
./download_data.sh  # Instructions in data/README.md
```

## Usage

### Single GPU Training
```bash
./scripts/run_single.sh
```

### Distributed Training
```bash
./scripts/run_distributed.sh
```

## Project Structure
```
my-distributed-ml-project/
├─ configs/         # Configuration files
├─ data/           # Data handling scripts
├─ src/            # Main source code
├─ notebooks/      # Analysis notebooks
└─ scripts/        # Training scripts
```

## License
MIT

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.
