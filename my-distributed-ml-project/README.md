# Distributed Machine Learning Project

This project implements a distributed machine learning training pipeline.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Download and preprocess data following instructions in `data/README.md`

## Usage

### Single Node Training
```bash
./scripts/run_single.sh
```

### Distributed Training
```bash
./scripts/run_distributed.sh
```

## Project Structure

- `configs/`: Configuration files
- `data/`: Data processing scripts and instructions
- `src/`: Main source code
- `notebooks/`: Analysis notebooks
- `scripts/`: Training scripts

## License

MIT
