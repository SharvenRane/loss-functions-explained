# Loss Functions Explained

Focal loss, Dice loss, Tversky, contrastive losses — implemented and visually compared

`loss-functions` `deep-learning` `focal-loss` `dice-loss` `pytorch`

## Overview

This repository implements a complete pipeline for **loss functions explained**, covering
data preprocessing, model training, evaluation, and deployment.

## Features

- Clean, modular PyTorch implementation
- Reproducible experiments with MLflow tracking
- Comprehensive evaluation with standard benchmarks
- ONNX export for production deployment
- Detailed documentation and usage examples

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/loss-functions-explained.git
cd loss-functions-explained
pip install -r requirements.txt
```

## Quick Start

```python
from src.model import Model
from src.trainer import Trainer
from src.config import Config

config = Config.from_yaml("configs/default.yaml")
model = Model(config)
trainer = Trainer(model, config)
trainer.train()
```

## Project Structure

```
loss-functions-explained/
├── src/
│   ├── model.py        # Model architecture
│   ├── dataset.py      # Data loading and preprocessing
│   ├── trainer.py      # Training loop
│   ├── evaluate.py     # Evaluation metrics
│   └── utils.py        # Helper utilities
├── configs/
│   └── default.yaml    # Default configuration
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Results

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| Baseline | Standard | Primary | - |
| Ours | Standard | Primary | - |

## Usage

```bash
# Train
python train.py --config configs/default.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/best.pth

# Export to ONNX
python export.py --checkpoint checkpoints/best.pth
```

## References

- Relevant papers and resources for loss functions explained

## License

MIT
