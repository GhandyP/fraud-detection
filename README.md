# Fraud Detection Program

Organized scaffolding for a fraud detection pipeline using scikit-learn.

## Structure

- `data/` raw and processed datasets
- `models/` trained models and artifacts
- `src/` core pipeline, data loading, preprocessing, modeling
- `config/` configuration files

## Quick Start

1. Place your dataset at `data/raw/creditcard.csv`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training pipeline:

```bash
python -m src.pipeline
```
