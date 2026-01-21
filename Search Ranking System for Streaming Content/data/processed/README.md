# Processed Data Directory

This directory contains **derived datasets** produced during preprocessing and feature engineering.

## Typical Artifacts
- `sessions.parquet` – simulated search sessions
- `features.parquet` – model-ready ranking features

## How Files Are Generated
```bash
python -m src.make_dataset
python -m src.features
