# Robustness Under Distribution Shift (Streaming QoE)

This repo demonstrates **robust ML under distribution shift** using a **streaming QoE-like dataset** (synthetic but realistic).
It evaluates how models degrade when deployed into new regions/devices/networks, and implements practical robustness baselines:

- **ERM** (standard training)
- **Importance Weighting (IW)** (covariate shift)
- **CORAL** (feature alignment)
- **GroupDRO** (worst-group robustness)

## Why this stands out (Netflix-relevant)
Streaming systems often face:
- **Region shifts** (ISP/routing differences)
- **Device shifts** (mobile vs TV, CPU constraints)
- **Network shifts** (RTT/throughput changes)
- **Policy shifts** (ABR behavior updates)

This project measures:
- **Average performance**
- **Worst-group performance** (critical for fairness/reliability)
- **Calibration** (ECE) — important for decision thresholds in production

---

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install -e .

### 2) Run one experiment
python scripts/run_experiment.py --config configs/default.yaml --method erm
python scripts/run_experiment.py --config configs/default.yaml --method groupdro
python scripts/run_experiment.py --config configs/default.yaml --method coral
python scripts/run_experiment.py --config configs/default.yaml --method iw

### 3) Results
runs/<timestamp>/metrics.json
runs/<timestamp>/preds.csv
runs/<timestamp>/plots/
