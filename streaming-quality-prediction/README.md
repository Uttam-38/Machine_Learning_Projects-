# Streaming Quality Prediction (QoE) — Offline Training + Online Inference + Drift Monitoring

Predict **streaming Quality of Experience (QoE)** from session/network/player telemetry:
- **Targets**
  - `rebuffer_ratio` (regression)
  - `startup_time_ms` (regression)
  - `quality_label` (classification: GOOD / OK / BAD)

## Why this stands out (production ML, not just a notebook)
- Feature pipeline with leakage-safe splits (session/user aware)
- Multi-task modeling: regressions + classification
- Model registry + reproducible artifacts
- FastAPI microservice for online predictions
- Drift detection (PSI + Wasserstein) with a simple monitoring report
- CI with unit tests + linting-style checks

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 1) Generate synthetic telemetry (replace later with real logs)
python -m streaming_quality.data.generate_synthetic --out data/processed/sessions.parquet --n 200000

# 2) Build features
python -m streaming_quality.features.build_features --in data/processed/sessions.parquet --out data/processed/features.parquet

# 3) Train + evaluate + register best model
python -m streaming_quality.models.train --data data/processed/features.parquet
python -m streaming_quality.models.evaluate --data data/processed/features.parquet

# 4) Run API
uvicorn streaming_quality.api.app:app --reload --port 8000

# Example request
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "session_id":"s1","user_id":"u1","device":"tv","cdn":"A",
  "throughput_mbps":12.4,"rtt_ms":32,"packet_loss":0.01,"jitter_ms":6,
  "bitrate_kbps":2800,"resolution":"1080p","fps":60,"codec":"h264",
  "buffer_level_s":12.0,"dropped_frames":10,"timestamp":"2026-01-02T10:00:00Z"
}'
