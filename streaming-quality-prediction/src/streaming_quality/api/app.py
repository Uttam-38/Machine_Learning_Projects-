from __future__ import annotations
import glob
from fastapi import FastAPI
from pydantic import BaseModel, Field
from streaming_quality.models.registry import load_bundle
from streaming_quality.inference.predict import predict_one

def latest_model_path(models_dir: str = "artifacts/models") -> str:
    paths = sorted(glob.glob(f"{models_dir}/model_*.joblib"))
    if not paths:
        raise FileNotFoundError("Train a model first: python -m streaming_quality.models.train ...")
    return paths[-1]

bundle = load_bundle(latest_model_path())

app = FastAPI(title="Streaming QoE Prediction API", version=bundle.version)

class SessionRequest(BaseModel):
    session_id: str
    user_id: str
    device: str
    cdn: str
    throughput_mbps: float = Field(ge=0)
    rtt_ms: float = Field(ge=0)
    packet_loss: float = Field(ge=0, le=1)
    jitter_ms: float = Field(ge=0)
    bitrate_kbps: float = Field(ge=0)
    resolution: str
    fps: int = Field(ge=1)
    codec: str
    buffer_level_s: float = Field(ge=0)
    dropped_frames: int = Field(ge=0)
    timestamp: str

    # If feature builder added more columns, you can send them too (optional)
    hour: int | None = None
    dow: int | None = None
    is_weekend: int | None = None
    log_throughput: float | None = None
    log_rtt: float | None = None
    loss_x_rtt: float | None = None
    demand_mbps: float | None = None
    demand_over_throughput: float | None = None
    is_4k: int | None = None
    is_tv: int | None = None
    is_mobile: int | None = None
    codec_efficiency: float | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model_version": bundle.version}

@app.post("/predict")
def predict(req: SessionRequest):
    row = req.model_dump()
    # Ensure missing engineered features don't break selection
    for col in bundle.feature_columns:
        row.setdefault(col, None)
    # Lightweight fallbacks for missing engineered features
    if row.get("log_throughput") is None and row.get("throughput_mbps") is not None:
        import math
        row["log_throughput"] = math.log1p(row["throughput_mbps"])
    if row.get("log_rtt") is None and row.get("rtt_ms") is not None:
        import math
        row["log_rtt"] = math.log1p(row["rtt_ms"])
    if row.get("demand_mbps") is None and row.get("bitrate_kbps") is not None:
        row["demand_mbps"] = row["bitrate_kbps"] / 1000.0

    return predict_one(bundle, row)
