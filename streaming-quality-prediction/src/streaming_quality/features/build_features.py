from __future__ import annotations
import argparse
import pandas as pd
import numpy as np

from streaming_quality.constants import ID_COLS, TIME_COL

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    # Simple but realistic transformations
    out = df.copy()
    out["log_throughput"] = np.log1p(out["throughput_mbps"])
    out["log_rtt"] = np.log1p(out["rtt_ms"])
    out["loss_x_rtt"] = out["packet_loss"] * out["rtt_ms"]
    out["demand_mbps"] = out["bitrate_kbps"] / 1000.0
    out["demand_over_throughput"] = out["demand_mbps"] / (out["throughput_mbps"] + 1e-6)
    out["is_4k"] = (out["resolution"] == "4k").astype(int)
    out["is_tv"] = (out["device"] == "tv").astype(int)
    out["is_mobile"] = (out["device"] == "mobile").astype(int)
    out["codec_efficiency"] = out["codec"].map({"h264": 0.0, "hevc": 0.2, "av1": 0.35}).fillna(0.0)
    return out

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out[TIME_COL], errors="coerce", utc=True)
    out["hour"] = t.dt.hour.astype("Int64")
    out["dow"] = t.dt.dayofweek.astype("Int64")
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    df = add_session_features(df)
    df = add_time_features(df)

    # Ensure IDs exist
    for c in ID_COLS + [TIME_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df.to_parquet(args.out, index=False)
    print(f"Features saved -> {args.out} ({len(df):,} rows)")

if __name__ == "__main__":
    main()
