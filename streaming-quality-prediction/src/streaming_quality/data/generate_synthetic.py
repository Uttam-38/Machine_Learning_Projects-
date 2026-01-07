from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

DEVICES = ["mobile", "desktop", "tv", "tablet"]
CDNS = ["A", "B", "C"]
RES = ["480p", "720p", "1080p", "4k"]
CODECS = ["h264", "hevc", "av1"]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def make(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    user_id = rng.integers(1, 50000, size=n)
    session_id = np.arange(n)

    device = rng.choice(DEVICES, size=n, p=[0.45, 0.25, 0.20, 0.10])
    cdn = rng.choice(CDNS, size=n, p=[0.5, 0.3, 0.2])
    resolution = rng.choice(RES, size=n, p=[0.15, 0.35, 0.40, 0.10])
    codec = rng.choice(CODECS, size=n, p=[0.55, 0.25, 0.20])

    throughput_mbps = rng.lognormal(mean=2.1, sigma=0.55, size=n)  # ~8-25 typical
    rtt_ms = rng.lognormal(mean=3.3, sigma=0.35, size=n)           # ~20-60
    packet_loss = np.clip(rng.beta(1.2, 30, size=n) * 0.2, 0, 0.2)  # mostly low
    jitter_ms = np.clip(rng.lognormal(mean=1.7, sigma=0.45, size=n), 0, 80)

    fps = rng.choice([24, 30, 60], size=n, p=[0.12, 0.55, 0.33])
    bitrate_kbps = (
        (resolution == "480p") * rng.integers(600, 1400, size=n)
        + (resolution == "720p") * rng.integers(1200, 2600, size=n)
        + (resolution == "1080p") * rng.integers(2200, 5000, size=n)
        + (resolution == "4k") * rng.integers(7000, 16000, size=n)
    ).astype(int)

    # Latent "stress" of network vs content demand
    demand = bitrate_kbps / 1000.0
    stress = (demand / (throughput_mbps + 1e-6)) + 0.01 * (rtt_ms / 50) + 2.5 * packet_loss + 0.01 * jitter_ms

    # Buffer level inversely related to stress (with noise)
    buffer_level_s = np.clip(18 - 7 * stress + rng.normal(0, 2.0, size=n), 0, 30)

    # Rebuffer probability increases with stress and low buffer
    p_rebuffer = sigmoid(2.2 * (stress - 1.0) + 0.15 * (5 - buffer_level_s))
    rebuffer_ratio = np.clip(p_rebuffer * rng.lognormal(-2.0, 0.7, size=n), 0, 0.35)

    # Startup time worse on mobile + high RTT + CDN issues
    cdn_penalty = np.where(cdn == "C", 250, np.where(cdn == "B", 120, 0))
    device_penalty = np.where(device == "mobile", 150, np.where(device == "tv", 90, 0))
    startup_time_ms = np.clip(
        600 + 6.5 * rtt_ms + 200 * stress + cdn_penalty + device_penalty + rng.normal(0, 120, size=n),
        200, 8000
    )

    # Frame drops correlate with jitter/loss and high fps
    dropped_frames = np.clip((0.15 * jitter_ms + 220 * packet_loss + 0.02 * (fps == 60)) + rng.normal(0, 2.5, size=n), 0, 200).astype(int)

    # Quality label (business-friendly)
    quality_score = (
        -4.0 * rebuffer_ratio
        -0.00045 * startup_time_ms
        -0.005 * dropped_frames
        + 0.03 * np.log1p(buffer_level_s)
    )

    quality_label = np.where(quality_score > -0.6, "GOOD", np.where(quality_score > -1.2, "OK", "BAD"))

    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 60 * 60 * 24 * 14, size=n), unit="s")

    df = pd.DataFrame({
        "session_id": session_id.astype(str),
        "user_id": user_id.astype(str),
        "device": device,
        "cdn": cdn,
        "throughput_mbps": throughput_mbps,
        "rtt_ms": rtt_ms,
        "packet_loss": packet_loss,
        "jitter_ms": jitter_ms,
        "bitrate_kbps": bitrate_kbps,
        "resolution": resolution,
        "fps": fps,
        "codec": codec,
        "buffer_level_s": buffer_level_s,
        "dropped_frames": dropped_frames,
        "timestamp": ts.astype(str),
        "rebuffer_ratio": rebuffer_ratio,
        "startup_time_ms": startup_time_ms,
        "quality_label": quality_label,
    })

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = make(args.n, args.seed)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} rows -> {args.out}")

if __name__ == "__main__":
    main()
