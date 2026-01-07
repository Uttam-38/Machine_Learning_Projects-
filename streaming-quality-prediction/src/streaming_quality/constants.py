TARGETS = ["rebuffer_ratio", "startup_time_ms", "quality_label"]

CATEGORICAL_COLS = ["device", "cdn", "resolution", "codec"]
NUMERIC_COLS = [
    "throughput_mbps", "rtt_ms", "packet_loss", "jitter_ms",
    "bitrate_kbps", "fps", "buffer_level_s", "dropped_frames",
]
ID_COLS = ["session_id", "user_id"]
TIME_COL = "timestamp"
