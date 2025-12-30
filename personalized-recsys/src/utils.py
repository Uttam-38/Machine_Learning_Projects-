import os
import json
from typing import Any, Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def pretty_print_metrics(metrics: Dict[str, float]) -> str:
    lines = []
    for k, v in metrics.items():
        lines.append(f"{k:>15}: {v:.4f}")
    return "\n".join(lines)
