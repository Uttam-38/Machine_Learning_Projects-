from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")
    artifacts: Path = Path("artifacts")
    models: Path = Path("artifacts/models")
    reports: Path = Path("artifacts/reports")

def ensure_dirs(paths: Paths) -> None:
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_json(path: str | os.PathLike, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_json(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
