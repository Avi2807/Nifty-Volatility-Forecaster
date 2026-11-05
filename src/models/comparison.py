"""What the comparison models share: where they're saved, their freeze record and their training log.

Their checkpoints live apart from the Transformer's, so its freeze record never changes. Nothing here
sees the test set: forecasts for it are made once, by src/eval/compare.py, after this record is written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import RunPaths

COMPARISON_MODELS = ("garch", "mlp", "lstm")


def model_folder(paths: RunPaths) -> Path:
    folder = paths.artifacts / "comparison_models"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _log_path(paths: RunPaths) -> Path:
    return paths.artifacts / "comparison_training.json"


def _freeze_path(paths: RunPaths) -> Path:
    return paths.artifacts / "frozen_comparison_models.json"


def read_log(paths: RunPaths) -> dict:
    path = _log_path(paths)
    if not path.exists():
        return {"data_source": paths.data_source, "models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def record_training(paths: RunPaths, name: str, entry: dict) -> None:
    """Add one model to the log, then refresh the checksums of everything saved so far."""
    log = read_log(paths)
    log["models"][name] = entry
    _log_path(paths).write_text(json.dumps(log, indent=2, allow_nan=False), encoding="utf-8")
    record = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(model_folder(paths).iterdir())}
    _freeze_path(paths).write_text(json.dumps(record, indent=2), encoding="utf-8")


def verify_comparison_frozen(paths: RunPaths) -> None:
    """Refuse to score if a saved comparison model differs from the one recorded at the end of its training."""
    record = json.loads(_freeze_path(paths).read_text(encoding="utf-8"))
    for name, digest in record.items():
        if hashlib.sha256((model_folder(paths) / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Comparison model {name} has changed since it was frozen")
