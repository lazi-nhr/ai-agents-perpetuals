from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    artifacts_dir: Path
    control_file: Path
    pairs_file: Path
    model_pointer_file: Path
    signal_file: Path

def get_repo_root() -> Path:
    # Works whether run as module or script.
    return Path(__file__).resolve().parents[2]

def get_paths() -> Paths:
    root = get_repo_root()
    artifacts = root / "artifacts"
    return Paths(
        repo_root=root,
        artifacts_dir=artifacts,
        control_file=artifacts / "control" / "control.json",
        pairs_file=artifacts / "pairs" / "active_pairs.json",
        model_pointer_file=artifacts / "models" / "active_model.json",
        signal_file=root / "signal.json",
    )

def env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None else default
