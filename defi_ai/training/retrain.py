from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from defi_ai.core.config import get_paths
from defi_ai.utils.json_io import atomic_write_json
from defi_ai.utils.tracing import traceable

@dataclass(frozen=True)
class RetrainResult:
    new_model_path: Path
    version: str

@traceable(name="training:retrain", tags=["training", "automation"])
def retrain_from_checkpoint(checkpoint_path: Path, output_dir: Path | None = None) -> RetrainResult:
    """Production-safe retraining hook.

    This repo provides a *job boundary* suitable for LangSmith automation.
    The default behavior copies the current model checkpoint into a new versioned
    path and updates `artifacts/models/active_model.json`.

    Replace the copy step with your actual RL training pipeline when ready.
    """
    paths = get_paths()
    if output_dir is None:
        output_dir = paths.artifacts_dir / "models" / "versions"

    output_dir.mkdir(parents=True, exist_ok=True)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    new_path = output_dir / f"ppo_model_{version}.zip"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    shutil.copy2(checkpoint_path, new_path)

    atomic_write_json(
        paths.model_pointer_file,
        {"path": str(new_path), "version": version, "source_checkpoint": str(checkpoint_path)},
    )

    return RetrainResult(new_model_path=new_path, version=version)
