from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from defi_ai.utils.json_io import read_json

@dataclass(frozen=True)
class ActiveModel:
    path: Path
    version: str = "unknown"

def load_active_model(model_pointer_file: Path, default_model_path: Path) -> ActiveModel:
    data = read_json(model_pointer_file, default={}) or {}
    p = data.get("path")
    v = data.get("version", "unknown")
    if p:
        return ActiveModel(path=Path(str(p)), version=str(v))
    return ActiveModel(path=default_model_path, version=str(v))
