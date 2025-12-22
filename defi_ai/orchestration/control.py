from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from defi_ai.utils.json_io import read_json

@dataclass(frozen=True)
class ControlState:
    paused: bool = False
    force_pair: Optional[tuple[str, str]] = None
    force_model_path: Optional[str] = None

def load_control(control_file: Path) -> ControlState:
    data = read_json(control_file, default={}) or {}
    paused = bool(data.get("paused", False))
    fp = data.get("force_pair")
    force_pair = None
    if isinstance(fp, dict) and fp.get("asset1") and fp.get("asset2"):
        force_pair = (str(fp["asset1"]), str(fp["asset2"]))
    force_model_path = data.get("force_model_path")
    if force_model_path is not None:
        force_model_path = str(force_model_path)
    return ControlState(paused=paused, force_pair=force_pair, force_model_path=force_model_path)
