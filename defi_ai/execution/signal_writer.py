from __future__ import annotations

from pathlib import Path
from typing import Any

from defi_ai.utils.json_io import atomic_write_json

def write_signal(signal_path: Path, signal: dict[str, Any]) -> None:
    atomic_write_json(signal_path, signal)
