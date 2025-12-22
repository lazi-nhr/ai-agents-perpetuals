from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass
class RuntimeConfig:
    interval: str = "1h"
    candle_limit: int = 250
    lookback: int = 30
    notional_usd: float = 100.0
    tick_seconds: int = 60
    default_model_path: Path = Path("models/best_model.zip")
