from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from defi_ai.utils.json_io import read_json

@dataclass(frozen=True)
class ActivePair:
    asset1: str
    asset2: str
    alpha: float | None = None
    beta: float | None = None
    score: float | None = None
    corr: float | None = None
    p_value: float | None = None

def load_active_pair(pairs_file: Path) -> Optional[ActivePair]:
    data = read_json(pairs_file, default=None)
    if not data or "pairs" not in data or not data["pairs"]:
        return None
    p = data["pairs"][0]
    try:
        return ActivePair(
            asset1=str(p["asset1"]),
            asset2=str(p["asset2"]),
            alpha=float(p.get("alpha")) if p.get("alpha") is not None else None,
            beta=float(p.get("beta")) if p.get("beta") is not None else None,
            score=float(p.get("score")) if p.get("score") is not None else None,
            corr=float(p.get("corr")) if p.get("corr") is not None else None,
            p_value=float(p.get("p_value")) if p.get("p_value") is not None else None,
        )
    except Exception:
        return None
