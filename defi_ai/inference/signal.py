from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from defi_ai.inference.ppo_runner import PPOModelRunner

@dataclass(frozen=True)
class SignalConfig:
    bar_timeframe: str = "1h"
    exchange: str = "binance_perpetual"
    fresh_for_seconds: int = 3600
    version: int = 1

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_signal(
    *,
    runner: PPOModelRunner,
    model_path: Path,
    asset1: str,
    asset2: str,
    pair_features: dict[str, list[float]],
    lookback: int,
    notional_usd: float,
    cfg: SignalConfig,
) -> dict:
    # ensure runner points to current model
    runner.model_path = model_path
    w1, w2, action = runner.predict_weights(pair_features, lookback)
    return {
        "timestamp": now_iso_utc(),
        "bar_timeframe": cfg.bar_timeframe,
        "pair": {"asset1": asset1, "asset2": asset2},
        "markets": {"exchange": cfg.exchange, "asset1": f"{asset1}-USDT", "asset2": f"{asset2}-USDT"},
        "weights": {"asset1": w1, "asset2": w2},
        "notional_usd": float(notional_usd),
        "fresh_for_seconds": int(cfg.fresh_for_seconds),
        "version": int(cfg.version),
        "debug": {"raw_action": action},
    }
