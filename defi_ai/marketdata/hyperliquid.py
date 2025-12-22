from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import requests

Interval = Literal["1m", "5m", "15m", "1h", "4h", "1d"]

@dataclass(frozen=True)
class CandleConfig:
    interval: Interval = "1h"
    limit: int = 200

class HyperliquidClient:
    """Minimal Hyperliquid candleSnapshot client (REST).

    This is intentionally simple to keep the trading loop deterministic.
    """

    def __init__(self, base_url: str = "https://api.hyperliquid.xyz/info", timeout_s: int = 20):
        self.base_url = base_url
        self.timeout_s = timeout_s

    def fetch_candles(self, coin: str, cfg: CandleConfig) -> pd.DataFrame:
        # candleSnapshot requires startTime (ms)
        # For 1h candles, approximate start time from limit.
        seconds_per_bar = self._seconds_per_bar(cfg.interval)
        start_time_ms = int(time.time() * 1000) - (cfg.limit * seconds_per_bar * 1000)

        body = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": cfg.interval, "startTime": start_time_ms},
        }

        r = requests.post(self.base_url, headers={"Content-Type": "application/json"}, json=body, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        # keys: c/h/l/v
        for col, key in [("close", "c"), ("high", "h"), ("low", "l"), ("volume", "v")]:
            df[col] = df[key].astype(float)

        return df.set_index("timestamp").sort_index()

    @staticmethod
    def _seconds_per_bar(interval: Interval) -> int:
        if interval.endswith("m"):
            return int(interval[:-1]) * 60
        if interval.endswith("h"):
            return int(interval[:-1]) * 3600
        if interval.endswith("d"):
            return int(interval[:-1]) * 86400
        raise ValueError(f"Unsupported interval: {interval}")
