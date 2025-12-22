from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from defi_ai.marketdata.hyperliquid import CandleConfig, HyperliquidClient
from defi_ai.pairs.cointegration import PairScore, rank_pairs

@dataclass(frozen=True)
class PairSelectionConfig:
    assets: list[str]
    interval: str = "1h"
    limit: int = 250
    min_points: int = 120
    corr_threshold: float = 0.85
    pvalue_threshold: float = 0.05
    top_k: int = 5
    per_request_sleep_s: float = 0.05

def select_top_pairs(cfg: PairSelectionConfig, client: HyperliquidClient) -> list[PairScore]:
    price_series: dict[str, pd.Series] = {}
    c_cfg = CandleConfig(interval=cfg.interval, limit=cfg.limit)
    for coin in cfg.assets:
        try:
            df = client.fetch_candles(coin, c_cfg)
            if not df.empty and "close" in df.columns:
                price_series[coin] = df["close"]
        finally:
            time.sleep(cfg.per_request_sleep_s)

    return rank_pairs(
        price_series,
        min_points=cfg.min_points,
        corr_threshold=cfg.corr_threshold,
        pvalue_threshold=cfg.pvalue_threshold,
        top_k=cfg.top_k,
    )
