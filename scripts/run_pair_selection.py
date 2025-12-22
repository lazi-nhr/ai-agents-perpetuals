from __future__ import annotations

import argparse
from datetime import datetime, timezone

from defi_ai.core.config import get_paths
from defi_ai.marketdata.hyperliquid import HyperliquidClient
from defi_ai.pairs.selector import PairSelectionConfig, select_top_pairs
from defi_ai.utils.json_io import atomic_write_json
from defi_ai.utils.tracing import traceable


DEFAULT_UNIVERSE = [
    "AAVE", "ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC", "DOGE",
    "DOT", "ENA", "ETC", "ETH", "HBAR", "LINK", "LTC", "NEAR", "SOL", "SUI",
    "TON", "TRX", "UNI", "WLD", "XLM", "XRP"
]


@traceable(name="pairs:select_top_pairs", tags=["pairs", "stats"])
def run_selection(top_k: int, interval: str) -> dict:
    paths = get_paths()
    client = HyperliquidClient()
    cfg = PairSelectionConfig(assets=DEFAULT_UNIVERSE, top_k=top_k, interval=interval)
    ranked = select_top_pairs(cfg, client)

    out = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval": interval,
        "top_k": top_k,
        "pairs": [
            {
                "asset1": p.asset1,
                "asset2": p.asset2,
                "score": p.score,
                "corr": p.corr,
                "p_value": p.p_value,
                "alpha": p.alpha,
                "beta": p.beta,
            }
            for p in ranked
        ],
    }
    atomic_write_json(paths.pairs_file, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--interval", type=str, default="1h")
    args = parser.parse_args()

    res = run_selection(top_k=args.top_k, interval=args.interval)
    if res["pairs"]:
        p0 = res["pairs"][0]
        print(f"Active pair: {p0['asset1']}-{p0['asset2']} score={p0['score']:.4f}")
    else:
        print("No cointegrated pairs found at current thresholds.")


if __name__ == "__main__":
    main()
