# Binance Spot Data Acquisition Report

Report date: September 12, 2025
Author: Ernest

---

## 1. Summary

This report explains how we build a **reusable Binance spot price history** for research on Hyperliquid perpetual contracts, while keeping the asset universe aligned with Hyperliquid perps. Hyperliquid’s public trade archive starts on 2025-03-22, so we use a longer Binance history to approximate earlier price action. The workflow has two steps: first, fetch the active Hyperliquid perp list and **normalize names**, then take the intersection with Binance **/USDT** spot bases to produce the target asset list. Next, download spot OHLCV for **user-defined start/end dates** and **multiple timeframes**, save as Parquet, and support incremental runs with resume. The goal is to produce **clean, aligned, backtest-ready** price series.

---

## 2. Rationale and Feasibility

Using Binance spot to approximate Hyperliquid perp prices is **reasonable**. Hyperliquid’s pricing uses a basket of exchanges as anchors, with a higher weight on Binance. The funding reference focuses on spot baskets, and the mark price also includes perp mid prices from several exchanges. Given this structure, Binance spot is a practical proxy for most **correlation and cointegration** studies, especially when we perform a **return-scale calibration** on the overlapping period (for example a simple linear regression or a constant/EMA shift), and then apply the calibration to earlier dates. Note that perps and spot have a **basis** and **funding** effect. Short-term gaps can be larger in stress events. It is safer to model **returns** instead of levels and to include an error term in backtests.

The official docs state:

1. **Oracle (spot)** is published every 3 seconds as a **weighted median of spot prices** from several centralized exchanges, with weights: Binance=3, OKX=2, Bybit=2, Kraken=1, Kucoin=1, Gate=1, MEXC=1. Hyperliquid spot is only included for a few assets; for assets like BTC with deep external spot liquidity, Hyperliquid spot is **not** included. This oracle is used for **funding** and is also a component of **mark price**. [Oracle | Hyperliquid Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/hypercore/oracle)
2. **Mark price** is the **median** of three items:
   • Oracle price plus a 150-second EMA of the diff between Hyperliquid mid and oracle;
   • The median of Hyperliquid best bid, best ask, and last trade;
   • A weighted median of **perp mid prices** from several exchanges (weights: Binance=3, OKX=2, Bybit=2, Gate=1, MEXC=1).
   Thus mark price also gives Binance a higher weight, but it is not a single-source anchor; it is a multi-source design. [Robust price indices | Hyperliquid Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/robust-price-indices)

Binance spot history is stable to fetch through CCXT and supports many timeframes. Our scripts expose `--start` and `--end` to control the sample window. Files are named `{SYMBOL}_{TIMEFRAME}.parquet`; existing files are skipped for easy batching. Timestamps are UTC with de-duplication, and we keep only `open/high/low/close/volume` for direct downstream use. **Name differences** between Hyperliquid and Binance are handled in `create_target_list.py` by **normalization**: first use `/info meta` alias mapping; second, apply a safe prefix strip (only if the stripped name really exists in Binance bases); finally, fall back to the raw name. This lowers mismatch and omission risk.

---

## 3. Data Sources and Key Challenges

We use two sources: Hyperliquid `/info` (`allMids` and `meta`) for the active perp list and alias context, and Binance spot markets from CCXT filtered to **/USDT** pairs. The main challenges are **name normalization** and **date-range control**. Normalization uses alias mapping plus safe prefix removal. Date-range control uses CLI arguments and strict **end-exclusive** trimming in the download loop so no rows beyond `--end` are written.

---

## 4. Pipeline Design and Implementation

The pipeline has two scripts:

* `create_target_list.py`
  Fetch `allMids` from Hyperliquid, build a “chain-level name → standard name” map from `meta`, normalize each raw symbol, then intersect with Binance /USDT bases. Write the sorted result to `target_crypto_list.txt`.

* `get_binance_target_data.py`
  Read the target list and fetch OHLCV by `--start/--end/--timeframes`. Pagination advances by “last timestamp + 1 ms” to avoid repeats, and trims results with an end-exclusive rule. Data use UTC as the index, are de-duplicated, and keep only `open, high, low, close, volume`. Each (symbol, timeframe) is saved to one Parquet file. Existing files are skipped to support incremental and resumed runs.

Time and parameter rules:
(1) All timestamps are **UTC** for storage and display.
(2) CLI: `--start YYYY-MM-DD` is inclusive; `--end YYYY-MM-DD` is **exclusive**. If `--end` is missing, the script fetches up to now.
(3) Pagination: up to 1000 bars per call; advance by last timestamp + 1 ms to avoid gaps and duplicates.
(4) De-dup and columns: de-dup on `datetime`, keep the first; keep `open, high, low, close, volume`. `volume` is in the **base asset**.
(5) Idempotency: files are named `{SYMBOL}_{TIMEFRAME}.parquet`. If a file exists, it is skipped. This supports batching and resume.

---

## 5. Final Dataset Specification

Data are stored as Parquet under a chosen directory such as `./binance_data/`.
Each file is named `{SYMBOL}_{TIMEFRAME}.parquet`. The index is UTC `datetime`.
Columns are `open`, `high`, `low`, `close`, `volume`. `volume` is the **base asset amount** (for example BTC for BTC/USDT).
Use `pyarrow` for best Parquet I/O, or `fastparquet` as an alternative.

---

## 6. Usage Guide

Run `create_target_list.py` to produce `target_crypto_list.txt` (one upper-case base per line).
Then run `get_binance_target_data.py`. Supported flags include
`--start YYYY-MM-DD`,
`--end YYYY-MM-DD` (end is exclusive),
`--timeframes`,
`--data-dir`,
`--quote`,
`--exchange`.
If `--end` is not set, the script fetches up to the current time. To read data, use `pandas.read_parquet`.

Example:

```bash
# Build the target list (one-time)
python create_target_list.py

# Download 1m/5m/1h from 2024-01-01 to 2025-01-01 (end exclusive)
python get_binance_target_data.py --start 2024-01-01 --end 2025-01-01 \
  --timeframes 1m 5m 1h --data-dir ./binance_data
```

---

## 7. Scope and Limitations

This dataset only contains Binance spot OHLCV. It does not include funding, order books, or liquidation events. It is a **reasonable but bounded** proxy for Hyperliquid perps: basis and funding can cause short-term gaps, and residuals may be larger during stress. Calibrate on the overlapping period in **return space**, and model residuals or add noise in backtests to reduce bias. Name normalization covers common aliases and prefixes, but a few new assets may need updates later.

---

## 8. Maintenance and Extensions

Refresh the `meta` alias map regularly to keep normalization accurate. If you want a closer match to the Hyperliquid oracle, combine multiple spot sources with the public weights and then calibrate to Hyperliquid. Add a daily append mode for incremental updates. Add simple quality checks (time coverage, gap rate, outlier prices) and log any issues for stable long-term runs.

---

## 9. Appendix

Core scripts: `create_target_list.py` and `get_binance_target_data.py`.
Suggested dependencies: `pandas`, `ccxt`, `tqdm`, `pyarrow`, `requests`.
Pin versions as needed to ensure repeatability across environments.
