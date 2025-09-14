# filename: get_binance_target_data.py
#
# Purpose:
# Download Binance Spot OHLCV for a target symbol list across given timeframes
# and a user-defined date range, then write one Parquet per (symbol,timeframe).
#
# Changes in this version:
# - For each timeframe, create a subfolder under data_dir: e.g., ohlcv_1m/
# - File name now includes start and end (end is exclusive):
#     {SYMBOL}_ohlcv_{YYYY-MM-DD}_{YYYY-MM-DD}.parquet
#
# Requirements:
#   pip install pandas ccxt tqdm pyarrow
#   # or use 'fastparquet' instead of 'pyarrow'

import argparse
import os
import time
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from tqdm import tqdm


def load_symbols_from_file(filepath: str = "target_crypto_list.txt"):
    """
    Read target symbols from a text file. One symbol per line.
    Empty lines are ignored. Duplicates are removed. Symbols are uppercased.
    Returns a sorted list of symbols, or None on failure.
    """
    if not os.path.exists(filepath):
        print(f"Error: target list file '{filepath}' does not exist.")
        print("Please run 'create_target_list.py' to generate it first.")
        return None
    try:
        with open(filepath, "r") as f:
            symbols = sorted(list({line.strip().upper() for line in f if line.strip()}))
        print(f"Loaded {len(symbols)} target symbols from {filepath}.")
        return symbols
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None


def parse_date(date_str: str) -> datetime:
    """Parse 'YYYY-MM-DD' into a datetime. Raise ValueError on bad input."""
    return datetime.strptime(date_str, "%Y-%m-%d")


class CryptoDownloader:
    """
    A thin wrapper around a CCXT exchange for batched OHLCV downloads and storage.
    """

    def __init__(self, exchange_name: str = "binance", data_dir: str = "data"):
        """
        Build a CCXT exchange instance, set Spot mode, and prepare the output directory.
        """
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({"options": {"defaultType": "spot"}, "enableRateLimit": True})
            print(f"Exchange initialized: {self.exchange.id}")
        except AttributeError:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")

    def _fetch_ohlcv_robust(self, symbol_pair: str, timeframe: str, since: int, limit: int = 1000):
        """
        Fetch one page of OHLCV. Retry on network errors. Log and skip on exchange errors.
        Returns a list of candles or None on failure.
        """
        try:
            if self.exchange.has.get("fetchOHLCV"):
                return self.exchange.fetch_ohlcv(symbol_pair, timeframe, since, limit)
            else:
                tqdm.write(f"Warning: {self.exchange.id} does not support fetchOHLCV.")
                return None
        except ccxt.NetworkError as e:
            tqdm.write(f"Network error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            return self._fetch_ohlcv_robust(symbol_pair, timeframe, since, limit)
        except ccxt.ExchangeError as e:
            tqdm.write(f"Exchange error: {e} for {symbol_pair}")
            return None
        except Exception as e:
            tqdm.write(f"Unexpected error: {e}")
            return None

    def download_data(
        self,
        symbols,
        timeframes,
        start_date_str: str,
        end_date_str: str | None = None,
        quote_currency: str = "USDT",
    ):
        """
        For each (symbol, timeframe), download historical OHLCV in [start, end)
        and save to a Parquet file if any data is found. Existing files are skipped.
        Files are placed under data_dir/ohlcv_<timeframe>/ with names:
            {SYMBOL}_ohlcv_{START}_{END}.parquet
        """
        try:
            print("Loading markets from the exchange for validation...")
            self.exchange.load_markets()
            print("Markets loaded.")
        except Exception as e:
            print(f"Failed to load markets: {e}. Aborting.")
            return

        start_dt = parse_date(start_date_str)
        end_dt = parse_date(end_date_str) if end_date_str else datetime.now()

        if start_dt >= end_dt:
            print(f"Error: start date {start_date_str} must be earlier than end date {end_date_str}.")
            return

        # Labels used in file names
        start_label = start_dt.strftime("%Y-%m-%d")
        end_label = end_dt.strftime("%Y-%m-%d")

        since_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        total_tasks = len(symbols) * len(timeframes)

        with tqdm(total=total_tasks, desc="Overall progress") as pbar:
            for timeframe in timeframes:
                # Make timeframe subfolder, e.g., ohlcv_1m
                tf_tag = timeframe.replace("/", "")
                tf_dir = os.path.join(self.data_dir, f"ohlcv_{tf_tag}")
                os.makedirs(tf_dir, exist_ok=True)

                for symbol in symbols:
                    pbar.update(1)
                    symbol_pair = f"{symbol}/{quote_currency}"
                    pbar.set_description(f"Processing {symbol_pair} [{timeframe}]")

                    # New file name pattern with date range
                    filename = f"{symbol}_ohlcv_{start_label}_{end_label}.parquet"
                    filepath = os.path.join(tf_dir, filename)

                    if os.path.exists(filepath):
                        tqdm.write(f"Skip existing file: {os.path.relpath(filepath)}")
                        continue

                    all_ohlcv = []
                    current_since = since_ms

                    while True:
                        if current_since >= end_ms:
                            break

                        ohlcv = self._fetch_ohlcv_robust(symbol_pair, timeframe, current_since)
                        if not ohlcv:
                            break

                        # Respect end as an open interval (exclude rows >= end_ms)
                        trimmed = [row for row in ohlcv if row[0] < end_ms]
                        if trimmed:
                            all_ohlcv.extend(trimmed)

                        last_ts = ohlcv[-1][0]
                        # If exchange returned data that already reaches or passes end, stop.
                        if last_ts >= end_ms - 1 or len(trimmed) < len(ohlcv):
                            break

                        current_since = last_ts + 1  # advance by 1 ms to avoid duplicates

                    if all_ohlcv:
                        self.save_to_file(filepath, all_ohlcv)
                        tqdm.write(f"Saved {len(all_ohlcv)} rows -> {os.path.relpath(filepath)}")

    def save_to_file(self, filepath: str, ohlcv_data):
        """
        Convert raw OHLCV array to a DataFrame, add a UTC datetime index,
        drop duplicates, and write to the given Parquet path.
        """
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset="datetime", keep="first").set_index("datetime")
        df = df[["open", "high", "low", "close", "volume"]]
        df.to_parquet(filepath)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser.
    Examples:
      python get_binance_target_data.py --start 2024-01-01 --end 2025-01-01
      python get_binance_target_data.py --start 2024-01-01 --timeframes 1m 5m 1h
      python get_binance_target_data.py --symbols-file my_list.txt --data-dir ./binance_data
    """
    p = argparse.ArgumentParser(description="Download Binance Spot OHLCV for a target symbol list.")
    p.add_argument("--symbols-file", default="target_crypto_list.txt", help="Path to symbol list file.")
    p.add_argument("--exchange", default="binance", help="CCXT exchange id (default: binance).")
    p.add_argument("--quote", default="USDT", help="Quote currency (default: USDT).")
    p.add_argument(
        "--timeframes",
        nargs="+",
        default=["1m", "5m", "15m", "1h", "1d"],
        help="One or more CCXT timeframes, e.g., 1m 5m 1h (default: 1m 5m 15m 1h 1d).",
    )
    p.add_argument("--start", required=False, help="Start date (YYYY-MM-DD). Default: 365 days ago.")
    p.add_argument("--end", required=False, help="End date (YYYY-MM-DD), exclusive. Default: today.")
    p.add_argument("--data-dir", default="binance_data", help="Output directory for Parquet files.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    TARGET_SYMBOLS = load_symbols_from_file(args.symbols_file)

    if TARGET_SYMBOLS:
        start_str = args.start or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_str = args.end  # may be None -> will default to "now" inside download_data

        print("\n" + "=" * 50)
        print(">>> Run Configuration <<<")
        print(f"Symbols: {len(TARGET_SYMBOLS)}")
        print(f"Timeframes: {', '.join(args.timeframes)}")
        print(f"Start date: {start_str}")
        print(f"End date: {end_str or 'today'}")
        print(f"Quote: {args.quote}")
        print(f"Exchange: {args.exchange}")
        print(f"Output directory: {args.data_dir}")
        print("=" * 50 + "\n")

        downloader = CryptoDownloader(exchange_name=args.exchange, data_dir=args.data_dir)
        downloader.download_data(
            TARGET_SYMBOLS,
            args.timeframes,
            start_date_str=start_str,
            end_date_str=end_str,
            quote_currency=args.quote,
        )

        print("\nDownload job completed.")


# ------------------------------------------------------------------------------
# How to use (examples)
#
# 1) Prepare your target list:
#    Create "target_crypto_list.txt" with one symbol per line (e.g., BTC, ETH, SOL).
#
# 2) Install dependencies:
#    pip install pandas ccxt tqdm pyarrow
#    # or use 'fastparquet' instead of 'pyarrow'
#
# 3) Run with a custom date range (end is exclusive):
#    python get_binance_target_data.py --start 2024-08-22 --end 2025-08-22
#
# 4) Run with default start (365 days ago) and end (today):
#    python get_binance_target_data.py
#
# 5) Choose timeframes and output directory:
#    python get_binance_target_data.py --start 2024-06-01 --timeframes 1m 5m 1h 1d --data-dir ./binance_data
#
# 6) Output layout:
#    ./binance_data/
#      ├─ ohlcv_1m/
#      │   ├─ AAVE_ohlcv_2025-03-22_2025-08-22.parquet
#      │   └─ ...
#      ├─ ohlcv_1h/
#      │   └─ BTC_ohlcv_2024-01-01_2025-01-01.parquet
#      └─ ...
#
# 7) Switch quote currency or exchange (any CCXT exchange that supports fetchOHLCV):
#    python get_binance_target_data.py --quote USDC --exchange binanceus
# ------------------------------------------------------------------------------
