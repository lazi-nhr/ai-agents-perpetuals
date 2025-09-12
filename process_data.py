# process_data.py — parse Hyperliquid raw fills (LZ4) into OHLCV parquet for spot and perp.
# Features: stream LZ4 files, normalize spot/perp names using /info when available,
# apply IQR-based outlier handling (cap or drop), batch-aggregate to reduce memory use,
# and save per-asset parquet files under spot/ and perp/ folders.
#
# Input: ./hyperliquid_data/raw_fills with subfolders like node_trades/hourly/YYYYMMDD/*.lz4
# Output: ./hyperliquid_data/processed_ohlcv_<tf>/spot and .../perp parquet files.
#
# Notes:
# - This script does not alter raw files. It only reads and writes parquet outputs.
# - Keep the same file layout when running (START_DATE / END_DATE range in main).
# - All runtime messages (prints) remain as in the original code.

import os
import lz4.frame
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import requests
import threading
from datetime import date, timedelta
import argparse
import re

# =========================
# Config & constants
# =========================
REQUEST_TIMEOUT = 10  # seconds for /info requests
OUTLIER_MIN_COUNT = 10  # minimum points to compute IQR-based capping/drop
OUTLIER_MODE = "cap"  # "cap" or "drop" (how to handle IQR outliers)
CHUNK_AGG_BATCH = 50  # how many intermediate blocks to concat per batch to reduce memory spikes
LZ4_BLOCK_SIZE = 1024 * 1024  # informational: lz4 frame stream block size note

# Create output subfolders for spot and perp under the given base folder.
def make_output_dirs(base):
    spot_dir = os.path.join(base, "spot")
    perp_dir = os.path.join(base, "perp")
    os.makedirs(spot_dir, exist_ok=True)
    os.makedirs(perp_dir, exist_ok=True)
    return spot_dir, perp_dir

# =========================
# Thread-safe print helper
# =========================
print_lock = threading.Lock()
def tprint(text: str):
    with print_lock:
        print(text)

# =========================
# Hyperliquid info helpers
# =========================
def post_info(payload: dict) -> dict | None:
    """
    Send a POST to https://api.hyperliquid.xyz/info with timeout and basic checks.
    Return parsed JSON (dict) on success, otherwise None.
    """
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            tprint(f"[WARN] /info returned non-dict for {payload}")
            return None
        return data
    except requests.exceptions.RequestException as e:
        tprint(f"[ERROR] /info failed for {payload}: {e}")
        return None
    except ValueError:
        tprint(f"[ERROR] /info invalid JSON for {payload}")
        return None

def load_meta_and_spotmeta() -> tuple[dict | None, dict | None]:
    """
    Fetch meta (perps) and spotMeta (spot) from the API.
    Either may be None on network failure; code will still run without them.
    """
    meta = post_info({"type": "meta"})
    spot_meta = post_info({"type": "spotMeta"})
    # token_details = post_info({"type": "tokenDetails"})  # optional, not used here
    return meta, spot_meta

def build_spot_maps(spot_meta: dict | None):
    """
    Build mapping structures for spot universe:
      - spot_index_to_universe : index -> raw universe entry
      - spot_name_to_index     : display name -> index
      - spot_index_to_display  : index -> preferred display string
    When spot_meta is missing, returned maps are empty.
    """
    spot_index_to_universe = {}
    spot_name_to_index = {}
    spot_index_to_display = {}

    if not spot_meta or "universe" not in spot_meta or "tokens" not in spot_meta:
        return spot_index_to_universe, spot_name_to_index, spot_index_to_display

    tokens = spot_meta["tokens"]  # list of token dicts
    universe = spot_meta["universe"]  # list of spot pairs

    for u in universe:
        # Safe extraction of expected fields
        idx = u.get("index")
        nm = u.get("name")
        is_canon = u.get("isCanonical", False)
        toks = u.get("tokens", [])

        if idx is None or nm is None:
            continue

        spot_index_to_universe[idx] = u
        spot_name_to_index[nm] = idx

        # Determine a user-friendly display name:
        # 1) use canonical name if provided and not @index form
        # 2) else try BASE/QUOTE using the token list
        # 3) otherwise fallback to "@{index}"
        if is_canon and not str(nm).startswith("@"):
            spot_index_to_display[idx] = nm
        else:
            disp = None
            if isinstance(toks, list) and len(toks) == 2:
                try:
                    base_tok = tokens[toks[0]]
                    quote_tok = tokens[toks[1]]
                    base_name = base_tok.get("name")
                    quote_name = quote_tok.get("name")
                    if base_name and quote_name:
                        disp = f"{base_name}/{quote_name}"
                except Exception:
                    disp = None
            spot_index_to_display[idx] = disp if disp else f"@{idx}"

    return spot_index_to_universe, spot_name_to_index, spot_index_to_display

def build_perp_maps(meta: dict | None):
    """
    Build perp mappings from meta.universe:
      - perp_id_to_name : id -> name
      - perp_name_to_id : name -> id
      - perp_delisted   : name -> bool
    Returns empty maps if meta is None.
    """
    perp_id_to_name = {}
    perp_name_to_id = {}
    perp_delisted = {}
    if not meta or "universe" not in meta:
        return perp_id_to_name, perp_name_to_id, perp_delisted

    for i, it in enumerate(meta["universe"]):
        name = it.get("name")
        if not name:
            continue
        perp_id_to_name[i] = name
        perp_name_to_id[name] = i
        perp_delisted[name] = bool(it.get("isDelisted", False))
    return perp_id_to_name, perp_name_to_id, perp_delisted

# =========================
# LZ4 streaming JSON reader
# =========================
def iter_lz4_lines(path: str):
    """
    Open an .lz4 file as a stream and yield decoded text lines.
    Falls back to latin1 on decode errors to be robust.
    """
    try:
        with lz4.frame.open(path, mode="rb") as f:
            for raw in f:
                if not raw:
                    continue
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield line.decode("utf-8")
                except UnicodeDecodeError:
                    # best-effort fallback
                    try:
                        yield line.decode("latin1")
                    except Exception:
                        continue
    except (lz4.frame.LZ4FrameError, FileNotFoundError) as e:
        tprint(f"[WARN] LZ4 open failed for {path}: {e}")

# =========================
# Per-chunk worker
# =========================
def process_chunk(file_chunk: list, target_timeframe: str) -> dict:
    """
    Process a list of raw .lz4 files and return a dict mapping
    (coin_str, timeframe) -> list of OHLCV DataFrames.
    This function keeps raw coin strings; normalization occurs later.
    """
    timeframe_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '1d': '1D'}
    tf_code = timeframe_map.get(target_timeframe)
    if not tf_code:
        tprint(f"[ERROR] invalid timeframe '{target_timeframe}'")
        return {}

    chunk_aggregated_results: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for file_path in file_chunk:
        # Determine file layout variant by path
        if 'node_trades' in file_path:
            format_type = 1
        elif 'node_fills_by_block' in file_path:
            format_type = 3
        elif 'node_fills' in file_path:
            format_type = 2
        else:
            continue

        file_fills_clean = []
        for line in iter_lz4_lines(file_path):
            try:
                data = json.loads(line)
                if format_type == 1:
                    file_fills_clean.append(data)
                elif format_type == 2 and isinstance(data, list) and len(data) == 2:
                    file_fills_clean.append(data[1])
                elif format_type == 3 and isinstance(data, dict) and isinstance(data.get('events'), list):
                    for event in data['events']:
                        if isinstance(event, list) and len(event) == 2:
                            file_fills_clean.append(event[1])
            except (json.JSONDecodeError, TypeError):
                continue

        if not file_fills_clean:
            continue

        df_file = pd.DataFrame(file_fills_clean)

        # Normalize column names: 'symbol' -> 'coin'
        if 'symbol' in df_file.columns and 'coin' not in df_file.columns:
            df_file.rename(columns={'symbol': 'coin'}, inplace=True)

        if 'coin' not in df_file.columns:
            continue

        df_file['coin'] = df_file['coin'].astype('category')

        required_cols = {'coin', 'px', 'sz', 'time'}
        if not required_cols.issubset(df_file.columns):
            continue

        # Standardize column names for downstream processing
        df_file.rename(columns={'px': 'price', 'sz': 'qty'}, inplace=True)
        df_file = df_file[['coin', 'price', 'qty', 'time']]

        # Ensure numeric dtypes
        df_file['price'] = pd.to_numeric(df_file['price'], errors='coerce').astype('float64')
        df_file['qty'] = pd.to_numeric(df_file['qty'], errors='coerce').astype('float32')

        # Parse timestamps (support ms numeric and string forms)
        if pd.api.types.is_numeric_dtype(df_file['time']):
            df_file['time'] = pd.to_datetime(df_file['time'], unit='ms', errors='coerce')
        else:
            df_file['time'] = pd.to_datetime(df_file['time'], errors='coerce')

        df_file.dropna(subset=['time', 'coin', 'price', 'qty'], inplace=True)
        if df_file.empty:
            continue

        df_file.set_index('time', inplace=True)

        # Aggregate by coin inside this file
        for coin, df_coin in df_file.groupby('coin', observed=True):
            if df_coin.empty:
                continue

            # IQR outlier handling: either drop extreme rows or clip prices
            if len(df_coin) > OUTLIER_MIN_COUNT:
                Q1 = df_coin['price'].quantile(0.25)
                Q3 = df_coin['price'].quantile(0.75)
                IQR_value = Q3 - Q1
                lower = Q1 - 1.5 * IQR_value
                upper = Q3 + 1.5 * IQR_value
                if OUTLIER_MODE == "drop":
                    df_to_agg = df_coin[(df_coin['price'] >= lower) & (df_coin['price'] <= upper)]
                    if df_to_agg.empty:
                        df_to_agg = df_coin
                else:  # "cap"
                    df_to_agg = df_coin.copy()
                    df_to_agg['price'] = df_to_agg['price'].clip(lower=lower, upper=upper)
            else:
                df_to_agg = df_coin

            # Ensure time index sorted before resample
            df_to_agg = df_to_agg.sort_index()

            # Compute OHLCV per target timeframe
            agg_rules = {'price': ['first', 'max', 'min', 'last'], 'qty': 'sum'}
            ohlcv = df_to_agg.resample(tf_code).agg(agg_rules)
            ohlcv.dropna(how='all', inplace=True)
            if ohlcv.empty:
                continue

            # Flatten multiindex columns and rename to standard names
            ohlcv.columns = ['_'.join(col).strip() for col in ohlcv.columns.values]
            ohlcv.rename(columns={
                'price_first': 'open', 'price_max': 'high',
                'price_min': 'low', 'price_last': 'close',
                'qty_sum': 'volume'
            }, inplace=True)

            key = (str(coin), target_timeframe)
            chunk_aggregated_results.setdefault(key, []).append(ohlcv)

    # Merge per-key lists inside this worker chunk
    final_chunk_results = {}
    for key, df_list in chunk_aggregated_results.items():
        final_chunk_results[key] = pd.concat(df_list)

    return final_chunk_results

# =========================
# Helpers for name normalization & saving
# =========================
_re_at_index = re.compile(r"^@(\d+)$")

def classify_and_normalize_coin(coin_str: str,
                                spot_name_to_index: dict,
                                spot_index_to_display: dict) -> tuple[str, str, str]:
    """
    Classify a raw coin string and return (market, norm_key, display_name):
      - market: "spot" | "perp" | "unknown"
      - norm_key:
          spot -> index string; perp -> perp name; unknown -> original string
      - display_name: preferred human name for saved file
    """
    if coin_str is None:
        return "unknown", "", ""

    s = str(coin_str)

    # Case: @index form maps to spot by index
    m = _re_at_index.match(s)
    if m:
        idx = int(m.group(1))
        disp = spot_index_to_display.get(idx, f"@{idx}")
        return "spot", str(idx), disp

    # Case: "BASE/QUOTE" style — try to resolve to a spot pair by name
    if "/" in s:
        idx = spot_name_to_index.get(s)
        if idx is not None:
            disp = spot_index_to_display.get(idx, s)
            return "spot", str(idx), disp
        # Unrecognized slash form — mark as unknown but keep name
        return "unknown", s, s

    # Otherwise treat as perp name
    return "perp", s, s

def sanitize_filename(name):
    """
    Make a safe filename by replacing filesystem-special characters.
    """
    return (str(name)
            .replace('/', '_')
            .replace('\\', '_')
            .replace(':', '_')
            .replace('*', '_')
            .replace('?', '_')
            .replace('"', '_')
            .replace('<', '_')
            .replace('>', '_')
            .replace('|', '_'))

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Parse Hyperliquid raw fills into OHLCV with proper Spot/Perp normalization.")
    parser.add_argument('timeframe', type=str, choices=['1m', '5m', '15m', '1h', '1d'], help="target timeframe (e.g., 1h)")
    parser.add_argument('workers', type=int, nargs='?', default=None, help="number of worker processes (default: half cores)")
    args = parser.parse_args()

    LOCAL_DATA_DIR = './hyperliquid_data/raw_fills'
    OUTPUT_BASE = f'./hyperliquid_data/processed_ohlcv_{args.timeframe}'
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    SPOT_DIR, PERP_DIR = make_output_dirs(OUTPUT_BASE)

    tprint(f"目标时间周期: {args.timeframe}")
    tprint(f"输出目录: {OUTPUT_BASE} (spot/perp)")

    # Load API meta for name normalization; optional but recommended
    meta, spot_meta = load_meta_and_spotmeta()
    spot_index_to_universe, spot_name_to_index, spot_index_to_display = build_spot_maps(spot_meta)
    perp_id_to_name, perp_name_to_id, perp_delisted = build_perp_maps(meta)

    # File discovery range (adjust START_DATE / END_DATE as needed)
    START_DATE = date(2025, 3, 22)
    END_DATE = date(2025, 8, 22)

    all_files = []
    data_parent_folders = ['node_trades/hourly', 'node_fills/hourly', 'node_fills_by_block/hourly']
    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime('%Y%m%d')
        for parent in data_parent_folders:
            path_pattern = os.path.join(LOCAL_DATA_DIR, parent, date_str, '*.lz4')
            all_files.extend(glob(path_pattern))
        current_date += timedelta(days=1)

    print(f"发现 {len(all_files)} 个文件待处理。")
    if not all_files:
        return

    # Decide worker count (default: half of available cores)
    num_processes = os.cpu_count()
    num_workers = args.workers if args.workers is not None else max(1, num_processes // 2)
    tprint(f"系统有 {num_processes} 个核心. 使用 {num_workers} 个工作进程。")

    chunk_size = len(all_files) // num_workers + (1 if len(all_files) % num_workers > 0 else 0)
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    # Stage 1: parallel chunk processing (collect per-file OHLCV blocks keyed by raw coin string)
    all_results_raw: dict[tuple[str, str], list[pd.DataFrame]] = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_chunk, chunk, args.timeframe): i for i, chunk in enumerate(file_chunks)}
        for future in tqdm(as_completed(futures), total=len(file_chunks), desc=f"阶段 1: 并行处理 {args.timeframe} 数据"):
            try:
                chunk_result = future.result()
            except Exception as e:
                tprint(f"[ERROR] worker failed: {e}")
                continue
            for key, ohlcv_df in chunk_result.items():
                all_results_raw.setdefault(key, []).append(ohlcv_df)

    # Stage 2: classify raw coin names into spot or perp and group accordingly
    print("\n--- 阶段 2: 统一资产（Spot/Perp 分流 + 名字规范化） ---")
    spot_buckets: dict[tuple[str, str], list[pd.DataFrame]] = {}  # key=(spot_index_str, tf)
    spot_display_name: dict[str, str] = {}  # spot_index_str -> human display
    perp_buckets: dict[tuple[str, str], list[pd.DataFrame]] = {}  # key=(perp_name, tf)

    for key, df_list in all_results_raw.items():
        coin_str, tf_name = key
        market, norm_key, disp_name = classify_and_normalize_coin(
            coin_str,
            spot_name_to_index=spot_name_to_index,
            spot_index_to_display=spot_index_to_display
        )
        if market == "spot":
            spot_buckets.setdefault((norm_key, tf_name), []).extend(df_list)
            if norm_key not in spot_display_name:
                spot_display_name[norm_key] = disp_name
        elif market == "perp":
            perp_buckets.setdefault((norm_key, tf_name), []).extend(df_list)
        else:
            # For unknown names, keep them as perp-like to avoid accidental exclusion
            perp_buckets.setdefault((coin_str, tf_name), []).extend(df_list)

    # Free memory from stage 1 container
    del all_results_raw

    # Stage 3: aggregate per-asset groups in batches and save to parquet
    print("\n--- 阶段 3: 分块聚合并保存 ---")
    timeframe_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '1d': '1D'}
    tf_code = timeframe_map[args.timeframe]

    def finalize_and_save(df_list: list[pd.DataFrame], save_path: str):
        """
        Batch-concat intermediate blocks to keep memory use low, resample to target tf,
        forward-fill prices and set missing volume to zero, then save as parquet.
        """
        if not df_list:
            return
        partials = []
        # Process in batches to avoid one large concat
        for i in range(0, len(df_list), CHUNK_AGG_BATCH):
            chunk = df_list[i:i + CHUNK_AGG_BATCH]
            tmp = pd.concat(chunk)
            tmp = tmp.sort_index()
            part = tmp.resample(tf_code).agg(
                open=('open', 'first'), high=('high', 'max'),
                low=('low', 'min'),   close=('close', 'last'),
                volume=('volume', 'sum')
            ).dropna(how='all')
            partials.append(part)

        combined_df = pd.concat(partials).sort_index()
        # Final aggregation to align batch boundaries
        final_ohlcv = combined_df.resample(tf_code).agg(
            open=('open', 'first'), high=('high', 'max'),
            low=('low', 'min'),   close=('close', 'last'),
            volume=('volume', 'sum')
        ).dropna(how='all')

        # Forward-fill price columns and fill missing volume with 0
        price_cols = ['open', 'high', 'low', 'close']
        final_ohlcv[price_cols] = final_ohlcv[price_cols].ffill()
        final_ohlcv['volume'] = final_ohlcv['volume'].fillna(0)
        final_ohlcv.dropna(inplace=True)

        if final_ohlcv.empty:
            return

        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            final_ohlcv.sort_index().to_parquet(save_path)
        except OSError as e:
            tprint(f"[ERROR] Failed to save {save_path}: {e}")

    # Save Spot parquet files (use display name as filename)
    for (spot_index_str, tf_name), df_list in tqdm(spot_buckets.items(), desc="Saving Spot Parquet"):
        idx = int(spot_index_str)
        display = spot_display_name.get(spot_index_str, f"@{idx}")
        safe_name = sanitize_filename(display)
        save_path = os.path.join(SPOT_DIR, f"{safe_name}_{tf_name}_ohlcv.parquet")
        finalize_and_save(df_list, save_path)

    # Save Perp parquet files (do not filter delisted by default to avoid survivorship bias)
    for (perp_name, tf_name), df_list in tqdm(perp_buckets.items(), desc="Saving Perp Parquet"):
        # If you want to exclude currently delisted perpetuals, uncomment below lines:
        # if perp_delisted.get(perp_name, False):
        #     continue
        safe_name = sanitize_filename(perp_name)
        save_path = os.path.join(PERP_DIR, f"{safe_name}_{tf_name}_ohlcv.parquet")
        finalize_and_save(df_list, save_path)

    print(f"\nProcessing for timeframe '{args.timeframe}' complete! 🎉")

if __name__ == '__main__':
    main()

# -----------------------
# Usage examples (end of file)
# -----------------------
# Basic run for 1-hour OHLCV using default worker count (half of cores):
#   python process_data.py 1h
#
# Run with explicit number of worker processes (e.g., 4):
#   python process_data.py 1h 4
#
# Change target timeframe to 15 minutes and use 2 workers:
#   python process_data.py 15m 2
#
# Notes:
# - Adjust START_DATE and END_DATE inside main() to select the date range to scan.
# - Input files are expected under ./hyperliquid_data/raw_fills with folders like:
#     node_trades/hourly/YYYYMMDD/*.lz4
#     node_fills/hourly/YYYYMMDD/*.lz4
#     node_fills_by_block/hourly/YYYYMMDD/*.lz4
# - Output files are written to ./hyperliquid_data/processed_ohlcv_<tf>/spot and .../perp
# - To change outlier handling, set OUTLIER_MODE = "cap" or "drop" at top of file.
# - For name normalization, the script will try to fetch /info meta; if that fails it still runs.
#
# End of usage examples.
