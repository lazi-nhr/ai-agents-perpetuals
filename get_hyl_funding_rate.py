"""
Download Hyperliquid funding rate history for many perps,
with global rate limiting, robust retries, resume, and
configurable handling for 500 errors — now with *canonical name mapping*
from /info meta so coin names match the exchange exactly (case-sensitive).

Inputs:
  --perp-list: text file, one coin per line (e.g., BTC, ETH, kBONK)

Outputs (per coin):
  <out-dir>/<COIN>_funding_<YYYY-MM-DD>_<YYYY-MM-DD>.parquet
  and a temporary checkpoint:
  <out-dir>/tmp/<COIN>.partial.parquet  (removed when the coin finishes)

API: POST https://api.hyperliquid.xyz/info
Bodies:
  {"type":"fundingHistory","coin":"ETH","startTime":<ms>,"endTime":<ms>}
  {"type":"meta"}  # to fetch canonical perp names
"""

import os
import time
import random
import argparse
import datetime as dt
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests import HTTPError
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

API_URL = "https://api.hyperliquid.xyz/info"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "hl-funding-downloader/1.4"
}

# ===================== time helpers =====================

def to_ms(s: str) -> int:
    d = dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)

def ms_to_iso(ms: int) -> str:
    try:
        return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ms)

def windows(start_ms: int, end_ms: int, hours_per_window: int) -> List[Tuple[int, int]]:
    step = hours_per_window * 3600 * 1000
    out = []
    cur = start_ms
    while cur < end_ms:
        nxt = min(cur + step, end_ms)
        out.append((cur, nxt))
        cur = nxt
    return out

def next_hour_ms(ms: int) -> int:
    return ms + 3600 * 1000  # funding is hourly

# ===================== global rate limiter =====================

class TokenBucket:
    """Capacity=1; refill at rps tokens/sec. acquire() blocks until a token is available."""
    def __init__(self, rps: float):
        self.rps = max(0.1, rps)
        self._last = time.monotonic()
        self._tokens = 1.0

    def acquire(self):
        while True:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(1.0, self._tokens + elapsed * self.rps)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            need = (1.0 - self._tokens) / self.rps
            time.sleep(max(0.01, need))

# ===================== HTTP helpers & retries =====================

def post_info(session: requests.Session, payload: Dict[str, Any],
              timeout=(10.0, 30.0), limiter: Optional[TokenBucket] = None,
              expect_list: bool = True,
              verbose: bool = False) -> Any:
    """
    POST with limiter + retries.
    - On 429/503: respect Retry-After if present; else exponential backoff (with jitter).
    - On network/5xx: exponential backoff (with jitter).
    - If expect_list=True, assert JSON is a list (fundingHistory).
      Otherwise allow dict (meta).
    """
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        if limiter:
            limiter.acquire()
        try:
            r = session.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
            status = r.status_code

            if status in (429, 503):
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_s = float(retry_after)
                    except Exception:
                        wait_s = None
                else:
                    wait_s = None
                if wait_s is None:
                    base = [1, 2, 3, 5, 8, 13, 21, 34][min(attempt - 1, 7)]
                    wait_s = base + random.uniform(0, 0.5 * base)
                if verbose:
                    print(f"[{status}] retry after {wait_s:.2f}s payload={payload}")
                time.sleep(wait_s)
                continue

            r.raise_for_status()
            data = r.json()
            if expect_list and not isinstance(data, list):
                raise ValueError(f"Unexpected JSON type: {type(data)} (expected list)")
            if not expect_list and not isinstance(data, dict):
                raise ValueError(f"Unexpected JSON type: {type(data)} (expected dict)")
            return data

        except Exception as e:
            if attempt >= max_attempts:
                if verbose:
                    print(f"[ERROR] giving up after {attempt} attempts; payload={payload}; err={e}")
                raise
            base = min(60.0, 0.5 * (2 ** (attempt - 1)))  # 0.5,1,2,4,8,16,32,60 cap
            wait_s = base + random.uniform(0, 0.3 * base)
            if verbose:
                print(f"[WARN] attempt {attempt} failed; backoff {wait_s:.2f}s; payload={payload}; err={e}")
            time.sleep(wait_s)

# ===================== meta → canonical name mapping =====================

def fetch_perp_canonical_map(limiter: Optional[TokenBucket] = None,
                             verbose: bool = False) -> Dict[str, str]:
    """
    Returns a mapping: lower_name -> canonical 'name' from meta.universe for perps.
    Example: {'kbonk': 'kBONK', 'kdogs': 'kDOGS', 'btc': 'BTC', ...}
    """
    with requests.Session() as sess:
        meta = post_info(sess, {"type": "meta"}, timeout=(10.0, 30.0),
                         limiter=limiter, expect_list=False, verbose=verbose)
    uni = meta.get("universe") or []
    mapping: Dict[str, str] = {}
    for entry in uni:
        try:
            nm = entry.get("name")
            if not nm:
                continue
            mapping[nm.lower()] = nm
        except Exception:
            continue
    if verbose:
        print(f"[meta] loaded {len(mapping)} perp names")
    return mapping

def canonicalize_list(user_coins: List[str],
                      lower_to_canonical: Dict[str, str]) -> List[str]:
    """
    For each user coin (do NOT uppercase), map case-insensitively to the canonical name
    if we can find it in meta. Unknown names are kept as-is.
    Duplicates after canonicalization are de-duplicated preserving order.
    """
    seen = set()
    out: List[str] = []
    for c in user_coins:
        raw = c.strip()
        if not raw:
            continue
        can = lower_to_canonical.get(raw.lower(), raw)
        if can not in seen:
            seen.add(can)
            out.append(can)
    return out

# ===================== 500 handling (split/empty/skip-coin) =====================

class SkipCoin(Exception):
    pass

def fetch_with_policy(sess: requests.Session,
                      coin: str,
                      ws: int, we: int,
                      start_ms: int, end_ms: int,
                      limiter: TokenBucket,
                      connect_timeout: float,
                      read_timeout: float,
                      min_hours_per_window: int,
                      on500: str,
                      verbose: bool) -> List[Dict[str, Any]]:
    """
    Fetch one window using the selected 500 policy:
      - 'split'     : on 500, recursively split until min_hours_per_window, then give up tiny window
      - 'empty'     : on 500, treat as empty window (return [])
      - 'skip-coin' : on 500, raise SkipCoin to abort the whole coin
    """
    span_ms = we - ws
    try:
        payload = {"type": "fundingHistory", "coin": coin, "startTime": ws, "endTime": we}
        data = post_info(sess, payload, timeout=(connect_timeout, read_timeout),
                         limiter=limiter, expect_list=True, verbose=verbose)
        out: List[Dict[str, Any]] = []
        for it in data:
            try:
                t = int(it.get("time"))
                fr = float(it.get("fundingRate"))
                prem = float(it.get("premium"))
            except Exception:
                continue
            if t < start_ms or t >= end_ms:
                continue
            out.append({"time": t, "coin": coin, "fundingRate": fr, "premium": prem})
        return out

    except HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 500:
            if on500 == "empty":
                print(f"[WARN] coin={coin} treat 500 as empty window {ms_to_iso(ws)} -> {ms_to_iso(we)}")
                return []
            if on500 == "skip-coin":
                raise SkipCoin(f"coin={coin} hit 500 for window {ms_to_iso(ws)} -> {ms_to_iso(we)}")
            # on500 == "split"
            min_ms = max(1, min_hours_per_window) * 3600 * 1000
            if span_ms <= min_ms:
                print(f"[WARN] coin={coin} give up tiny window {ms_to_iso(ws)} -> {ms_to_iso(we)}: 500")
                return []
            mid = ws + span_ms // 2
            left = fetch_with_policy(sess, coin, ws, mid, start_ms, end_ms,
                                     limiter, connect_timeout, read_timeout,
                                     min_hours_per_window, on500, verbose)
            right = fetch_with_policy(sess, coin, mid, we, start_ms, end_ms,
                                      limiter, connect_timeout, read_timeout,
                                      min_hours_per_window, on500, verbose)
            return left + right
        # other HTTP errors: re-raise
        raise
    except Exception:
        raise

# ===================== partial IO (resume) =====================

def partial_path(out_dir: str, coin: str) -> str:
    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, f"{sanitize_filename(coin)}.partial.parquet")

def final_path(out_dir: str, coin: str, start: str, end: str) -> str:
    return os.path.join(out_dir, f"{sanitize_filename(coin)}_funding_{start}_{end}.parquet")

def read_partial(out_dir: str, coin: str) -> Optional[pd.DataFrame]:
    p = partial_path(out_dir, coin)
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            if isinstance(df.index, pd.DatetimeIndex):
                return df
        except Exception:
            pass
    return None

def save_parquet_atomic(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.sort_index().to_parquet(tmp)
    os.replace(tmp, path)

def sanitize_filename(name: str) -> str:
    return (name.replace("/", "_").replace("\\", "_").replace(":", "_")
                .replace("*", "_").replace("?", "_").replace('"', "_")
                .replace("<", "_").replace(">", "_").replace("|", "_"))

# ===================== per-coin fetch (resume + 500 policy) =====================

def fetch_coin_incremental(coin: str,
                           start_ms: int, end_ms: int,
                           hours_per_window: int,
                           out_dir: str,
                           limiter: TokenBucket,
                           connect_timeout: float = 10.0,
                           read_timeout: float = 30.0,
                           resume: bool = True,
                           min_hours_per_window: int = 24,
                           on500: str = "split",
                           verbose: bool = False) -> pd.DataFrame:
    """
    Fetch funding history for `coin` with resume support and configurable 500 handling.
    """
    final_fp = final_path(out_dir, coin, START_STR, END_STR)
    if os.path.exists(final_fp) and not FORCE:
        return pd.read_parquet(final_fp)

    # load partial if present
    existing = read_partial(out_dir, coin) if resume else None
    fetch_from_ms = start_ms
    if existing is not None and not existing.empty:
        last_ms = int(pd.Timestamp(existing.index.max(), tz="UTC").timestamp() * 1000)
        fetch_from_ms = max(fetch_from_ms, next_hour_ms(last_ms))

    if fetch_from_ms >= end_ms:
        # nothing to do; finalize from existing
        if existing is None or existing.empty:
            idx = pd.DatetimeIndex([], name="time")
            df = pd.DataFrame(index=idx, columns=["coin", "fundingRate", "premium"]).astype({
                "coin": "category", "fundingRate": "float64", "premium": "float64"
            })
        else:
            df = existing
        save_parquet_atomic(df, final_fp)
        p = partial_path(out_dir, coin)
        if os.path.exists(p):
            os.remove(p)
        return df

    # windows to fetch
    wnds = windows(fetch_from_ms, end_ms, hours_per_window)
    agg_rows: List[Dict[str, Any]] = []

    with requests.Session() as sess:
        try:
            for (ws, we) in wnds:
                rows = fetch_with_policy(sess, coin, ws, we,
                                         start_ms, end_ms, limiter,
                                         connect_timeout, read_timeout,
                                         min_hours_per_window, on500, verbose)
                agg_rows.extend(rows)

                # small pacing even with limiter
                time.sleep(0.01)

                # periodic partial flush
                if len(agg_rows) >= 20 * 24:  # ~20 days of hourly points at most
                    partial_flush(out_dir, coin, existing, agg_rows)
                    existing = read_partial(out_dir, coin)
                    agg_rows.clear()
        except SkipCoin as e:
            print(f"[WARN] {e} -> skip the rest of this coin.")

    # final merge and write
    df = partial_flush(out_dir, coin, existing, agg_rows)
    save_parquet_atomic(df, final_fp)
    p = partial_path(out_dir, coin)
    if os.path.exists(p):
        os.remove(p)
    return df

def partial_flush(out_dir: str, coin: str,
                  existing: Optional[pd.DataFrame],
                  new_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge (existing partial) + (new_rows) and write back to partial parquet.
    Return the merged DataFrame.
    """
    base = existing.copy() if existing is not None else None

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_new = df_new.drop_duplicates(subset=["time"]).sort_values("time")
        idx = pd.to_datetime(df_new["time"].values, unit="ms", utc=True).tz_localize(None)
        df_new = df_new.drop(columns=["time"])
        df_new.index = idx
        df_new.index.name = "time"
        df_new["coin"] = df_new["coin"].astype("category")
        df_new["fundingRate"] = df_new["fundingRate"].astype("float64")
        df_new["premium"] = df_new["premium"].astype("float64")
        if base is None or base.empty:
            merged = df_new
        else:
            merged = pd.concat([base, df_new], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = base if base is not None else pd.DataFrame(
            index=pd.DatetimeIndex([], name="time"),
            columns=["coin", "fundingRate", "premium"]
        ).astype({"coin": "category", "fundingRate": "float64", "premium": "float64"})

    save_parquet_atomic(merged, partial_path(out_dir, coin))
    return merged

# ===================== IO helpers =====================

def read_coin_list(path: str) -> List[str]:
    # IMPORTANT: do NOT uppercase; keep user-provided form and let canonicalize() fix case.
    with open(path, "r", encoding="utf-8") as f:
        coins = [line.strip() for line in f if line.strip()]
    return coins

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Download funding history (resume + rate limit + 500 policy + canonical names).")
    ap.add_argument("--perp-list", default="./perp_list.txt", help="Path to coin list (one coin per line).")
    ap.add_argument("--start", default="2025-03-22", help="Start date (YYYY-MM-DD), inclusive.")
    ap.add_argument("--end", default="2025-08-22", help="End date (YYYY-MM-DD), exclusive.")
    ap.add_argument("--out-dir", default="./hyperliquid_funding", help="Output folder.")
    ap.add_argument("--workers", type=int, default=2, help="Concurrent worker threads.")
    ap.add_argument("--rps", type=float, default=0.5, help="Global requests per second (token bucket).")
    ap.add_argument("--hours-per-window", type=int, default=14*24, help="Hours per fundingHistory request.")
    ap.add_argument("--min-hours-per-window", type=int, default=24, help="Minimum hours for recursive split.")
    ap.add_argument("--on-500", choices=["split", "empty", "skip-coin"], default="empty",
                    help="Behavior when a window returns 500. Default: empty (treat as no data).")
    ap.add_argument("--resume", action="store_true", help="Resume from partial parquet if present.")
    ap.add_argument("--force", action="store_true", help="Force overwrite final files if they exist.")
    ap.add_argument("--verbose", action="store_true", help="Verbose HTTP/backoff logs.")
    args = ap.parse_args()

    global START_STR, END_STR, FORCE
    START_STR, END_STR = args.start, args.end
    FORCE = args.force

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tmp"), exist_ok=True)

    # 1) read list as-is; 2) canonicalize via /info meta (case-insensitive to canonical)
    user_coins = read_coin_list(args.perp_list)
    # Get canonical perp names from meta once
    meta_map = fetch_perp_canonical_map(limiter=None, verbose=args.verbose)  # not rate-limited; 1 call
    coins = canonicalize_list(user_coins, meta_map)

    # Log mapping info
    unknown = [c for c in user_coins if c.lower() not in meta_map]
    if unknown:
        print(f"[meta] {len(unknown)} names not found in current meta; we'll try them as-is (could be delisted historically):")
        print("       " + ", ".join(sorted(set(unknown))[:20]) + (" ..." if len(set(unknown)) > 20 else ""))
    print(f"Coins (after canonicalize & dedupe): {len(coins)} | Window: {args.start} -> {args.end} | hours/window: {args.hours_per_window}")

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)
    limiter = TokenBucket(args.rps)

    print(f"Global rate limit: {args.rps:.2f} req/s | Workers: {args.workers} | Resume: {args.resume} | Force: {args.force}")

    def run_one(coin: str) -> Tuple[str, Optional[str]]:
        try:
            _ = fetch_coin_incremental(
                coin=coin,
                start_ms=start_ms, end_ms=end_ms,
                hours_per_window=args.hours_per_window,
                out_dir=args.out_dir,
                limiter=limiter,
                resume=args.resume,
                min_hours_per_window=args.min_hours_per_window,
                on500=args.on_500,
                verbose=args.verbose
            )
            return coin, final_path(args.out_dir, coin, START_STR, END_STR)
        except Exception as e:
            return coin, f"ERROR: {e}"

    bar = tqdm(total=len(coins), desc="Fetching", ncols=80) if tqdm else None
    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, c) for c in coins]
        for fut in as_completed(futs):
            coin, msg = fut.result()
            results[coin] = msg
            if bar:
                bar.update(1)
        if bar:
            bar.close()

    ok = sum(1 for v in results.values() if v and not v.startswith("ERROR"))
    bad = len(results) - ok
    print(f"Done. Success: {ok}, Failed: {bad}. Output: {args.out_dir}")
    if bad:
        print("Failed coins:")
        for k, v in results.items():
            if v and v.startswith("ERROR"):
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

"""
Usage examples:

Basic run (default options):
python get_hyl_funding_rate.py --perp-list ./perp_list.txt --start 2025-03-22 --end 2025-08-22 --out-dir ./hyperliquid_funding

Resume from partial parquet files (do not re-download finished coins):
python get_hyl_funding_rate.py --perp-list ./perp_list.txt --start 2025-03-22 --end 2025-08-22 --out-dir ./hyperliquid_funding --resume

Force overwrite final files even if they exist:
python get_hyl_funding_rate.py --perp-list ./perp_list.txt --start 2025-03-22 --end 2025-08-22 --out-dir ./hyperliquid_funding --force

Change how the script treats HTTP 500 windows:
# treat 500 as empty (default)
python get_hyl_funding_rate.py --on-500 empty ...
# attempt recursive split on 500
python get_hyl_funding_rate.py --on-500 split ...
# skip the whole coin on 500
python get_hyl_funding_rate.py --on-500 skip-coin ...

Adjust concurrency and global rate:
python get_hyl_funding_rate.py --perp-list ./perp_list.txt --workers 4 --rps 1.0 --out-dir ./out_dir

Example with more options and verbose logs:
python get_hyl_funding_rate.py --perp-list ./perp_list.txt --start 2025-01-01 --end 2025-06-01
--out-dir ./out --workers 3 --rps 0.5 --hours-per-window 336 --resume --verbose

Short notes:
- perp_list must be a plain text file, one coin per line (for example: BTC, ETH, kBONK).
- The script will fetch /info meta and map user names case-insensitively to canonical perp names when possible.
- Partial data is written to <out-dir>/tmp/<SANITIZED_COIN>.partial.parquet and is removed when the coin finishes.
- Final output files use the pattern: <out-dir>/<COIN>funding<YYYY-MM-DD>_<YYYY-MM-DD>.parquet
- --start is inclusive, --end is exclusive. Use the YYYY-MM-DD format.
- If a coin is not found in current meta, the script will try the name as given (this may be needed for delisted or historical coins).
"""