# filename: create_target_list.py
#
# Goal:
# Build a clean target symbol list for downstream downloads by taking the
# intersection of:
#   - Hyperliquid active perp assets (from /info type=allMids), normalized to
#     human-facing tickers (e.g., UBTC -> BTC, kBONK -> BONK) using /info meta
#     when available, with safe fallbacks.
#   - Binance Spot active /USDT markets' base assets (via CCXT).
#
# Output:
#   target_crypto_list.txt  (one symbol per line, e.g., BTC, ETH, SOL)

import requests
import ccxt
from typing import Set, Dict, Optional

INFO_URL = "https://api.hyperliquid.xyz/info"


def get_hyperliquid_symbols() -> Set[str]:
    """Fetch Hyperliquid active perp asset names (raw chain-level keys)."""
    payload = {"type": "allMids"}
    print("1) Fetching Hyperliquid active perp assets ...")
    try:
        resp = requests.post(INFO_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        symbols = set(data.keys())
        print(f"   -> OK: {len(symbols)} Hyperliquid assets.")
        return symbols
    except Exception as e:
        print(f"   -> ERROR: failed to fetch Hyperliquid list: {e}")
        return set()


def get_hyperliquid_meta_alias_map() -> Dict[str, str]:
    """
    Try to build 'raw_symbol -> display/canonical symbol' map from /info meta.
    This is best-effort: if the expected fields are not present, it returns {}.
    """
    payload = {"type": "meta"}
    print("   Fetching Hyperliquid meta for alias mapping ...")
    alias_map: Dict[str, str] = {}
    try:
        resp = requests.post(INFO_URL, json=payload, timeout=20)
        resp.raise_for_status()
        meta = resp.json()

        # Common containers to probe; structure may evolve over time.
        candidates = []
        if isinstance(meta, dict):
            if "assetCtxs" in meta:
                candidates.append(meta["assetCtxs"])
            if "assets" in meta:
                candidates.append(meta["assets"])
            if "perpMeta" in meta and isinstance(meta["perpMeta"], dict):
                # sometimes nested containers live here
                for v in meta["perpMeta"].values():
                    candidates.append(v)
            if "spotMeta" in meta and isinstance(meta["spotMeta"], dict):
                candidates.append(meta["spotMeta"])

        def extract_pair(raw_key: str, obj: dict) -> Optional[tuple]:
            # Heuristics: prefer raw-like keys from these fields,
            # and display-like values from these fields.
            raw_fields = ["name", "symbol", "ticker", "asset", "perpSymbol"]
            disp_fields = ["displaySymbol", "displayName", "uiSymbol", "underlying", "base"]
            raw_val = None
            disp_val = None
            # guess "raw"
            for k in raw_fields:
                if isinstance(obj, dict) and k in obj and isinstance(obj[k], str):
                    raw_val = obj[k].strip().upper()
                    break
            # guess "display"
            for k in disp_fields:
                if isinstance(obj, dict) and k in obj and isinstance(obj[k], str):
                    disp_val = obj[k].strip().upper()
                    break
            if raw_val and disp_val and raw_val != disp_val:
                return raw_val, disp_val
            return None

        # Case A: assetCtxs is a dict mapping raw->ctx
        for block in candidates:
            if isinstance(block, dict):
                # direct mapping form: { "UBTC": {...}, ... }
                for raw, ctx in block.items():
                    if isinstance(raw, str) and isinstance(ctx, dict):
                        raw_u = raw.strip().upper()
                        pair = extract_pair(raw_u, ctx)
                        if pair:
                            alias_map[pair[0]] = pair[1]
                # nested dicts
                for ctx in block.values():
                    if isinstance(ctx, dict):
                        pair = extract_pair("", ctx)
                        if pair:
                            alias_map[pair[0]] = pair[1]

            elif isinstance(block, list):
                # list of ctx objects: [{...}, ...]
                for ctx in block:
                    if isinstance(ctx, dict):
                        pair = extract_pair("", ctx)
                        if pair:
                            alias_map[pair[0]] = pair[1]

        # Deduplicate to itself (last write wins); print a short summary
        if alias_map:
            print(f"   -> OK: built {len(alias_map)} alias entries from meta.")
        else:
            print("   -> No explicit alias entries found in meta (will use fallbacks).")
        return alias_map

    except Exception as e:
        print(f"   -> WARN: failed to fetch/parse meta: {e} (will use fallbacks).")
        return {}


def get_binance_usdt_spot_symbols() -> Set[str]:
    """Fetch Binance Spot active /USDT markets and return their base assets."""
    print("2) Fetching Binance Spot /USDT market bases via CCXT ...")
    try:
        exchange = ccxt.binance({"options": {"defaultType": "spot"}, "enableRateLimit": True})
        exchange.load_markets()
        bases = {s.split("/")[0] for s in exchange.markets if s.endswith("/USDT")}
        print(f"   -> OK: {len(bases)} Binance /USDT base assets.")
        return bases
    except Exception as e:
        print(f"   -> ERROR: failed to fetch Binance list: {e}")
        return set()


def normalize_hl_symbol(raw: str, binance_bases: Set[str], alias_map: Dict[str, str]) -> str:
    """
    Normalize a Hyperliquid raw symbol to a canonical ticker suitable for
    intersection with Binance bases.
    Order of preference:
      1) explicit alias from meta (if any and in Binance set),
      2) strip known prefixes ('U', 'k') if the stripped form exists in Binance,
      3) keep original raw.
    """
    u = raw.strip().upper()

    # 1) explicit alias
    if u in alias_map:
        cand = alias_map[u].upper()
        if cand in binance_bases:
            return cand

    # 2) safe prefix strip (only accept if it truly exists on Binance)
    for prefix in ("U", "K"):
        if u.startswith(prefix) and len(u) > 1:
            cand = u[1:]
            if cand in binance_bases:
                return cand

    # 3) keep raw (may still intersect if identical)
    return u


if __name__ == "__main__":
    # Step A: fetch both sets
    hl_raw = get_hyperliquid_symbols()
    binance_bases = get_binance_usdt_spot_symbols()

    print("-" * 60)

    if hl_raw and binance_bases:
        # Step B: fetch meta alias map and normalize HL symbols
        alias_map = get_hyperliquid_meta_alias_map()
        hl_norm = {normalize_hl_symbol(s, binance_bases, alias_map) for s in hl_raw}

        # Step C: build intersection
        common_symbols = sorted(list(hl_norm.intersection(binance_bases)))

        # Step D: save to file
        output_filename = "target_crypto_list.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            for symbol in common_symbols:
                f.write(symbol + "\n")

        print("✅ Done.")
        print(f"Intersection size: {len(common_symbols)} assets.")
        print(f"Saved list to: {output_filename}")
    else:
        print("❌ Could not fetch one or both lists; cannot build intersection.")
