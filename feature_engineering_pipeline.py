"""
Feature Engineering Pipeline for Cryptocurrency Pairs

This module provides a complete pipeline for:
1. Loading cointegration results
2. Normalizing spreads (selection vs trading windows)
3. Computing technical features (MACD, RSI, Kalman, EWMA volatility, etc.)
4. Merging funding rate data
5. Exporting final feature matrix

Usage:
    from feature_engineering_pipeline import run_full_pipeline
    
    features_df = run_full_pipeline(
        cointegration_folder="path/to/cointegration_csvs",
        input_csv="bin_futures_historical_pairs_with_spreads.csv",
        funding_data_folder="path/to/funding_data",
        output_file="final_features.csv"
    )
"""

import os
import glob
import pandas as pd
import numpy as np
import ta
from pathlib import Path


# ============================================================================
# 1. LOADING COINTEGRATION RESULTS
# ============================================================================

def load_cointegration_csvs(folder_path):
    """
    Load all CSV files with filenames like 'SYM1_SYM2_bin_futures_window_cointegration.csv'
    into a dictionary {(sym1, sym2): DataFrame}
    
    Args:
        folder_path (str): Path to folder containing cointegration CSV files
    
    Returns:
        dict: {(sym1, sym2): cointegration_results_df}
    """
    csv_files = glob.glob(os.path.join(folder_path, "*_bin_futures_window_cointegration.csv"))
    data_dict = {}

    for f in csv_files:
        base = os.path.basename(f).replace(".csv", "")
        try:
            # Extract symbols
            parts = base.split("_")
            sym1, sym2 = parts[0], parts[1]
        except Exception as e:
            print(f"Skipping file {f}, cannot parse symbols: {e}")
            continue

        # Read CSV
        df = pd.read_csv(f)
        data_dict[(sym1, sym2)] = df

    return data_dict


def select_top_pairs_per_window(cointegration_dict, top_k=5):
    """
    Select top-K cointegrated pairs per rolling window based on correlation.
    
    Args:
        cointegration_dict (dict): {(sym1, sym2): cointegration_results_df}
        top_k (int): Number of top pairs to keep per window (default: 5)
    
    Returns:
        dict: {(start, end): [(sym1, sym2), ...]}
    """
    top_pairs_per_window = {}
    
    # Collect all cointegrated pairs per window
    for pair, df in cointegration_dict.items():
        for _, row in df.iterrows():
            window_key = (pd.to_datetime(row["start"] + " 00:00:00" if isinstance(row["start"], str) and len(str(row["start"])) == 10 else row["start"], format="%Y-%m-%d %H:%M:%S"), 
                         pd.to_datetime(row["end"], format="%Y-%m-%d %H:%M:%S"))
            if window_key not in top_pairs_per_window:
                top_pairs_per_window[window_key] = []
            if row["cointegrated"]:
                top_pairs_per_window[window_key].append((pair, row["correlation"]))
    
    # Keep only top-K by absolute correlation
    for window_key, pairs in top_pairs_per_window.items():
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_pairs_per_window[window_key] = [p[0] for p in pairs[:top_k]]
    
    return top_pairs_per_window


# ============================================================================
# 2. SPREAD NORMALIZATION
# ============================================================================

def normalize_spreads(full_df, top_pairs_per_window, folder_path="", trade_minutes=1440):
    """
    Normalize spreads for trading day using selection window beta/alpha 
    from per-pair CSVs and propagate beta/alpha/correlation to trade period.

    Parameters:
    - full_df: DataFrame with timestamp, price columns for all symbols
    - top_pairs_per_window: dict {(sel_start, sel_end): [('SYM1','SYM2'), ...]}
    - folder_path: folder containing per-pair CSVs named 
                   '{sym1}_{sym2}_bin_futures_window_cointegration.csv'
    - trade_minutes: int, duration of trading window (default 1440 = 1 day)

    Returns:
    - DataFrame with new columns:
        '{pair}_spreadNorm', '{pair}_alpha', '{pair}_beta', '{pair}_corr', '{pair}_pval'
    """
    # Drop old spread/alpha/beta/corr/adf columns
    spread_cols = [col for col in full_df.columns if ("alpha" in col) or ("beta" in col) or 
                   ("corr" in col) or ("adf" in col)]
    df = full_df.drop(columns=spread_cols, axis=1).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"]

    for (sel_start, sel_end), top_pairs in top_pairs_per_window.items():
        # Selection window mask
        sel_mask = (timestamps >= sel_start) & (timestamps <= sel_end)
        if not any(sel_mask):
            print(f"No selection data for window {sel_start} to {sel_end}. Skipping.")
            continue

        # Trading window mask (next day after selection)
        trade_start = sel_end + pd.Timedelta(minutes=1)
        trade_end = trade_start + pd.Timedelta(minutes=trade_minutes - 1)
        trade_mask = (timestamps >= trade_start) & (timestamps <= trade_end)
        if not any(trade_mask):
            print(f"No trading data for window {sel_start} to {sel_end}. Skipping.")
            continue

        for sym1, sym2 in top_pairs:
            csv_file = os.path.join(folder_path, f"{sym1}_{sym2}_bin_futures_window_cointegration.csv")
            if not os.path.exists(csv_file):
                print(f"CSV not found for {sym1}-{sym2}. Skipping.")
                continue

            pair_df = pd.read_csv(csv_file, parse_dates=['start', 'end'])
            sel_row = pair_df[(pair_df['start'] == sel_start) & (pair_df['end'] == sel_end)]

            if sel_row.empty or not sel_row['cointegrated'].values[0]:
                continue

            alpha = sel_row['alpha'].values[0]
            beta = sel_row['beta'].values[0]
            corr = sel_row['correlation'].values[0]
            p_val = sel_row["adf_p"].values[0]

            if pd.isna(alpha) or pd.isna(beta):
                print(f"NaN alpha/beta for {sym1}-{sym2} in window {sel_start} to {sel_end}. Skipping.")
                continue

            # Compute selection window spread for mean/std
            y_sel = df.loc[sel_mask, f"{sym1}_close"]
            x_sel = df.loc[sel_mask, f"{sym2}_close"]
            sel_spread = y_sel - (alpha + beta * x_sel)
            mu = sel_spread.mean()
            sigma = sel_spread.std()
            if sigma == 0:
                sigma = 1e-9

            # Compute trading window spread and normalize
            y_trade = df.loc[trade_mask, f"{sym1}_close"]
            x_trade = df.loc[trade_mask, f"{sym2}_close"]
            if y_trade.isna().all() or x_trade.isna().all():
                print(f"All NaNs in prices for {sym1}-{sym2} in window {trade_start} to {trade_end}. Skipping.")
                continue

            trade_spread = y_trade - (alpha + beta * x_trade)
            if trade_spread.isna().any():
                print(f"NaNs in trade spread for {sym1}-{sym2} in window {trade_start} to {trade_end}")

            # Column names
            spread_norm_col = f"{sym1}_{sym2}_spreadNorm"
            alpha_col = f"{sym1}_{sym2}_alpha"
            beta_col = f"{sym1}_{sym2}_beta"
            corr_col = f"{sym1}_{sym2}_corr"
            pval_col = f"{sym1}_{sym2}_pval"

            # Store normalized spread
            df.loc[trade_mask, spread_norm_col] = (trade_spread - mu) / sigma

            # Store alpha, beta, correlation for the trade period
            df.loc[trade_mask, alpha_col] = alpha
            df.loc[trade_mask, beta_col] = beta
            df.loc[trade_mask, corr_col] = corr
            df.loc[trade_mask, pval_col] = p_val

    return df


# ============================================================================
# 3. TECHNICAL FEATURE ENGINEERING
# ============================================================================

def ewma_volatility(series, lambda_=0.94):
    """
    Exponentially weighted volatility (EWMA).
    
    Args:
        series (pd.Series): Price series
        lambda_ (float): Decay factor (default: 0.94)
    
    Returns:
        pd.Series: EWMA volatility
    """
    returns = series.pct_change().dropna()
    return returns.ewm(alpha=1 - lambda_).std()


def kalman_ma(series, window=20):
    """
    Simple rolling mean as placeholder for Kalman filter.
    
    Args:
        series (pd.Series): Price series
        window (int): Rolling window size
    
    Returns:
        pd.Series: Rolling mean
    """
    return series.rolling(window=window, min_periods=1).mean()


def compute_features_day(df_day):
    """
    Compute all features for a single day dataframe.
    Only for relevant assets and spreads (non-NA normalized spreads)
    
    Args:
        df_day (pd.DataFrame): Daily data with prices and spreads
    
    Returns:
        pd.DataFrame: Data with computed features
    """
    out = df_day.copy()
    
    # Identify spread columns for today that are not NA
    spread_cols = [col for col in df_day.columns if col.endswith("_spreadNorm") and df_day[col].notna().any()]
    
    # Identify all assets involved today
    assets = set()
    for col in spread_cols:
        parts = col.split("_")
        sym1, sym2 = parts[0], parts[1]
        assets.add(sym1)
        assets.add(sym2)
    
    price_cols = [f"{asset}_close" for asset in assets if f"{asset}_close" in df_day.columns]

    # --------------------------
    # Price-based features
    # --------------------------
    for col in price_cols:
        prices = df_day[col].astype(float)
        # MACD
        out[f"{col}Macd"] = ta.trend.MACD(prices, window_slow=14, window_fast=5).macd()
        # RSI
        out[f"{col}Rsi"] = ta.momentum.RSIIndicator(prices, window=20).rsi()
        # Kalman MA
        out[f"{col}Kalman"] = kalman_ma(prices, window=20)
        # EWMA volatility
        out[f"{col}EwmaVol"] = ewma_volatility(prices)
        # Bias (price - dynamic MA)
        out[f"{col}Bias"] = prices - out[f"{col}Kalman"]
        # Signs: ratio of positive vs negative returns over 10-min rolling window
        returns = prices.pct_change()
        out[f"{col}Signs"] = returns.rolling(window=10, min_periods=1).apply(
            lambda x: (x > 0).sum() / max(1, len(x)), raw=False
        )
        # Stochastic RSI (simplified)
        rsi = out[f"{col}Rsi"]
        out[f"{col}StochRsi"] = (rsi - rsi.rolling(20, min_periods=1).min()) / (
            rsi.rolling(20, min_periods=1).max() - rsi.rolling(20, min_periods=1).min() + 1e-12
        )
        # Upper/Lower shadow proxies
        out[f"{col}UpperShadow"] = prices - prices.rolling(2).min()
        out[f"{col}LowerShadow"] = prices.rolling(2).max() - prices
    
    # --------------------------
    # Spread-based features
    # --------------------------
    for col in spread_cols:
        s = df_day[col].astype(float)
        out[col] = s
        out[f"{col}Vol"] = ewma_volatility(s)
        out[f"{col}Kalman"] = kalman_ma(s, window=20)
        # Moving average of spread
        out[f"{col}Ma"] = s.rolling(window=20, min_periods=1).mean()

    return out


def compute_all_features(df):
    """
    Compute features day by day for all data.
    
    Args:
        df (pd.DataFrame): Full dataframe with timestamp, log prices, normalized spreads
    
    Returns:
        pd.DataFrame: Data with all computed features
    """
    df["date"] = df["timestamp"].dt.date
    feature_dfs = []
    
    for day, df_day in df.groupby("date"):
        print(f"Processing {day} ...")
        day_feats = compute_features_day(df_day)
        feature_dfs.append(day_feats)
    
    # Combine all days
    full_df = pd.concat(feature_dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    
    # Drop intermediate columns
    cols_to_drop = [col for col in full_df.columns if col.endswith("_spread")]
    full_df = full_df.drop(columns=cols_to_drop, axis=1)
    
    return full_df


# ============================================================================
# 4. FUNDING RATE MERGING
# ============================================================================

def funding_interval(symbol):
    """
    Get funding interval for a symbol.
    
    Args:
        symbol (str): Symbol name (without USDT)
    
    Returns:
        pd.Timedelta: Funding interval
    """
    if symbol in ["TON", "ENA"]:
        return pd.Timedelta(hours=4)
    else:
        return pd.Timedelta(hours=8)


def merge_funding_rates(features_df, funding_data_folder, symbols=None):
    """
    Merge funding rate data into features dataframe.
    
    Args:
        features_df (pd.DataFrame): Features dataframe with 'datetime' column
        funding_data_folder (str): Path to folder with funding rate parquet files
        symbols (list): List of symbols to merge (e.g., ['BTC', 'ETH', ...])
                       If None, uses default 25 symbols
    
    Returns:
        pd.DataFrame: Features dataframe with funding rate columns
    """
    if symbols is None:
        symbols = [
            "AAVE", "ADA", "APT", "ARB", "ATOM",
            "AVAX", "BCH", "BNB", "BTC", "DOGE",
            "DOT", "ENA", "ETC", "ETH", "HBAR",
            "LINK", "LTC", "NEAR", "SUI", "SOL",
            "TON", "TRX", "UNI", "WLD", "XLM", "XRP"
        ]
    
    data_dir = Path(funding_data_folder)
    df = features_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for sym in symbols:
        try:
            path = data_dir / f"{sym}USDT_funding_bin_futures.parquet"
            funding_df = pd.read_parquet(path)
            funding_df["datetime"] = pd.to_datetime(funding_df["datetime"])
            funding_df = funding_df.sort_values("datetime").reset_index(drop=True)

            # Compute next funding timestamp for each record
            interval = funding_interval(sym)
            funding_df["next_funding_time"] = funding_df["datetime"].shift(-1)
            funding_df.loc[funding_df["next_funding_time"].isna(), "next_funding_time"] = (
                funding_df["datetime"] + interval
            )

            # Initialize columns
            rate_col = f"{sym}_funding"
            time_col = f"{sym}_fundingMinutesLeft"
            df[rate_col] = None
            df[time_col] = None

            for _, row in funding_df.iterrows():
                mask = (df["datetime"] >= row["datetime"]) & (df["datetime"] < row["next_funding_time"])
                minutes_left = ((row["next_funding_time"] - df.loc[mask, "datetime"]).dt.total_seconds() / 60).astype(int)
                df.loc[mask, rate_col] = row["funding_rate"]
                df.loc[mask, time_col] = minutes_left

            print(f"✅ Added {sym}: funding rate + minutes left (interval {interval.components.hours}h)")

        except Exception as e:
            print(f"⚠️ Error adding {sym}: {e}")

    print("\n🎯 Done — funding rates and countdown columns added!")
    return df


# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================

def run_full_pipeline(cointegration_folder,
                      input_csv,
                      output_file="final_features.csv",
                      funding_data_folder=None,
                      symbols=None,
                      trade_minutes=1440,
                      top_k=5,
                      sep=";"):
    """
    Complete feature engineering pipeline from cointegration to final features.
    
    Args:
        cointegration_folder (str): Path to folder with cointegration CSVs
        input_csv (str): Path to input CSV with prices and raw spreads
        output_file (str): Path to save output CSV (default: "final_features.csv")
        funding_data_folder (str): Path to folder with funding rate parquets
        symbols (list): List of crypto symbols for funding rates
        trade_minutes (int): Trading window duration in minutes (default: 1440 = 1 day)
        top_k (int): Number of top pairs to keep per window (default: 5)
        sep (str): CSV separator (default: ";")
    
    Returns:
        pd.DataFrame: Final features dataframe
    """
    print("=" * 80)
    print("STEP 1: Loading cointegration results")
    print("=" * 80)
    cointegration_dict = load_cointegration_csvs(cointegration_folder)
    print(f"✅ Loaded {len(cointegration_dict)} pairs\n")

    print("=" * 80)
    print("STEP 2: Selecting top pairs per window")
    print("=" * 80)
    top_pairs_per_window = select_top_pairs_per_window(cointegration_dict, top_k=top_k)
    print(f"✅ Selected top {top_k} pairs for {len(top_pairs_per_window)} windows\n")

    print("=" * 80)
    print("STEP 3: Loading input data")
    print("=" * 80)
    df = pd.read_csv(input_csv, sep=sep)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"✅ Loaded data with shape {df.shape}\n")

    print("=" * 80)
    print("STEP 4: Normalizing spreads")
    print("=" * 80)
    df_norm = normalize_spreads(df, top_pairs_per_window, folder_path=cointegration_folder, 
                                trade_minutes=trade_minutes)
    print(f"✅ Normalized spreads. Shape: {df_norm.shape}\n")

    print("=" * 80)
    print("STEP 5: Computing technical features")
    print("=" * 80)
    df_features = compute_all_features(df_norm)
    print(f"✅ Computed all features. Shape: {df_features.shape}\n")

    print("=" * 80)
    print("STEP 6: Merging funding rates")
    print("=" * 80)
    if funding_data_folder and os.path.exists(funding_data_folder):
        df_features = df_features.rename(columns={"timestamp": "datetime"})
        df_final = merge_funding_rates(df_features, funding_data_folder, symbols=symbols)
    else:
        print("⚠️ No funding data folder provided. Skipping funding rate merge.\n")
        df_final = df_features

    print("=" * 80)
    print("STEP 7: Saving output")
    print("=" * 80)
    df_final.to_csv(output_file, index=False)
    print(f"✅ Saved final features to {output_file}\n")
    print(f"Final shape: {df_final.shape}")

    return df_final


if __name__ == "__main__":
    # Example usage
    features_df = run_full_pipeline(
        cointegration_folder="",  # Replace with your path
        input_csv="bin_futures_historical_pairs_with_spreads.csv",
        output_file="bin_futures_full_features.csv",
        funding_data_folder="",  # Replace with your path (optional)
        symbols=None,  # Will use default 26 symbols
        trade_minutes=1440,
        top_k=5,
        sep=";"
    )
