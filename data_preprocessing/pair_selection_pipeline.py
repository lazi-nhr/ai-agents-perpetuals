"""
Cryptocurrency Pair Selection and Cointegration Pipeline

This module provides a complete pipeline for:
1. Loading and filtering crypto data
2. Computing correlation matrices
3. Identifying cointegrated pairs
4. Selecting top pairs per rolling window
5. Building feature matrices with spreads and statistics
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import Counter
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller


# ============================================================================
# 1. DATA LOADING AND FILTERING
# ============================================================================

def load_and_filter(folder, start="2024-05-01 00:00:00", end="2025-05-01 00:00:00", 
                    file_pattern="*_1m_bin_futures.parquet"):
    """
    Load 1-minute crypto parquet files with full data coverage,
    filter by datetime range, and compute log prices & returns.
    
    Args:
        folder (str): Path to folder containing parquet files
        start (str): Start timestamp (default: "2024-05-01 00:00:00")
        end (str): End timestamp (default: "2025-05-01 00:00:00")
        file_pattern (str): Glob pattern for files to load
    
    Returns:
        dict: {symbol: DataFrame with log prices and returns}
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    data = {}

    for f in glob.glob(os.path.join(folder, file_pattern)):
        sym = os.path.basename(f).replace("_1m_bin_futures.parquet", "").replace("USDT", "")
        df = pd.read_parquet(f)

        # Convert to datetime if needed
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        else:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Filter by the specified time window
        df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
        df = df.set_index("datetime").sort_index()

        # Log prices and returns
        df["close"] = np.log(df["close"])
        df["open"] = np.log(df["open"])
        df["high"] = np.log(df["high"])
        df["low"] = np.log(df["low"])
        df["log_return"] = df["close"].diff()

        data[sym] = df
        print(f"Loaded {sym}, {len(df)} rows")

    return data


# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================

def compute_correlation_matrix(crypto_data):
    """
    Compute correlation matrix of closing prices for all cryptos.
    
    Args:
        crypto_data (dict): {symbol: DataFrame}
    
    Returns:
        pd.DataFrame: Correlation matrix with symbols as index and columns
    """
    symbols = list(crypto_data.keys())
    close_prices = pd.DataFrame({sym: crypto_data[sym]["close"] for sym in symbols})
    corr_matrix = close_prices.corr()
    return corr_matrix


def find_high_correlation_pairs(crypto_data, correlation_matrix, threshold=0.85):
    """
    Select pairs with correlation above threshold.
    
    Args:
        crypto_data (dict): {symbol: DataFrame}
        correlation_matrix (pd.DataFrame): Correlation matrix
        threshold (float): Minimum absolute correlation (default: 0.85)
    
    Returns:
        list: List of tuples (sym1, sym2) with high correlation
    """
    high_corr_pairs = []
    symbols = list(crypto_data.keys())
    
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sym1, sym2 = symbols[i], symbols[j]
            corr = correlation_matrix.loc[sym1, sym2]
            if abs(corr) > threshold:
                high_corr_pairs.append((sym1, sym2))
                print(f"High correlation: {sym1} & {sym2} = {corr:.2f}")
    
    return high_corr_pairs


# ============================================================================
# 3. COINTEGRATION TESTING
# ============================================================================

def rolling_cointegration(y, x, window=4320, adf_pval=0.05):
    """
    Rolling Engle–Granger cointegration test with beta estimation.
    
    Args:
        y (pd.Series): First price series (index: datetime)
        x (pd.Series): Second price series (index: datetime)
        window (int): Rolling window size (default: 4320 = 3 days of 1-min bars)
        adf_pval (float): ADF test p-value threshold (default: 0.05)
    
    Returns:
        pd.DataFrame: Cointegration results with columns:
            - start: Window start timestamp
            - end: Window end timestamp
            - alpha: Intercept from OLS regression
            - beta: Slope from OLS regression
            - adf_p: ADF test p-value
            - cointegrated: Boolean cointegration status
            - correlation: Pearson correlation in window
    """
    y, x = y.align(x, join="inner")
    y, x = y.sort_index(), x.sort_index()

    results = []
    step = window // 3  # 1/3 overlap
    timestamps = y.index

    for i in range(0, len(timestamps) - window, step):
        start_time = timestamps[i]
        end_time = timestamps[i + window - 1]

        y_win = y.loc[start_time:end_time]
        x_win = x.loc[start_time:end_time]

        if len(y_win) < window or y_win.isna().any() or x_win.isna().any():
            continue

        model = OLS(y_win, add_constant(x_win)).fit()
        alpha, beta = model.params

        residuals = y_win - model.predict(add_constant(x_win))
        adf_p = adfuller(residuals)[1]
        corr = y_win.corr(x_win)

        results.append({
            "start": start_time,
            "end": end_time,
            "alpha": alpha,
            "beta": beta,
            "adf_p": adf_p,
            "cointegrated": adf_p <= adf_pval,
            "correlation": corr
        })

    return pd.DataFrame(results)


def prepare_all_pairs(crypto_data, list_of_pairs, window=4320, adf_pval=0.05):
    """
    Run rolling cointegration test on all pairs.
    
    Args:
        crypto_data (dict): {symbol: DataFrame}
        list_of_pairs (list): List of (sym1, sym2) tuples
        window (int): Rolling window size
        adf_pval (float): ADF p-value threshold
    
    Returns:
        dict: {(sym1, sym2): cointegration_results_df}
    """
    pair_df = {}
    
    for i, (sym1, sym2) in enumerate(list_of_pairs, 1):
        print(f"Processing pair: {sym1}, {sym2}. {i} of {len(list_of_pairs)}")
        
        y_ohlc = crypto_data[sym1]
        x_ohlc = crypto_data[sym2]

        # Align close prices
        df_close = pd.concat([y_ohlc["close"], x_ohlc["close"]], axis=1, join="inner").dropna()
        y_aligned, x_aligned = df_close.iloc[:, 0], df_close.iloc[:, 1]

        # Rolling cointegration
        coint_df = rolling_cointegration(y_aligned, x_aligned, window=window, adf_pval=adf_pval)
        pair_df[(sym1, sym2)] = coint_df

    return pair_df


# ============================================================================
# 4. PAIR SELECTION (TOP-K PER WINDOW)
# ============================================================================

def select_top_pairs_per_window(coint_df, top_k=5):
    """
    Select top-K cointegrated pairs per rolling window based on correlation.
    
    Args:
        coint_df (dict): {(sym1, sym2): cointegration_results_df}
        top_k (int): Number of top pairs to keep per window (default: 5)
    
    Returns:
        dict: {(start, end): [(pair, corr, beta, alpha, adf_p), ...]}
    """
    top_pairs_per_window = {}
    
    # Collect all cointegrated pairs per window
    for pair, df in coint_df.items():
        for _, row in df.iterrows():
            window_key = (row["start"], row["end"])
            if window_key not in top_pairs_per_window:
                top_pairs_per_window[window_key] = []
            if row["cointegrated"]:
                top_pairs_per_window[window_key].append((
                    pair, row["correlation"], row["beta"], row["alpha"], row["adf_p"]
                ))
    
    # Keep only top-K by absolute correlation
    for window_key, pairs in top_pairs_per_window.items():
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_pairs_per_window[window_key] = pairs[:top_k]
    
    return top_pairs_per_window


def count_pair_occurrences(top_pairs_per_window):
    """
    Count how many times each pair appears in top-K across all windows.
    
    Args:
        top_pairs_per_window (dict): Output from select_top_pairs_per_window
    
    Returns:
        Counter: {(sym1, sym2): count}
    """
    pair_counter = Counter()
    for pairs in top_pairs_per_window.values():
        for pair_info in pairs:
            pair = pair_info[0]
            pair_counter[pair] += 1
    return pair_counter


# ============================================================================
# 5. FEATURE BUILDING
# ============================================================================

def build_full_features(crypto_data, top_pairs_per_window):
    """
    Build a comprehensive feature matrix with:
    - Log prices for all symbols
    - Spreads for cointegrated pairs (window-specific beta/alpha)
    - Beta, alpha, ADF p-value, and correlation for each pair per window
    
    Args:
        crypto_data (dict): {symbol: DataFrame}
        top_pairs_per_window (dict): Output from select_top_pairs_per_window
    
    Returns:
        pd.DataFrame: Feature matrix with timestamp index
    """
    # Collect all unique symbols from top pairs
    all_symbols = sorted({
        sym
        for pairs in top_pairs_per_window.values()
        for pair_info in pairs
        for sym in pair_info[0]
    })

    # Initialize base DataFrame with full timestamp index
    first_sym = all_symbols[0]
    full_df = pd.DataFrame(index=crypto_data[first_sym].index)
    full_df.index.name = "timestamp"

    # Add log-price columns for each symbol
    for sym in all_symbols:
        df = crypto_data[sym][["close"]].rename(columns={"close": f"{sym}_close"})
        full_df = full_df.join(df, how="left")

    # For each window and cointegrated pair, compute spreads and statistics
    for (start, end), pairs in top_pairs_per_window.items():
        mask = (full_df.index >= start) & (full_df.index <= end)

        for pair_info in pairs:
            (sym1, sym2), corr, beta, alpha, adf_p = pair_info

            spread_col = f"{sym1}_{sym2}_spread"
            beta_col = f"{sym1}_{sym2}_beta"
            alpha_col = f"{sym1}_{sym2}_alpha"
            adf_col = f"{sym1}_{sym2}_adf_p"
            corr_col = f"{sym1}_{sym2}_corr"

            # Initialize columns if they don't exist
            for col in [spread_col, beta_col, alpha_col, adf_col, corr_col]:
                if col not in full_df.columns:
                    full_df[col] = np.nan

            # Compute spread for this window
            y = full_df.loc[mask, f"{sym1}_close"]
            x = full_df.loc[mask, f"{sym2}_close"]

            full_df.loc[mask, spread_col] = y - (alpha + beta * x)
            full_df.loc[mask, beta_col] = beta
            full_df.loc[mask, alpha_col] = alpha
            full_df.loc[mask, adf_col] = adf_p
            full_df.loc[mask, corr_col] = corr

    return full_df


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def run_pair_selection_pipeline(folder, start_date, end_date, 
                                corr_threshold=0.85, window=4320, 
                                adf_pval=0.05, top_k=5, 
                                file_pattern="*_1m_bin_futures.parquet",
                                output_file=None):
    """
    Complete pipeline: load data → correlations → cointegration → feature building
    
    Args:
        folder (str): Path to parquet files
        start_date (str): Start timestamp
        end_date (str): End timestamp
        corr_threshold (float): Correlation threshold (default: 0.85)
        window (int): Rolling window size (default: 4320)
        adf_pval (float): ADF p-value threshold (default: 0.05)
        top_k (int): Number of top pairs per window (default: 5)
        file_pattern (str): Glob pattern for files
        output_file (str): Optional path to save CSV output
    
    Returns:
        tuple: (crypto_data, correlation_matrix, high_corr_pairs, 
                coint_df, top_pairs_per_window, full_features_df)
    """
    print("=" * 80)
    print("STEP 1: Loading and filtering data")
    print("=" * 80)
    crypto_data = load_and_filter(folder, start_date, end_date, file_pattern)
    print(f"✅ Loaded {len(crypto_data)} cryptocurrencies\n")

    print("=" * 80)
    print("STEP 2: Computing correlation matrix")
    print("=" * 80)
    corr_matrix = compute_correlation_matrix(crypto_data)
    print(f"✅ Correlation matrix computed\n")

    print("=" * 80)
    print("STEP 3: Finding high-correlation pairs")
    print("=" * 80)
    high_corr_pairs = find_high_correlation_pairs(crypto_data, corr_matrix, corr_threshold)
    print(f"✅ Found {len(high_corr_pairs)} high-correlation pairs\n")

    print("=" * 80)
    print("STEP 4: Running rolling cointegration tests")
    print("=" * 80)
    coint_df = prepare_all_pairs(crypto_data, high_corr_pairs, window, adf_pval)
    print(f"✅ Cointegration tests completed\n")

    print("=" * 80)
    print("STEP 5: Selecting top pairs per window")
    print("=" * 80)
    top_pairs_per_window = select_top_pairs_per_window(coint_df, top_k)
    pair_counter = count_pair_occurrences(top_pairs_per_window)
    print(f"✅ Selected top {top_k} pairs for {len(top_pairs_per_window)} windows")
    print(f"✅ Found {len(pair_counter)} unique pairs across all windows\n")

    print("=" * 80)
    print("STEP 6: Building feature matrix")
    print("=" * 80)
    full_features_df = build_full_features(crypto_data, top_pairs_per_window)
    print(f"✅ Feature matrix shape: {full_features_df.shape}\n")

    if output_file:
        full_features_df.to_csv(output_file)
        print(f"✅ Saved features to {output_file}\n")

    return (crypto_data, corr_matrix, high_corr_pairs, 
            coint_df, top_pairs_per_window, full_features_df)


# ============================================================================
# 7. UTILITY FUNCTIONS
# ============================================================================

def save_cointegration_results(coint_df, output_dir):
    """
    Save cointegration results for each pair to CSV.
    
    Args:
        coint_df (dict): {(sym1, sym2): cointegration_results_df}
        output_dir (str): Directory to save files
    """
    for (sym1, sym2), df in coint_df.items():
        filename = os.path.join(output_dir, f"{sym1}_{sym2}_cointegration.csv")
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")


def get_unique_symbols(top_pairs_per_window):
    """
    Get all unique symbols appearing in top pairs across all windows.
    
    Args:
        top_pairs_per_window (dict): Output from select_top_pairs_per_window
    
    Returns:
        set: Unique symbols
    """
    unique_symbols = set()
    for pairs in top_pairs_per_window.values():
        for pair_info in pairs:
            sym1, sym2 = pair_info[0]
            unique_symbols.add(sym1)
            unique_symbols.add(sym2)
    return unique_symbols


if __name__ == "__main__":
    # Example usage
    crypto_data, corr_matrix, high_corr_pairs, coint_df, top_pairs_per_window, features = (
        run_pair_selection_pipeline(
            folder="",
            start_date="2024-05-01 00:00:00",
            end_date="2025-05-01 00:00:00",
            corr_threshold=0.85,
            window=4320,
            adf_pval=0.05,
            top_k=5,
            file_pattern="*_1m_bin_futures.parquet",
            output_file="historical_pairs_features.csv"
        )
    )
