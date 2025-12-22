import os
import json
import time
import requests
import numpy as np
import pandas as pd
import ta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter
from datetime import datetime
from itertools import combinations

# Import your fixed inference logic
try:
    from inference_rl import generate_signal_from_features
except ImportError:
    print("CRITICAL ERROR: Could not import 'inference_rl.py'.")
    exit()

# ================= CONFIGURATION =================
MODEL_PATH = "models/final_model.zip"
OUTPUT_FILE = "signal.json"
TIMEFRAME = "1h"
LOOKBACK = 30 

# The Universe (from your log)
ASSETS = [
    "AAVE", "ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC", "DOGE", 
    "DOT", "ENA", "ETC", "ETH", "HBAR", "LINK", "LTC", "NEAR", "SOL", "SUI", 
    "TON", "TRX", "UNI", "WLD", "XLM", "XRP"
]

# ================= DATA FETCHING =================
def fetch_all_candles(assets, limit=200):
    """Fetches history for ALL assets in the universe."""
    data_store = {}
    url = "https://api.hyperliquid.xyz/info"
    start_time = int(time.time() * 1000) - (limit * 3600 * 1000)
    
    print(f"Fetching data for {len(assets)} assets...")
    
    for coin in assets:
        body = {"type": "candleSnapshot", "req": {"coin": coin, "interval": TIMEFRAME, "startTime": start_time}}
        try:
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=body)
            raw = resp.json()
            if raw:
                df = pd.DataFrame(raw)
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                for c, key in [('close', 'c'), ('high', 'h'), ('low', 'l'), ('volume', 'v')]:
                    df[c] = df[key].astype(float)
                data_store[coin] = df.set_index('timestamp').sort_index()
                time.sleep(0.05) # Rate limit safety
        except Exception as e:
            print(f"Failed to fetch {coin}: {e}")
            
    return data_store

# ================= PAIR SELECTION LOGIC =================
# Adapted from 'pair_selection_pipeline.py' for live use
def check_cointegration(p1, p2):
    """
    Returns (is_coint, score, alpha, beta)
    """
    # 1. Correlation Check
    corr = p1.corr(p2)
    if abs(corr) < 0.85: # Threshold from your pipeline
        return False, 0, 0, 0

    # 2. Cointegration (Engle-Granger)
    # OLS: p1 = alpha + beta * p2 + epsilon
    X = sm.add_constant(p2)
    model = sm.OLS(p1, X).fit()
    alpha, beta = model.params[0], model.params[1]
    
    residuals = p1 - (alpha + beta * p2)
    
    # ADF Test on residuals
    # If p-value < 0.05, residuals are stationary -> Pairs are cointegrated
    adf_result = adfuller(residuals)
    p_value = adf_result[1]
    
    is_coint = p_value < 0.05
    
    # We return correlation as the "score" to pick the best one, as per your pipeline
    return is_coint, abs(corr), alpha, beta

# ================= FEATURE ENGINEERING =================
def z_score(series):
    return (series - series.rolling(LOOKBACK).mean()) / (series.rolling(LOOKBACK).std() + 1e-8)

def kalman_ma(series):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], 
                      initial_state_mean=0, initial_state_covariance=1, 
                      observation_covariance=1, transition_covariance=0.01)
    state_means, _ = kf.filter(series.values)
    return pd.Series(state_means.flatten(), index=series.index)

def ewma_volatility(series, alpha=0.06):
    return series.pct_change().ewm(alpha=alpha).std()

def calculate_features(df1, df2, alpha, beta):
    """Calculates features using the Alpha/Beta found during selection."""
    idx = df1.index.intersection(df2.index)
    p1 = df1.loc[idx, 'close']
    p2 = df2.loc[idx, 'close']

    # Spread using the specific Alpha/Beta for this pair
    spread = p1 - (beta * p2) # Note: Simplified spread definition usually used in live
    spread_norm = z_score(spread)
    
    pair_feats = {
        'alpha': pd.Series(alpha, index=idx), 
        'beta': pd.Series(beta, index=idx), 
        'corr': p1.rolling(30).corr(p2),
        'pval': pd.Series(0, index=idx), # Placeholder
        'spreadNorm': spread_norm,
        'spreadNormKalman': z_score(kalman_ma(spread)),
        'spreadNormMa': z_score(spread.rolling(20).mean()),
        'spreadNormVol': z_score(spread.rolling(20).std())
    }

    # Asset Features
    def get_feats(df_raw, p):
        rsi = ta.momentum.RSIIndicator(p, window=20).rsi()
        return {
            'close': z_score(p),
            'closeBias': z_score((p - p.ewm(span=30).mean())/p.ewm(span=30).mean()),
            'closeEwmaVol': z_score(ewma_volatility(p)),
            'closeKalman': z_score(kalman_ma(p)),
            'closeLowerShadow': z_score(df_raw.loc[idx, 'low'] - p),
            'closeMacd': z_score(ta.trend.MACD(p).macd()),
            'closeRsi': z_score(rsi),
            'closeSigns': np.sign(p.diff()).fillna(0),
            'closeStochRsi': z_score(rsi), 
            'closeUpperShadow': z_score(df_raw.loc[idx, 'high'] - p),
            'funding': p*0, 'fundingMinutesLeft': p*0
        }

    return get_feats(df1, p1), get_feats(df2, p2), pair_feats

def to_list_dict(d):
    """Converts Series dict to list dict for the model."""
    return {k: v.iloc[-LOOKBACK:].fillna(0).tolist() for k, v in d.items()}

# ================= EXECUTION LOOP =================
if __name__ == "__main__":
    print(f"--- STARTING UNIVERSE SCANNER ---")
    print(f"Universe: {len(ASSETS)} assets")
    print(f"Model: {MODEL_PATH}")
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning Universe...")
            
            # 1. Fetch Data for ALL assets
            market_data = fetch_all_candles(ASSETS)
            
            # Filter out empty dataframes
            valid_assets = [k for k, v in market_data.items() if len(v) > LOOKBACK + 50]
            if len(valid_assets) < 2:
                print("Not enough valid data. Sleeping...")
                time.sleep(10)
                continue

            # 2. Find Best Pair
            best_pair_info = None
            best_score = -1.0 # Correlation score
            
            # Generate all unique pairs
            pairs = list(combinations(valid_assets, 2))
            print(f"Scanning {len(pairs)} pairs for cointegration...")
            
            for a1, a2 in pairs:
                df1 = market_data[a1]
                df2 = market_data[a2]
                idx = df1.index.intersection(df2.index)
                
                if len(idx) < 100: continue

                # Check Cointegration
                is_coint, corr, alpha, beta = check_cointegration(df1.loc[idx, 'close'], df2.loc[idx, 'close'])
                
                if is_coint and corr > best_score:
                    best_score = corr
                    best_pair_info = (a1, a2, alpha, beta)
            
            if not best_pair_info:
                print("No cointegrated pairs found above threshold.")
                time.sleep(60)
                continue

            # 3. Generate Signal for the Winner
            a1, a2, alpha, beta = best_pair_info
            print(f"WINNER: {a1}-{a2} (Corr: {best_score:.4f})")
            
            f1, f2, f_pair = calculate_features(market_data[a1], market_data[a2], alpha, beta)
            
            signal = generate_signal_from_features(
                model_path=MODEL_PATH,
                asset1=a1, asset2=a2,
                asset1_features=to_list_dict(f1),
                asset2_features=to_list_dict(f2),
                pair_features=to_list_dict(f_pair),
                lookback_window=LOOKBACK,
                notional_usd=100
            )
            
            # 4. Save Signal
            with open(OUTPUT_FILE, "w") as f:
                json.dump(signal, f, indent=4)
                
            print(f"SIGNAL SAVED for {a1}-{a2}: Weights {signal['weights']}")
            print("Sleeping for 60 seconds...")
            time.sleep(60)

        except Exception as e:
            print(f"Global Error: {e}")
            time.sleep(10)