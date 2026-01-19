import os
import json
import time
import requests
import numpy as np
import pandas as pd
import ta
from pykalman import KalmanFilter
from datetime import datetime

# Import the logic from other file
try:
    from inference_rl import generate_signal_from_features
except ImportError:
    print("CRITICAL ERROR: Could not import 'inference_rl.py'.")
    print("Make sure inference_rl.py is in the same folder as this script.")
    exit()

# ================= CONFIGURATION =================
MODEL_PATH = "models/best_model.zip"
OUTPUT_FILE = "signal.json"
ASSET1 = "ETH"
ASSET2 = "BTC"
TIMEFRAME = "1h"
LOOKBACK = 30  # Must match model's training lookback

# ================= DEBUGGING PRINT =================
print(f"--- INITIALIZING GENERATOR FOR {ASSET1}-{ASSET2} ---")
print(f"Model Path: {MODEL_PATH}")

# ================= DATA FETCHING =================
def fetch_candles(coin, limit=200):
    """Fetch live candles from Hyperliquid"""
    url = "https://api.hyperliquid.xyz/info"
    # Calculate start time in ms
    start_time = int(time.time() * 1000) - (limit * 3600 * 1000)
    
    body = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": TIMEFRAME,
            "startTime": start_time
        }
    }
    
    try:
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=body)
        data = resp.json()
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        # Convert types
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        for c, key in [('close', 'c'), ('high', 'h'), ('low', 'l'), ('volume', 'v')]:
            df[c] = df[key].astype(float)
            
        return df.set_index('timestamp').sort_index()
    except Exception as e:
        print(f"Error fetching {coin}: {e}")
        return pd.DataFrame()

# ================= MATH HELPERS (From your pipeline) =================
def z_score(series):
    return (series - series.rolling(LOOKBACK).mean()) / (series.rolling(LOOKBACK).std() + 1e-8)

def kalman_ma(series):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], 
                      initial_state_mean=0, initial_state_covariance=1, 
                      observation_covariance=1, transition_covariance=0.01)
    # Filter returns tuple (means, covariances), we want means
    state_means, _ = kf.filter(series.values)
    return pd.Series(state_means.flatten(), index=series.index)

def ewma_volatility(series, alpha=0.06):
    return series.pct_change().ewm(alpha=alpha).std()

# ================= MAIN EXECUTION LOOP =================
if __name__ == "__main__":
    print("--- Starting Loop ---")
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching fresh data...")
            
            # 1. Fetch Data
            df1 = fetch_candles(ASSET1)
            df2 = fetch_candles(ASSET2)
            
            if df1.empty or df2.empty:
                print("No data received. Retrying in 10s...")
                time.sleep(10)
                continue
                
            print(f"Got {len(df1)} rows for {ASSET1} and {len(df2)} rows for {ASSET2}")

            # 2. Align Data
            idx = df1.index.intersection(df2.index)
            if len(idx) < LOOKBACK:
                print(f"Not enough common data ({len(idx)} rows). Need {LOOKBACK}.")
                time.sleep(10)
                continue
                
            p1 = df1.loc[idx, 'close']
            p2 = df2.loc[idx, 'close']

            # 3. Engineer Features
            # A. Pair Features
            cov = p1.rolling(30).cov(p2)
            var = p2.rolling(30).var()
            beta = (cov / var).fillna(1.0)
            alpha = p1.rolling(30).mean() - beta * p2.rolling(30).mean()
            spread = p1 - (beta * p2)
            spread_norm = z_score(spread)
            
            pair_feats = {
                'alpha': alpha, 'beta': beta, 'corr': p1.rolling(30).corr(p2),
                'pval': spread*0, 'spreadNorm': spread_norm,
                'spreadNormKalman': z_score(kalman_ma(spread)),
                'spreadNormMa': z_score(spread.rolling(20).mean()),
                'spreadNormVol': z_score(spread.rolling(20).std())
            }

            # B. Asset Features Helper
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
                    'closeStochRsi': z_score(rsi), # Simplified stoch
                    'closeUpperShadow': z_score(df_raw.loc[idx, 'high'] - p),
                    'funding': p*0, 'fundingMinutesLeft': p*0
                }

            f1 = get_feats(df1, p1)
            f2 = get_feats(df2, p2)

            # 4. Convert to List Format for Inference
            # We take the last 'LOOKBACK' values
            def to_list_dict(d):
                return {k: v.iloc[-LOOKBACK:].fillna(0).tolist() for k, v in d.items()}

            print("Running AI Inference...")
            signal = generate_signal_from_features(
                model_path=MODEL_PATH,
                asset1=ASSET1, asset2=ASSET2,
                asset1_features=to_list_dict(f1),
                asset2_features=to_list_dict(f2),
                pair_features=to_list_dict(pair_feats),
                lookback_window=LOOKBACK,
                notional_usd=100
            )

            # 5. Save Signal
            with open(OUTPUT_FILE, "w") as f:
                json.dump(signal, f, indent=4)
                
            print(f"SUCCESS! Signal generated at {signal['timestamp']}")
            print(f"Weights: {signal['weights']}")
            print("Sleeping for 60 seconds...")
            time.sleep(60)

        except Exception as e:
            print(f"\n!!! ERROR IN LOOP !!! : {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)
