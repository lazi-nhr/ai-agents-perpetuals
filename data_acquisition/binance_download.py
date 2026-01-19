"""
Crypto OHLCV Downloader Script
------------------------------

This script downloads historical OHLCV (Open, High, Low, Close, Volume) data 
for the top N cryptocurrencies based on market capitalization. 

It uses the following components:
- CoinGecko API: to retrieve a list of the top N cryptocurrencies by market cap.
- CCXT: to connect to cryptocurrency exchanges (e.g., Binance) and fetch historical OHLCV data.
- Pandas: to store and save the data as parquet files.
- tqdm: to show progress during data download.

Key Features:
-------------
1. Robust error handling: retries on network failures or API issues.
2. Efficient download: skips existing files (resumable downloads).
3. Configurable mode: test mode (few days of selected symbols) or production mode (full year of top 50 symbols).
4. Supports multiple timeframes: 1m, 5m, 15m, 1h, 1d.

Usage Requirements:
-------------------
- Python 3.8+
- pip install: `ccxt pandas tqdm requests`

The script stores data locally in the specified folder, organized per symbol and timeframe.
"""

import requests
import pandas as pd
import ccxt
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm

def get_top_n_crypto_by_market_cap(n=50):
    """
    Use the CoinGecko API to fetch the top N cryptocurrencies by market capitalization.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': n, 'page': 1, 'sparkline': 'false'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        symbols = [item['symbol'].upper() for item in data]
        print(f"Successfully retrieved Top {len(symbols)} cryptocurrencies.")
        return symbols
    except requests.exceptions.RequestException as e:
        print(f"Error while fetching data from CoinGecko: {e}")
        print("Falling back to a predefined static list.")
        return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'SHIB', 'AVAX', 'DOT', 'TRX', 'LINK', 'MATIC', 'ICP', 'BCH', 'LTC', 'NEAR', 'UNI', 'LEO', 'XLM', 'OKB', 'INJ', 'ETC', 'HBAR', 'FIL', 'CRO', 'APT', 'IMX', 'ATOM', 'XMR', 'STX', 'MKR', 'GRT', 'RNDR', 'VET', 'OP', 'AAVE', 'LDO', 'THETA', 'SEI', 'JUP', 'TIA', 'ARB', 'KAS', 'BSV', 'EGLD', 'SUI', 'ALGO', 'RUNE']

class CryptoDownloader:
    def __init__(self, exchange_name='binance', data_dir='data'):
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({'options': {'defaultType': 'spot'}, 'enableRateLimit': True})
            print(f"Successfully initialized exchange: {self.exchange.id}")
        except AttributeError:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
            
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")

    def _fetch_ohlcv_robust(self, symbol_pair, timeframe, since, limit=1000):
        try:
            if self.exchange.has['fetchOHLCV']:
                return self.exchange.fetch_ohlcv(symbol_pair, timeframe, since, limit)
            else:
                tqdm.write(f"Warning: {self.exchange.id} does not support fetchOHLCV.")
                return None
        except ccxt.NetworkError as e:
            tqdm.write(f"Network error: {e}, retrying in 5 seconds...")
            time.sleep(5)
            return self._fetch_ohlcv_robust(symbol_pair, timeframe, since, limit)
        except ccxt.ExchangeError as e:
            tqdm.write(f"Exchange error: {e} for {symbol_pair}")
            return None
        except Exception as e:
            tqdm.write(f"Unexpected error: {e}")
            return None

    def download_data(self, symbols, timeframes, start_date_str, quote_currency='USDT'):
        # Optimization 1: Load market data once before starting
        try:
            print("Loading all market information from the exchange. This might take some time...")
            self.exchange.load_markets()
            print("Market information loaded successfully.")
        except Exception as e:
            print(f"Failed to load market data: {e}. Script will terminate.")
            return

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        since = int(start_date.timestamp() * 1000)
        total_tasks = len(symbols) * len(timeframes)
        
        with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
            for timeframe in timeframes:
                for symbol in symbols:
                    pbar.update(1)
                    symbol_pair = f'{symbol}/{quote_currency}'
                    pbar.set_description(f"Processing {symbol_pair} [{timeframe}]")

                    # Optimization 2: Resume from breakpoint — skip if file exists
                    filename = f"{symbol}_{timeframe.replace('/', '')}.parquet"
                    filepath = os.path.join(self.data_dir, filename)
                    if os.path.exists(filepath):
                        tqdm.write(f"File {filename} already exists. Skipping.")
                        continue

                    # Optimization 3: Pre-check if the trading pair exists
                    if symbol_pair not in self.exchange.markets:
                        tqdm.write(f"Warning: Trading pair {symbol_pair} not found on {self.exchange.id}. Skipping.")
                        continue
                    
                    all_ohlcv = []
                    current_since = since
                    
                    while True:
                        ohlcv = self._fetch_ohlcv_robust(symbol_pair, timeframe, current_since)
                        
                        if ohlcv is None or len(ohlcv) == 0:
                            break
                        
                        all_ohlcv.extend(ohlcv)
                        last_timestamp = ohlcv[-1][0]
                        current_since = last_timestamp + 1 
                        
                        if last_timestamp > datetime.now().timestamp() * 1000:
                            break
                        
                        # Although ccxt's enableRateLimit manages throttling, a slight pause is still friendly to the server
                        # time.sleep(self.exchange.rateLimit / 1000) # Usually can be commented out

                    if all_ohlcv:
                        self.save_to_file(symbol, timeframe, all_ohlcv)
                        tqdm.write(f"Downloaded and saved {len(all_ohlcv)} entries for {symbol_pair} [{timeframe}].")

    def save_to_file(self, symbol, timeframe, ohlcv_data):
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset='datetime', keep='first').set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']]  # Reorder columns
        
        filename = f"{symbol}_{timeframe.replace('/', '')}.parquet"
        filepath = os.path.join(self.data_dir, filename)
        df.to_parquet(filepath)

# --- Script Entry Point ---
if __name__ == '__main__':
    # --- Step 1: Fetch crypto list ---
    top_symbols_list = get_top_n_crypto_by_market_cap(50)
    print("The number of Crypto =", len(top_symbols_list))
    print("The list of Crypto:\n", top_symbols_list)

    # --- Step 2: Configuration ---
    IS_TEST_RUN = False  # <--- Toggle True (test) / False (production)

    if IS_TEST_RUN:
        print("\n" + "="*50 + "\n!!! WARNING: Running in TEST MODE !!!\n" + "="*50)
        TARGET_SYMBOLS = ['BTC', 'ETH', 'SOL'] 
        TIME_FRAMES = ['1m', '5m', '15m', '1h', '1d'] 
        START_DATE = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        DATA_DIR = 'data_test'
    else:
        print("\n" + "="*50 + "\n>>> WARNING: Running in PRODUCTION MODE — Full Data Download <<<\n" + "="*50)
        TARGET_SYMBOLS = top_symbols_list
        TIME_FRAMES = ['1m', '5m', '15m', '1h', '1d']
        START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        DATA_DIR = 'data_production'

    # --- Step 3: Start download ---
    downloader = CryptoDownloader(exchange_name='binance', data_dir=DATA_DIR)
    downloader.download_data(TARGET_SYMBOLS, TIME_FRAMES, START_DATE, quote_currency='USDT')

    print("\nDownload task completed.")


"""
Example Usage:
--------------
1. Make sure you have all required packages installed:

   pip install ccxt pandas tqdm requests

2. Run the script normally for production mode:

   python crypto_downloader.py

   This will download OHLCV data for the top 50 cryptocurrencies over the past year 
   and save them in the 'data_production' folder.

3. To test the script with a smaller dataset:

   Set IS_TEST_RUN = True in the script, then run:

   python crypto_downloader.py

   It will download only 2 days of data for BTC, ETH, and SOL and save them in 'data_test'.

Output Files:
-------------
The script saves data in .parquet format, named as:

   SYMBOL_TIMEFRAME.parquet

For example:

   BTC_1m.parquet
   ETH_1h.parquet
"""
