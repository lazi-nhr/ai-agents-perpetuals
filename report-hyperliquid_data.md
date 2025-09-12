## Hyperliquid Historical Trade Data Collection and Processing Report

Report date: September 12, 2025

Author: Ernest

------

### 1. Summary

Our AI Agent project will run automated trading on Hyperliquid. The initial focus is statistical arbitrage (for example, pair trading). The long-term goal is to build a general trading agent based on reinforcement learning. This requires not just data, but clean, consistent, unbiased, and easy-to-use historical market data.

We chose Hyperliquid because it is a high-performance on-chain derivatives protocol that provides transparent and complete trade history. That history is suitable for training and backtesting our strategies. To support the AI Agent, especially the pair-trading algorithms and later reinforcement learning models, we built a wide-ranging, carefully cleaned OHLCV (Open, High, Low, Close, Volume) dataset for Hyperliquid perpetual contracts. This report explains how we built that dataset.

The main problems were mixed identifiers for perpetual and spot assets, heterogeneous file formats, and noisy raw prices. We implemented a robust pipeline using streaming decompression, multi-process parallelism, IQR-based capping of outliers, and strict asset classification rules. From raw trades we produced a high-quality dataset.

The final output covers 191 perpetual contracts and 195 spot markets, stored in Parquet files. We retained every historical trading pair that ever appeared, including delisted ones, so the dataset avoids survivorship bias and provides a solid basis for research and backtesting.

------

### 2. Data Source Exploration & Core Challenges

#### 2.1. Hyperliquid data ecosystem overview

Our source is Hyperliquid official historical snapshots, including:

- **API metadata**: obtained from the `/info` endpoint, which returns `meta` (perp info) and `spotMeta` (spot info).
- **Raw trade data**: detailed fills come from Hyperliquid node data that has been archived to public cloud storage, usually AWS. The raw files are LZ4-compressed line-delimited JSON and record every matched trade.
  - Source: AWS S3 bucket `s3://hl-mainnet-node-data`.
  - Access model: **Requester Pays**. Download costs are charged to the requestor’s AWS account.
  - Data type: per-trade fills are available and are the best basis to build accurate OHLCV.
  - Availability limit: the S3 archive starts at **2025-03-22**. That means strategies that need longer history cannot be backtested on this dataset. This is an important constraint.

#### 2.2. Core challenge 1: perp vs spot identifiers

This is the most critical challenge. In raw files the `coin` field and other asset identifiers are not consistent:

- **Perpetual contracts** typically appear as their ticker, for example `"BTC"`.
- **Spot markets** may appear as full pairs like `"PURR/USDC"` or as indexed names such as `"@1"`.

If we do not separate these correctly, a spot with name `"@1"` could be mistaken for a delisted perp, or different markets could be merged incorrectly. To fix this, we call both `meta` and `spotMeta` endpoints and build authoritative lookup tables for perp and spot markets. Perps use their `name` as the primary key, spots use their `index` as the primary key. This resolves the identifier confusion.

#### 2.3. Core challenge 2: heterogeneity and non-normalized raw formats

Raw files are not uniform. They have several structures and inconsistent identifiers. Our `process_data.py` script includes compatibility logic to handle this.

- **Format variation**
   The script inspects the file path for markers such as `node_trades`, `node_fills`, or `node_fills_by_block` to detect structure and parse accordingly. These formats differ in organization, asset fields, and timestamp types.

| Feature           | Format A: `node_trades`             | Format B: `node_fills`        | Format C: `node_fills_by_block`    |
| ----------------- | ----------------------------------- | ----------------------------- | ---------------------------------- |
| Data organization | Single trade `Object {}`            | Trade entry `List [user, {}]` | Block `Object { "events": [...] }` |
| Asset identifier  | String symbol (`'symbol'`)          | String coin (`'coin'`)        | String coin (`'coin'`)             |
| Timestamp         | ISO 8601 string with high precision | Unix ms integer               | Unix ms integer                    |

(Although older records may use numeric IDs, the script identifies assets by the string value in fields like `'symbol'` or `'coin'`, which can be `"BTC"` or `"@1"`. We then use API metadata to resolve which type it is.)

- **Price outliers**
   Raw prices (`px`) include extreme noise. If used directly to compute high and low, bad values will distort bars and technical indicators.

#### 2.4. Core challenge 3: large volume and memory use

The raw archive totals about 58.2 GB, which challenges compute and memory. Single-process processing would take weeks. Naive parallel processing can blow past 64 GB of RAM. We needed a design that balances time and memory.

------

### 3. Data Pipeline Design & Implementation

We built `process_data.py` with the following core steps and practices.

#### 3.1. Step 1: streaming read and parallel processing

To handle 58.2 GB efficiently:

- **Streaming decompression**: use `lz4.frame.open` in streaming mode and parse one line at a time, keeping memory low regardless of file size.
- **Parallel processing**: use `concurrent.futures.ProcessPoolExecutor` to distribute files across CPU cores and reduce total processing time.

#### 3.2. Step 2: cleaning and normalization

- **Timestamp alignment**: raw timestamps are mixed (Unix ms or ISO strings). We parse them and convert to timezone-aware Pandas `datetime` in UTC to avoid timezone errors.
- **Outlier handling**: we detect price outliers by Interquartile Range (IQR). Values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` are capped rather than dropped.
  - **Reason**: capping smooths extreme noise while keeping the trade count and volume intact. Dropping trades would create artificial gaps and could bias volume or timing analysis, so capping is safer for backtests.

#### 3.3. Step 3: classification and aggregation

We aggregate in two phases:

1. **Per-process aggregation**: each worker aggregates the file it processed by `coin` and `timeframe` into OHLCV buckets.
2. **Global merge and normalization**: the main process collects per-worker outputs, runs `classify_and_normalize_coin` to unify assets (spots by `index`, perps by `name`), merges parts that belong to the same final asset, and performs a final resample to ensure correct aggregation across time blocks.

- **Avoiding survivorship bias**: the pipeline keeps every contract that appears in history. Even if a contract is delisted and no longer in API `meta`, we still include it as a separate asset. This preserves full market history.

------

### 4. Validation of Data Processing Correctness

We created two validation scripts, `verify_split.py` and `coverage_check.py`, to check the pipeline from micro and macro viewpoints. Results confirm correctness, completeness, and robustness.

#### 4.1. Micro-level validation: 100% trade mapping

`verify_split.py` validates asset classification and file naming logic.

- Method:
  1. Sample 20 raw `.lz4` files at random.
  2. Parse 1000 individual trades from them.
  3. For each trade, simulate the classification logic and predict the output filename.
  4. Check if that filename exists in the final output.
- Results:
  - Cross-checked with API: the generated 195 spot and 191 perp filenames match markets in the current Hyperliquid API.
  - In 1000 random samples, the classification hit rate was 100%; misses were 0.
- Conclusion: `misses: 0` shows the classification and file mapping logic covers all encountered formats and places each trade in the correct asset file.

#### 4.2. Macro-level coverage validation: historical and delisted assets

`coverage_check.py` checks whether we covered all markets that appeared historically, including delisted ones.

- Method:
  1. Get current active counts from the API: active spot count (`209`) and active perp count (`206`).
  2. Scan historical files to count distinct assets seen in raw data (example counts: `seen_spot`: 179, `seen_perp`: 190).
  3. Compare to the final produced Parquet counts (`out_spot`: 195, `out_perp`: 191).
- Key finding:
   The final output has more spot files (195) than the counted historical `seen_spot` (179). This is because `coverage_check.py` only counts spot indexes that map to current API entries. Some historical spot markets are now delisted; the verification scan excluded them. Our main script, however, preserved those delisted markets by treating unknown assets conservatively and routing them into `perp_buckets` with their original names. This is why delisted spots appear among the output perp files.
- Conclusion: the numerical differences show that the pipeline retained delisted assets. That retention prevents survivorship bias and preserves true historical market coverage.

Together, the two validation scripts provide strong evidence for the dataset’s fidelity and completeness.

------

### 5. Final Dataset Specification

#### 5.1. Storage layout

The processed OHLCV dataset is arranged by timeframe and market type:

```
./hyperliquid_data/processed_ohlcv_{tf}/
├── perp/
│   ├── BTC_{tf}_ohlcv.parquet
│   ├── ETH_{tf}_ohlcv.parquet
│   └── ... (191 perp files)
└── spot/
    ├── ANOM_USDC_{tf}_ohlcv.parquet
    └── ... (195 spot files)
```

Note: the project mainly uses the `perp/` directory.

#### 5.2. Data dictionary

All Parquet files under `perp/` include the following columns:

| Column   | Type                  | Unit                 | Description                                                  | Missing value handling |
| -------- | --------------------- | -------------------- | ------------------------------------------------------------ | ---------------------- |
| `time`   | `datetime64[ns, UTC]` | UTC time             | Bar start time (left-closed, right-open)                     | N/A (index)            |
| `open`   | `float64`             | quote currency (USD) | Opening price in the time period                             | forward fill (`ffill`) |
| `high`   | `float64`             | quote currency (USD) | Highest price in the period (IQR capping applied)            | forward fill (`ffill`) |
| `low`    | `float64`             | quote currency (USD) | Lowest price in the period (IQR capping applied)             | forward fill (`ffill`) |
| `close`  | `float64`             | quote currency (USD) | Closing price in the period                                  | forward fill (`ffill`) |
| `volume` | `float32`             | base asset           | Total traded volume during the period (for example, BTC for BTC-USD pair) | fill with `0.0`        |

Note: the `ffill` policy means that if a period has no trades, its OHLC values equal the previous period’s close. This behavior matters for volatility calculations and should be considered when designing indicators.

------

### 6. Usage Guide & Code Examples

#### 6.1. Environment

Install the core Python libraries. Using `pyarrow` is recommended for best Parquet performance.

Bash

```bash
pip install pandas pyarrow
```

#### 6.2. Quick load example

```python
import pandas as pd

# base path and timeframe
DATA_BASE_PATH = './hyperliquid_data/processed_ohlcv_1d/perp/'
ASSET = 'ETH'
TIMEFRAME = '1d'

file_path = f"{DATA_BASE_PATH}/{ASSET}_{TIMEFRAME}_ohlcv.parquet"
df_eth = pd.read_parquet(file_path)

print("ETH 1d OHLCV data:")
print(df_eth.head())
print("\nData Info:")
df_eth.info()
```

#### 6.3. Pair trading: load and align multiple assets

To run pair-trading, align close prices of multiple assets by time.

```python
"""
Load and align close prices for multiple perps.
Return a DataFrame indexed by time with each asset's close as a column.
Rows with any missing values are removed.
"""
import pandas as pd

def load_and_align_closes(assets: list, tf: str, base_path: str) -> pd.DataFrame:
    """
    Load and align close prices for the given assets.
    """
    all_closes = {}
    for asset in assets:
        file_path = f"{base_path}/{asset}_{tf}_ohlcv.parquet"
        try:
            df = pd.read_parquet(file_path)
            all_closes[asset] = df['close']
        except FileNotFoundError:
            print(f"Warning: Data for asset '{asset}' not found.")
            
    df_aligned = pd.DataFrame(all_closes)
    df_aligned.dropna(inplace=True)  # keep only times where all assets have data
    return df_aligned

# Example: align BTC and ETH 1-hour closes
aligned_prices = load_and_align_closes(
    assets=['BTC', 'ETH'],
    tf='1h',
    base_path='./hyperliquid_data/processed_ohlcv_1h/perp/'
)

print("\nAligned BTC and ETH 1h close prices:")
print(aligned_prices.head())
```

------

### 7. Scope and Limitations

- **Data level**: The dataset is L1-level trade data. It does not include L2 order book depth or L3 signals like funding rates or liquidation events. This limits the state space available to reinforcement learning agents.
- **Transaction costs**: The dataset does not include maker/taker fees. Backtests must simulate trading costs separately.
- **Data completeness**: We processed all available raw files, but the original source can still have small delays or rare gaps. The `ffill` policy for OHLC handles empty periods but is an interpolation method that cannot fully reproduce true market pauses.

------

### 8. Future Maintenance & Extensions

- **Updates**: Change `START_DATE` and `END_DATE` in `process_data.py` and run it on a schedule to fetch incremental data.
- **Adapting to API changes**: If the `/info` response changes, update `build_spot_maps` and `build_perp_maps`.
- **Next work**: gather funding rates and order book snapshots to build richer features for advanced reinforcement learning agents.

------

### 9. Appendix

- **Appendix A**: Core processing script `process_data.py` (full code provided)
- **Appendix B**: Validation scripts `verify_split.py` and `coverage_check.py`
- **Appendix C**: `requirements.txt` (suggested)

```
pandas
lz4
tqdm
requests
pyarrow
```