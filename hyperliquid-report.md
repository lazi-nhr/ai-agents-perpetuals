### Acquisition, Processing, and Analysis of Hyperliquid Perpetual Futures Historical Trade Data

**Version:** 1.0 

**Date:** August 25, 2025

**Author:** Ernest

**Data Time Range:** March 22, 2025 - August 22, 2025

#### **1. Summary**

This report details the end-to-end construction of a data processing pipeline for Hyperliquid exchange's historical trade data, built to support the project's algorithmic development and strategy backtesting needs. The project successfully processed approximately 58.2 GB of raw data from March 2025 to the present. The process overcame several significant challenges, including the structural evolution of the data source across three distinct historical periods, inconsistent formatting of key fields (asset identifiers, timestamps), and the prevalence of "truncated series" where many assets did not exist at the beginning of the data range.

The final output is a unified, clean dataset stored in the high-performance **Parquet format**. It consists of two parts: 
1) Independent OHLCV time series for each asset, aggregated into standard intervals (1m, 5m, 15m, 1h, 1d) with **trading gaps filled**; and 
2) A strictly time-aligned data panel designed for multi-asset strategies (e.g., pairs trading) that **preserves information about data unavailability**. This report will elaborate on the processing workflow, key decisions and their trade-offs, and provide core recommendations for end-users of the dataset.

#### **2. Data Source Overview**

- **Location**: Amazon S3 bucket `s3://hl-mainnet-node-data`.
  
- **Nature of Data**:
  
    - **Public Data**: Accessible without special authorization.
      
    - **Requester Pays**: Egress costs for data downloads are billed to the data user's (our) AWS account. All API requests must include the corresponding declaration.
    
- **Data Type**: Provides the most granular **tick-by-tick trade data (Fills/Trades)**, which is the ideal foundation for constructing accurate OHLCV bars.
  
- **Available Time Range**: A comprehensive exploration confirmed that the earliest publicly available trade data from this source **begins on March 22, 2025**. This is a critical limitation for analysis.
  

#### **3. Data Acquisition Process**

A custom download script was developed using Python and the `boto3` library with the following workflow:

1. **Authentication**: Configured local environment with AWS IAM user credentials possessing S3 read permissions.
   
2. **Parallel Downloading**: The script utilizes a `concurrent.futures.ThreadPoolExecutor` (16 threads) to implement multi-threaded parallel downloads, significantly improving efficiency when dealing with a massive number of small files.
   
3. **Incremental Updates**: The script is idempotent, automatically skipping local files that already exist. This facilitates daily incremental updates and allows resuming interrupted downloads.
   

#### **4. Raw Data Challenges & Exploration**

The raw data was not in a single, consistent format. It underwent several structural changes, and key fields exhibited various inconsistencies. This was the primary challenge of this project.

- 4.1 Data Format Evolution:
  
    We identified three distinct historical paths where trade data was stored, each with different structures and key fields.
    

|                       |                                                  |                                                 |                                                      |
| --------------------- | ------------------------------------------------ | ----------------------------------------------- | ---------------------------------------------------- |
| **Feature**           | **Format 1: node_trades (approx. Mar-May 2025)** | **Format 2: node_fills (approx. May-Jul 2025)** | **Format 3: node_fills_by_block (Jul 2025-Present)** |
| **Data Organization** | Single trade `Object {}`                         | Single fill `List [user, {}]`                   | Block `Object { "events": [...] }`                   |
| **Asset Identifier**  | **String Ticker** (`'BTC'`)                      | **Numeric ID** (`'0'`)                          | **Numeric ID** (`0`)                                 |
| **Timestamp**         | **High-precision String**                        | **Millisecond Integer**                         | **Millisecond Integer**                              |

- **4.2 Key Field Inconsistencies**:
  
    - **Asset Identifiers**: Existed in three forms: string names (`'BTC'`), numeric IDs (`1`), and numeric IDs with an `@` prefix (`'@1'`).
      
    - **Asset Identifier Key Name**: Early data used the key `'symbol'`, while later data used `'coin'`.
      
    - **Timestamp Format**: Early data used high-precision ISO 8601 strings, while later data used numeric Unix timestamps in milliseconds. **Failure to correctly handle both formats was the most insidious bug leading to data loss**.
    
- 4.3 Data Coverage (Truncated Time Series):
  
    Although the data files are continuous from March 2025, a large number of assets did not exist at the start of this period. Trade data for most assets only begins to appear around May 26, 2025. This "maturity" issue must be considered when performing cross-sectional analysis across multiple assets.
    

#### **5. Data Processing & Cleaning Pipeline**

We designed a two-stage automated processing pipeline composed of two core scripts.

- 5.1 Stage One: process_data.py - From Raw Data to Independent OHLCV
  
    The goal of this stage was to transform all the messy raw data into clean, independent OHLCV files, separated by asset and time frame.
    
    1. **Parallel Initial Aggregation**: Used a `ProcessPoolExecutor` to distribute thousands of `.lz4` files across all CPU cores. Worker processes determined the data format from the file path, applied the corresponding parsing logic, and performed an initial in-memory aggregation to OHLCV, achieving high parallelization for this compute-intensive task.
       
    2. **Robustness Fixes and Unification**:
       
        - **Asset Identity Unification**: Using an asset mapping table fetched from the official API, all forms of asset IDs (`'BTC'`, `1`, `'@1'`) were unified into standard asset names (`'BTC'`).
          
        - **Column Name Compatibility**: Automatically detected and unified the column name for the asset identifier (`'symbol'` -> `'coin'`).
          
        - **Timestamp Compatibility**: Intelligently detected the data type of the `time` column and applied different parsing methods for numeric and string formats, resolving the critical data loss bug.
        
    3. **OHLCV Aggregation & [Key Decision] Null Value Filling**:
       
        - Used `pandas.resample` to aggregate tick data into standard OHLCV time frames.
          
        - `resample` creates `NaN` (null) values for intervals with no trades. We filled these nulls as follows:
          
            - **Price (O,H,L,C)**: Filled using **`ffill()` (forward fill)**.
              
            - **Volume**: Filled with `0`.
            
        - **Trade-off Analysis**:
          
            - **Pros**: Ensures that the output file for each asset is **temporally continuous**, making it easy to plot directly or calculate technical indicators.
              
            - **Cons**: This method assumes the price remains equal to the previous period's close during no-trade intervals, which may mask periods of illiquidity. High-frequency strategies must be aware of this assumption.
    
- 5.2 Stage Two: align_data.py - From Independent OHLCV to Aligned Data Panel
  
    The goal of this stage was to create a strictly time-aligned dataset for cross-sectional strategies like pairs trading.
    
    1. **Master Time Index Creation**: The script automatically scanned the files generated in the previous stage to determine a global, uninterrupted "master time index" for each time frame.
       
    2. **Alignment & [Key Decision] `NaN` Preservation**:
       
        - `reindex`ed each asset's data to the master time index.
          
        - Unlike the previous stage, this script **did not** perform a `fillna` operation after alignment.
          
        - **Trade-off Analysis**:
          
            - **Pros**: The final aligned data panel **accurately reflects the availability of each asset at every point in time**. Periods where an asset was not yet listed or had missing data are explicitly shown as `NaN`, giving downstream strategies maximum flexibility to handle them as needed.
              
            - **Cons**: Strategy code using this data directly must include logic to handle `NaN` values.
              

#### **6. Final Dataset Deliverables**

- **Storage Format**: **Parquet**.
  
    - **Advantages (Pros)**: Smaller file sizes, extremely fast read/write speeds compared to CSV, and native support for data types and compression. It is the ideal choice for analysis with Python/Pandas.
      
    - **Disadvantages (Cons)**: Binary format, cannot be viewed directly with a text editor.
    
- **Deliverable 1: Independent OHLCV Files**
  
    - **Location**: `./hyperliquid_data/processed_ohlcv/`
      
    - **Naming**: `[ASSET_TICKER]_[TIMEFRAME]_ohlcv.parquet` (e.g., `BTC_1h_ohlcv.parquet`)
      
    - **Schema**:
      
        - **Index**: `time` (UTC timestamp, `pandas.DatetimeIndex`)
          
        - **Columns**: `open`, `high`, `low`, `close`, `volume` (all `float64`)
    
- **Deliverable 2: Aligned Data Panel**
  
    - **Location**: `./hyperliquid_data/alignment_ohlcv/`
      
    - **Naming**: `aligned_[TIMEFRAME].parquet` (e.g., `aligned_1h.parquet`)
      
    - **Schema**:
      
        - **Index**: `time` (UTC timestamp, `pandas.DatetimeIndex`)
          
        - **Columns**: Pandas MultiIndex `(Asset Name, OHLCV Field)`
          

#### **7. Usage Guide & Recommendations**

- **Data Loading**: `pandas.read_parquet()` is recommended. For the aligned data panel, the `.xs()` method is convenient for extracting data.
  
    ```
    import pandas as pd
    df_panel = pd.read_parquet('./hyperliquid_data/alignment_ohlcv/aligned_1h.parquet')
    # Extract all close prices
    close_prices = df_panel.xs('close', level=1, axis=1)
    ```
    
- **Core Recommendations**:
  
    1. **Time Range Limitation**: All analysis and backtesting must be based on data **after March 22, 2025**.
       
    2. **Prioritize Mid-Frequency Data**: For most strategy research, it is advisable to start with the **`1h`** or **`15m`** aligned data to balance signal-to-noise ratio and computational efficiency.
       
    3. **Address the "Maturity" Issue**: Before any cross-sectional comparison, it is **strongly recommended to filter for "mature assets"**—those with sufficiently high data coverage (e.g., > 95%) during your analysis period—to avoid biases from truncated series.
       
    4. **Handle `NaN` as Needed**: When using the aligned data panel, `NaN` values should be handled according to your strategy's logic. For example, dynamically finding the common historical window for a trading pair (`pair_df.dropna()`) is excellent practice.
       

#### **8. Risks & Future Maintenance**

This processing pipeline is highly dependent on Hyperliquid's current data structure. **If Hyperliquid changes its S3 directory structure or internal data format again in the future, the parsing logic described in Section 4 of this report may fail**. At that point, the exploration and analysis steps will need to be repeated to adapt to the new format.

#### **9. Appendix**

- A. Data Processing & Cleaning Script (`process_data.py`)
  
- B. Data Alignment Script (`align_data.py`)
  
- C. Sample Hyperliquid Asset ID Map (`hyperliquid_asset_map.csv`)
  
- D. Example Analysis Script (`run_integrated_pairs_trader.py`)