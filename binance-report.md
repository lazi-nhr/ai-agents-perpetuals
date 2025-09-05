# Biance Historical Market Data of the Top 50 Cryptocurrencies

- **Report Version**: V1.0

- **Author**: Ernest

- **Date**: August 25, 2025


---

## **1. Summary**

This report details the end-to-end process of acquiring, cleaning, and aligning historical market data for the top 50 cryptocurrencies over the past year (August 2024 - August 2025) for our quantitative trading project. We successfully built an automated data acquisition pipeline to download complete OHLCV data for **31** major crypto assets from the leading exchange, **Binance**, at **zero cost**. The data was collected across five granularities: **1m, 5m, 15m, 1h, and 1d**.

Subsequently, we designed and executed a rigorous data cleaning and alignment process. This process focused on addressing key challenges such as **insufficient historical data for new coins, synchronization of timestamps across assets, and filling temporary liquidity gaps**. The final output consists of **five distinct Parquet data files, ready for direct use in multivariate quantitative analysis**. Each file corresponds to a specific timeframe and contains **30** assets with complete, comparable histories.

This report strongly recommends that the algorithm team **prioritize the `1h` and `15m` aligned datasets for initial research**, as they offer the best balance between signal clarity and the frequency of trading opportunities. Furthermore, users must be aware that this dataset is based on spot `USDT` quotes from a single exchange, does not include order book depth or tick-level information, and that the asset selection is subject to survivorship bias.

**Core Deliverables**:

- `aligned_data_1m_close.parquet`
  
- `aligned_data_5m_close.parquet`
  
- `aligned_data_15m_close.parquet`
  
- `aligned_data_1h_close.parquet`
  
- `aligned_data_1d_close.parquet`
  

---

## **2. Introduction & Project Objectives**

This project aims to develop quantitative strategies centered on pairs trading and other forms of statistical arbitrage. The cornerstone of any quantitative strategy is high-quality, high-frequency, long-term historical data. The primary objective of this data engineering effort is to provide the strategy research team with a clean, well-structured, and reliable data foundation, meeting the following requirements:

- **Asset Scope**: Top 50 cryptocurrencies by market capitalization.
  
- **Time Horizon**: The most recent year.
  
- **Data Granularity**: 1m, 5m, 15m, 1d (with 1h added subsequently).
  
- **Cost Constraint**: Zero or minimal cost.
  
- **Core Requirement**: Data must be rigorously aligned to support synchronous, cross-asset analysis.
  

---

## **3. Data Acquisition Pipeline**

We designed and implemented an automated data acquisition pipeline using Python.

### **3.1 Data Source Selection & Rationale**

- **Source for Cryptocurrency List**: We utilized the **CoinGecko API** to retrieve the list of the current top 50 cryptocurrencies. This choice was based on its generous free API rate limits, authoritative data, and its status as a recognized industry standard.
  
- **Source for Historical Market Data**: We selected the public API of the **Binance** exchange as our sole data source for the following reasons:
  
    - **Liquidity and Data Quality**: As the world's largest exchange by trading volume, Binance's data is highly representative, comprehensive, and of high quality.
      
    - **Cost**: The API is free to use, meeting our project's cost constraints.
      
    - **Data Consistency**: **Using a single data source was a core principle of this project.** This approach completely avoids the noise introduced by inconsistencies in price, timestamps, and data cleaning rules that can arise when using multiple exchanges, which is critical for spread-sensitive strategies.
      

### **3.2 Technical Implementation**

- **Core Tools**: The pipeline was developed in `Python`, leveraging the powerful `CCXT` library. `CCXT` unifies the API interfaces for hundreds of exchanges and has built-in handling for rate limiting and pagination, which significantly improved development efficiency and script stability.
  
- **Execution Process**: The script runs automatically, iterating through each asset in the top 50 list and each specified timeframe. It pieces together the full year of historical data through looped API requests and saves the output locally in the efficient `Parquet` columnar storage format.
  

### **3.3 Execution Summary**

- **Successfully Acquired Assets**: From the CoinGecko top 50 list, we successfully retrieved data for the USDT spot trading pairs of **31** assets from Binance.
  
- **Data Completeness**: For these 31 assets, we obtained the complete OHLCV data from `2024-08-25` to `2025-08-25`.
  

---

## **4. Raw Data Overview & Initial Findings**

Before cleaning, we conducted a preliminary analysis of the raw data.

### **4.1 Asset Coverage Analysis**

Data for approximately 19 assets could not be acquired, primarily because **these assets do not have a spot USDT trading pair on Binance**. These can be categorized as:

- **Competing exchange tokens**: e.g., `OKB`, `BGB`, `LEO`.
  
- **Native DeFi tokens**: e.g., `STETH`, `WSTETH`, `WEETH`, whose primary liquidity is on-chain.
  
- **Other stablecoins or wrapped assets**: e.g., `USDE`, `BSC-USD`.
  
- _For a complete list of the finally acquired assets, please refer to Appendix A._
  

### **4.2 Data Volume & Granularity**

In total, we generated `31 * 5 = 155` raw data files. The approximate number of rows per file is as follows:

- **1d**: ~366 rows
  
- **1h**: ~8,784 rows
  
- **15m**: ~35,126 rows
  
- **5m**: ~105,377 rows
  
- **1m**: ~526,870 rows
  

### **4.3 Known Issues with Raw Data**

- **Incomplete Historical Period**: We noted that some coins (e.g., `ONDO`) were listed within the past year and thus have less than one year of historical data. This was addressed during the cleaning phase.
  

---

## **5. The ETL Process (Data Cleaning & Alignment)**

This is the most critical stage, transforming the raw data into a ready-to-use dataset. Our process (`clean_data_V2.py`) strictly adheres to the following steps:

### **5.1 The Need for Cleaning & Alignment**

Individual raw data files are like separate race recordings for each athlete. For multivariate strategies like pairs trading, we need to bring all these recordings into a multi-track editor and synchronize them to a single, absolute timeline for comparison. This process achieves exactly that.

### **5.2 Detailed Methodology**

1. **Load Data**: For each timeframe, batch-load all 31 corresponding asset Parquet files.
   
2. **Timezone Unification**: To prevent any timestamp ambiguity, the time index of all data is standardized and converted to the **UTC timezone**.
   
3. **Cross-Asset Alignment**: All asset price series (`close` price) are merged into a single `Pandas DataFrame`. This DataFrame is built on a **perfect, uninterrupted time index**. If an asset has no data at a specific timestamp (e.g., it had not been listed yet), its value at that point is set to `NaN` (Not a Number).
   
4. **Handling Missing Data**: We employ a **forward-fill (`ffill`)** method to handle `NaN` values caused by brief periods of market illiquidity.
   
    - **Rationale**: This assumes that if no new trades occur within a candle's interval, the price remains consistent with the previous interval.
      
    - **Risk Control**: We set a `limit=5` parameter, meaning we will not fill gaps of more than five consecutive periods. This prevents the erroneous infilling of extended periods of missing data, such as those caused by exchange downtime.
    
5. **Asset Pool Filtering**: To ensure all assets used in the analysis have a comparable and sufficiently long shared history, we perform the following filtering:
   
    - **Rule**: Remove any asset with less than **80%** of available data points over the full one-year window.
      
    - **Result**: `ONDO` was removed based on this rule across all timeframes. Consequently, our final asset pool for analysis is fixed at **30** highly active cryptocurrencies.
      

### **5.3 Final Data Product**

- The process outputs one `aligned_data_{timeframe}_close.parquet` file for each timeframe.
  
- The data within these files is a matrix where the **index consists of unified UTC timestamps** and the **columns represent the close prices of the 30 assets**.
  
- **Important**: These files still contain `NaN` values, as we have preserved the complete individual history for each asset, only removing `ONDO` for its overall short history. Downstream algorithms, when analyzing a specific pair (e.g., `BTC` and `ETH`), should **dynamically extract their common time window and remove any `NaN`s within that specific window**.
  

---

## **6. Data: Strengths, Limitations & Recommendations**

### **6.1 Strengths**

1. **Single Source of Truth**: All data comes from Binance, ensuring absolute consistency in price and time, making it ideal for researching spread-based strategies.
   
2. **Transparent Process**: This report documents every processing step, giving users full insight into the data's provenance.
   
3. **Standardized Format**: The use of standard Parquet format and `DataFrame` structure allows for direct integration with algorithmic trading systems.
   

### **6.2 Limitations & Potential Risks**

1. **Survivorship Bias**: We selected the **current** top 50 list. This list excludes coins that were delisted or experienced a significant drop in market cap over the past year. This could lead to overly optimistic backtest results, potentially overestimating a strategy's performance in a bear market.
   
2. **No Microstructure Data**: This dataset does not include **tick-level** trade data or **order book depth**. Therefore, it cannot be used for backtesting high-frequency arbitrage strategies, and all backtests must incorporate a **slippage model** to simulate real-world execution impact.
   
3. **Synthetic Cross-Rates**: All non-`USDT` cross-rates (e.g., `ETH/BTC`) are synthetically calculated from their `USDT` prices. This ignores the independent liquidity and bid-ask spread of the true `ETH/BTC` trading pair.
   

### **6.3 Recommendations for Algorithm Developers**

1. **Prioritize Mid-Frequency Data**: We recommend starting your pairs trading and statistical arbitrage research with the `1h` and `15m` datasets to balance signal quality with transaction costs.
   
2. **Handle NaN Values Dynamically**: When analyzing any asset pair, be sure to first select the two columns of data and then use a method like `.dropna()` to find their common historical range before performing calculations.
   
3. **Costs and Slippage**: All strategy backtests **must** include a transaction cost model of at least `0.1%` (for the round trip) and a reasonable slippage model. Otherwise, the backtest results will be meaningless.
   
4. **Use a Sanity Check**: The `(BTC, WBTC)` pair serves as a "ground truth" or sanity check in our dataset. You can use it to validate the correctness of your algorithm.
   

---

## **7. Conclusion & Future Work**

### **7.1 Conclusion**

This phase of the project has successfully established a solid and reliable data foundation. We have not only acquired the necessary data but, more importantly, have built a transparent and reproducible data processing pipeline, clearly articulating the dataset's characteristics, strengths, and limitations.