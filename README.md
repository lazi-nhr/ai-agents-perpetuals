# AI Agents in Perpetual Futures Markets

This repository contains the codebase accompanying the project **“AI Agents in Perpetual Futures Markets”**, which investigates reinforcement learning–based statistical arbitrage strategies in cryptocurrency perpetual futures and demonstrates a proof of concept for end-to-end automated deployment using AI agents.

The repository is organized around **two main pipelines**:

1. **Reinforcement Learning Training & Evaluation**
2. **AI Agent Deployment & Orchestration**

---

## 1. Reinforcement Learning Training Pipeline

This pipeline implements the offline research workflow used to train, evaluate, and analyze reinforcement learning (RL) trading strategies under realistic market conditions.

### Overview

The training pipeline follows a structured sequence:

1. Data Acquisition  
2. Data Preprocessing & Feature Engineering  
3. Reinforcement Learning Training  
4. Out-of-Sample Evaluation  

All experiments explicitly incorporate transaction costs and realistic execution assumptions.

---

### 1.1 Data Acquisition

- Historical market data is sourced from **Binance USDⓈ-M Perpetual Futures** via the CCXT API.
- Data includes 1-minute OHLCV candles and funding rates.
- The dataset spans **May 2024 – April 2025**.
- Asset selection is restricted to instruments tradable on both Binance and Hyperliquid to ensure deployability.

---

### 1.2 Data Preprocessing & Feature Engineering

Raw market data is transformed into a consistent state representation suitable for reinforcement learning:

- Time alignment to a fixed 1-minute grid  
- Forward-filling and removal of missing values  
- Rolling-window cointegration testing (Engle–Granger)  
- Construction of spread-based features (z-scores, volatility, Kalman-filtered signals)  
- Asset-level technical indicators (RSI, MACD, momentum, volatility)  
- Funding-rate features specific to perpetual futures  
- Portfolio state variables (current exposure and cash)  

The resulting feature vectors form the state space of the RL environment.

---

### 1.3 Reinforcement Learning Training

- The trading problem is formulated as a **Markov Decision Process (MDP)**.
- A **Proximal Policy Optimization (PPO)** agent is trained in a continuous action space.
- Actions correspond to the signed exposure to a cointegrated asset spread.
- The reward function is based on **quadratic utility**, balancing returns and risk while explicitly accounting for transaction costs.
- Training is performed offline over multiple million timesteps using Stable-Baselines3.

---

### 1.4 Evaluation & Results

- Models are evaluated **out-of-sample** on a held-out test set.
- Performance is compared across multiple transaction fee regimes:
  - No fees (frictionless benchmark)
  - Low fees (1.44 bps)
  - High fees (4.50 bps)
- Metrics include cumulative return, volatility, Sharpe ratio, drawdown, and action statistics.
- Results show that transaction costs are a key limiting factor for high-frequency RL-based statistical arbitrage.

---

## 2. AI Agent Deployment Pipeline

The second pipeline implements a **proof of concept AI agent** for coordinating model inference, market data ingestion, and live execution.

This pipeline focuses on **orchestration and automation**, not on improving trading performance.

---

### 2.1 Agent Architecture

The system is split into two decoupled workflows:

#### Slow-Tick Workflow (Structural Decisions)
- Periodic selection of tradable asset pairs  
- Cointegration and correlation filtering  
- Output written to explicit artifacts (`active_pairs.json`)  

#### Fast-Tick Workflow (Inference & Signals)
- Live market data ingestion  
- Feature reconstruction consistent with training  
- Inference using a pretrained PPO policy  
- Risk checks and position constraints  
- Output written to `signal.json`  

All coordination occurs via explicit, versioned artifacts.

---

### 2.2 Execution Layer

- Order execution is handled externally by **Hummingbot**
- Hummingbot monitors `signal.json` and translates signals into market orders
- Execution logic is isolated from model inference and orchestration
- The system is containerized using Docker for reproducibility

---

### 2.3 Scope and Limitations

- The AI agent is intended as a **systems-level proof of concept**
- No online fine-tuning, continual learning, or automated retraining is implemented
- LangGraph and LangSmith provide orchestration and observability only
- Economic constraints (fees, liquidity, latency) remain binding regardless of automation

---

## Reproducibility

- All experiments are configured via versioned configuration files
- Dependencies are fixed and documented
- Evaluation artifacts and logs are retained for inspection
- The codebase is designed to support systematic experimentation and extension

---

## Disclaimer

This repository is provided **for research and educational purposes only**.  
It does not constitute financial advice, and no component is intended for production trading without further validation and risk assessment.