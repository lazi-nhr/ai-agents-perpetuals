# RL DeFi statistical arbitrage agent

Train and evaluate a reinforcement learning (RL) agent (PPO) that allocates continuous portfolio weights across crypto assets (BTC & ETH by default), with cash as an optional asset. The environment rebalances every bar, enforces long-only, no leverage, and applies transaction costs. It supports walk-forward evaluation and produces benchmark comparisons and metrics.

---

## TL;DR

- **Data**: OHLCV parquet files on Google Drive.
- **Sampling**: Default `1h`. Forward-fills gaps, aligns timezones (CET ➜ UTC).
- **Env**: Continuous weight actions, optional cash, softmax projection, log-return-minus-risk reward.
- **Algo**: PPO (`stable-baselines3`) with configurable hyperparameters.
- **Evaluation**: Walk-forward splits, equity curve plots, metrics, and benchmark baselines (`EW`, `B&H BTC`, `B&H ETH`).
- **Outputs**: Saved models (`./models`), TensorBoard logs (`./tb`), and CSV/JSON reports (`./reports`).

---

## Project Structure

This notebook/script does everything end-to-end:

1. **Config** — Centralized `CONFIG` dict (data, env, splits, RL, eval, IO).
2. **Setup** — Installs dependencies; sets seeds; defines utilities (I/O, annualization, plotting).
3. **Data** — Mirrors Google Drive folder, locates parquet files, cleans & aligns indexes/timezones.
4. **Features** — Returns, rolling volatility, RSI, volume change; optional rolling z-score scaler.
5. **Tensors** — Builds `(assets, features, lookback)` state tensors, aligned against next-step returns/vol.
6. **Splits** — Static or walk-forward masks built over the common timestamp index.
7. **Env** — `PortfolioWeightsEnv` implementing action projection, costs, reward, and step dynamics.
8. **Training** — PPO over `train` mask; saved best via `EvalCallback`.
9. **Backtest** — Deterministic policy rollout on `test` mask; compute metrics and plots.
10. **Reporting** — Aggregate metrics into a DataFrame and save CSV/JSON.