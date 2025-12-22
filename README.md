# DeFi-AI (Production-ready LangSmith automation checkpoint)

This repository is a production-oriented checkpoint for an autonomous trading agent.

## Key components

- Pair selection: correlation filtering + Engle–Granger cointegration (OLS + ADF) over a configurable universe.
- Trading tick: Hyperliquid candle polling → pair feature engineering → PPO inference → atomic `signal.json`.
- Execution: Hummingbot watches `signal.json` and executes orders when it changes.
- LangSmith: trace boundaries and metadata are emitted to enable automation rules (pause, switch pair, retrain).

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment (optional but recommended):

```bash
cp .env.example .env
```

3. Run pair selection (recommended every 6–24h):

```bash
python scripts/run_pair_selection.py --top-k 5
```

4. Run the live trading tick loop:

```bash
python scripts/run_live_loop.py --tick-seconds 60
```

5. (Optional) Run the automation webhook server used by LangSmith automations:

```bash
python scripts/run_webhook_server.py --host 0.0.0.0 --port 8080
```

## Files written

- `artifacts/pairs/active_pairs.json`: ranked list of the current best cointegrated pairs.
- `signal.json`: atomic signal output consumed by Hummingbot.
- `artifacts/control/control.json`: runtime control overlay (pause, force pair, switch model).

