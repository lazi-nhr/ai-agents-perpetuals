from __future__ import annotations

import argparse
import time
from pathlib import Path

from defi_ai.core.config import get_paths
from defi_ai.core.state import RuntimeConfig
from defi_ai.execution.signal_writer import write_signal
from defi_ai.features.pair_features import compute_pair_features
from defi_ai.inference.ppo_runner import PPOModelRunner
from defi_ai.inference.signal import SignalConfig, generate_signal
from defi_ai.marketdata.hyperliquid import CandleConfig, HyperliquidClient
from defi_ai.orchestration.control import load_control
from defi_ai.orchestration.models import load_active_model
from defi_ai.orchestration.pairs import load_active_pair
from defi_ai.utils.tracing import traceable


@traceable(name="agent:trading_tick", tags=["live", "trading"])
def trading_tick(
    *,
    client: HyperliquidClient,
    runner: PPOModelRunner,
    runtime: RuntimeConfig,
) -> dict | None:
    paths = get_paths()
    control = load_control(paths.control_file)
    if control.paused:
        return None

    active_pair = None
    alpha = beta = None

    if control.force_pair:
        a1, a2 = control.force_pair
        active_pair = (a1, a2)
    else:
        ap = load_active_pair(paths.pairs_file)
        if ap is None:
            raise RuntimeError(f"No active pair found. Run pair selection to write {paths.pairs_file}")
        active_pair = (ap.asset1, ap.asset2)
        alpha, beta = ap.alpha, ap.beta

    a1, a2 = active_pair

    model = load_active_model(paths.model_pointer_file, runtime.default_model_path)
    model_path = model.path
    if control.force_model_path:
        model_path = Path(control.force_model_path)

    # Fetch candles
    c_cfg = CandleConfig(interval=runtime.interval, limit=runtime.candle_limit)
    df1 = client.fetch_candles(a1, c_cfg)
    df2 = client.fetch_candles(a2, c_cfg)
    if df1.empty or df2.empty:
        raise RuntimeError("Empty candle data received")

    p1 = df1["close"]
    p2 = df2["close"]
    pair_features = compute_pair_features(p1, p2, runtime.lookback, alpha=alpha, beta=beta)

    signal = generate_signal(
        runner=runner,
        model_path=model_path,
        asset1=a1,
        asset2=a2,
        pair_features=pair_features,
        lookback=runtime.lookback,
        notional_usd=runtime.notional_usd,
        cfg=SignalConfig(bar_timeframe=runtime.interval),
    )

    write_signal(paths.signal_file, signal)
    return signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick-seconds", type=int, default=60)
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--notional-usd", type=float, default=100.0)
    parser.add_argument("--candle-limit", type=int, default=250)
    parser.add_argument("--default-model", type=str, default="models/best_model.zip")
    args = parser.parse_args()

    runtime = RuntimeConfig(
        interval=args.interval,
        candle_limit=args.candle_limit,
        lookback=args.lookback,
        notional_usd=args.notional_usd,
        tick_seconds=args.tick_seconds,
        default_model_path=Path(args.default_model),
    )

    client = HyperliquidClient()
    runner = PPOModelRunner(model_path=runtime.default_model_path)

    while True:
        try:
            signal = trading_tick(client=client, runner=runner, runtime=runtime)
            if signal:
                print(f"Signal emitted: {signal['pair']} weights={signal['weights']}")
            else:
                print("Paused by control.json")
            time.sleep(runtime.tick_seconds)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Tick error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
