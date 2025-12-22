from __future__ import annotations

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

def zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + 1e-8)

def kalman_smooth(series: pd.Series) -> pd.Series:
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01,
    )
    means, _ = kf.filter(series.values.astype(float))
    return pd.Series(means.flatten(), index=series.index)

def compute_pair_features(
    p1: pd.Series,
    p2: pd.Series,
    lookback: int,
    alpha: float | None = None,
    beta: float | None = None,
) -> dict[str, list[float]]:
    """Compute pair features required by the PPO policy.

    Returns a dict of lists of length `lookback` for:
    ['alpha','beta','corr','pval','spreadNorm','spreadNormKalman','spreadNormMa','spreadNormVol'].
    """
    idx = p1.index.intersection(p2.index)
    p1 = p1.loc[idx]
    p2 = p2.loc[idx]

    # If alpha/beta not provided, estimate rolling (simple hedge ratio proxy)
    if beta is None or alpha is None:
        cov = p1.rolling(lookback).cov(p2)
        var = p2.rolling(lookback).var()
        beta_s = (cov / var).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        alpha_s = p1.rolling(lookback).mean() - beta_s * p2.rolling(lookback).mean()
    else:
        beta_s = pd.Series(beta, index=idx)
        alpha_s = pd.Series(alpha, index=idx)

    spread = p1 - (beta_s * p2)
    spread_norm = zscore(spread, lookback)

    # pval is optional in current policy; keep placeholder zeros for compatibility
    pval = pd.Series(0.0, index=idx)

    feats = {
        "alpha": alpha_s,
        "beta": beta_s,
        "corr": p1.rolling(lookback).corr(p2),
        "pval": pval,
        "spreadNorm": spread_norm,
        "spreadNormKalman": zscore(kalman_smooth(spread), lookback),
        "spreadNormMa": zscore(spread.rolling(max(2, lookback // 2)).mean(), lookback),
        "spreadNormVol": zscore(spread.rolling(max(2, lookback // 2)).std(), lookback),
    }

    # last lookback values to list
    out: dict[str, list[float]] = {}
    for k, s in feats.items():
        tail = s.iloc[-lookback:].fillna(0.0).astype(float).tolist()
        if len(tail) != lookback:
            # pad left if insufficient
            tail = ([0.0] * (lookback - len(tail))) + tail
        out[k] = tail
    return out
