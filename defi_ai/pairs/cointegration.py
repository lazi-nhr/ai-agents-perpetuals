from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

@dataclass(frozen=True)
class PairScore:
    asset1: str
    asset2: str
    score: float
    corr: float
    p_value: float
    alpha: float
    beta: float

def check_cointegration(p1: pd.Series, p2: pd.Series, *, corr_threshold: float = 0.85, pvalue_threshold: float = 0.05) -> PairScore | None:
    corr = float(p1.corr(p2))
    if abs(corr) < corr_threshold:
        return None

    X = sm.add_constant(p2)
    model = sm.OLS(p1, X).fit()
    alpha = float(model.params.iloc[0])
    beta = float(model.params.iloc[1])

    residuals = p1 - (alpha + beta * p2)
    p_value = float(adfuller(residuals)[1])
    if p_value >= pvalue_threshold:
        return None

    return PairScore(
        asset1="",
        asset2="",
        score=abs(corr),
        corr=corr,
        p_value=p_value,
        alpha=alpha,
        beta=beta,
    )

def rank_pairs(
    price_series: dict[str, pd.Series],
    *,
    min_points: int = 120,
    corr_threshold: float = 0.85,
    pvalue_threshold: float = 0.05,
    top_k: int = 5,
) -> list[PairScore]:
    assets = list(price_series.keys())
    scored: list[PairScore] = []
    for a1, a2 in combinations(assets, 2):
        s1 = price_series[a1]
        s2 = price_series[a2]
        idx = s1.index.intersection(s2.index)
        if len(idx) < min_points:
            continue
        p1 = s1.loc[idx].astype(float)
        p2 = s2.loc[idx].astype(float)

        ps = check_cointegration(p1, p2, corr_threshold=corr_threshold, pvalue_threshold=pvalue_threshold)
        if ps is None:
            continue
        scored.append(PairScore(asset1=a1, asset2=a2, score=ps.score, corr=ps.corr, p_value=ps.p_value, alpha=ps.alpha, beta=ps.beta))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]
