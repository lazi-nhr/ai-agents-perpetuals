"""
Utility functions and classes for reinforcement learning in statistical arbitrage.
Extracted from rl_stat_arb_ppo.ipynb for modular usage.
"""

import os
import re
import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Set, Tuple


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_csv_to_df(
    path: str,
    sep: str = ",",
    timestamp_index_col: str | None = "datetime",
    encoding: str = "utf-8-sig",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Filesystem path to the CSV.
    parse_timestamp_col : str | None
        If provided and present in the CSV, this column will be parsed to datetime.
        Set to None to skip datetime parsing.
    **read_csv_kwargs :
        Extra arguments passed to `pd.read_csv` (e.g., sep, dtype, usecols).

    Returns
    -------
    pd.DataFrame
    """

    # Parse header-only to check for timestamp col presence
    head = pd.read_csv(path, sep=sep, encoding=encoding, nrows=0)
    if timestamp_index_col and timestamp_index_col in head.columns:
        read_csv_kwargs = {
            **read_csv_kwargs,
            "parse_dates": [timestamp_index_col],
        }

    df = pd.read_csv(path, sep=sep, encoding=encoding, engine="pyarrow", **read_csv_kwargs)

    df = df.set_index("datetime")

    return df


def identify_assets_features_pairs(
    df: pd.DataFrame,
    single_asset_format: str,
    pair_feature_format: str,
) -> tuple[list[str], list[str], list[str], list[tuple[str, str]]]:
    """
    Returns distinct
      1. assets
      2. single-asset feature names (ARB_closeUpperShadow → closeUpperShadow)
      3. pair feature names (ARB_ETH_spreadNorm → spreadNorm)
      4. unordered asset pairs
    """

    def format_to_regex(fmt: str) -> re.Pattern:
        escaped = re.escape(fmt)

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            char_class = r"[A-Za-z0-9_]+" if "FEATURE" in name.upper() else r"[A-Za-z0-9]+"
            return f"(?P<{name}>{char_class})"

        escaped = re.sub(r"\\\{(\w+)\\\}", repl, escaped)
        return re.compile(f"^{escaped}$")

    single_asset_pattern = format_to_regex(single_asset_format)
    pair_feature_pattern = format_to_regex(pair_feature_format)
    generic_single_pattern = re.compile(r"^(?P<ASSET>[A-Za-z0-9]+)_(?P<FEATURE>[A-Za-z0-9_]+)$")

    assets: Set[str] = set()
    single_features: Set[str] = set()
    pair_features: Set[str] = set()
    pairs: Set[Tuple[str, str]] = set()

    literal_feature = None
    if "{FEATURE}" not in single_asset_format:
        literal_feature = single_asset_format.replace("{ASSET}", "").lstrip("_")

    skip_cols = {"timestamp", "datetime", "date"}

    for col in df.columns:
        if col in skip_cols:
            continue

        match_pair = pair_feature_pattern.match(col)
        if match_pair:
            a1, a2, feat = match_pair.group("ASSET1"), match_pair.group("ASSET2"), match_pair.group("FEATURE")
            assets.update((a1, a2))
            pairs.add(tuple(sorted((a1, a2))))
            pair_features.add(feat)
            continue

        match_single = single_asset_pattern.match(col)
        if match_single:
            asset = match_single.group("ASSET")
            assets.add(asset)
            feat = match_single.groupdict().get("FEATURE") or literal_feature
            if feat:
                single_features.add(feat)
            continue

        match_generic = generic_single_pattern.match(col)
        if match_generic:
            asset, feat = match_generic.group("ASSET"), match_generic.group("FEATURE")
            assets.add(asset)
            single_features.add(feat)
            continue

    return (
        sorted(assets),
        sorted(single_features),
        sorted(pair_features),
        sorted(pairs),
    )


def build_state_tensor_for_interval(
    df: pd.DataFrame,
    pair: tuple[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback: int,
    single_asset_features: list[str],
    pair_features: list[str],
    single_asset_format: str,
    pair_feature_format: str,
    timestamp_col: str = "datetime",
):
    """
    Build state tensor for a single asset pair and time interval.
    
    Returns:
        X: State tensor of shape (n_samples, n_features, lookback)
        R: Returns tensor of shape (n_samples, 2) - [asset1_return, asset2_return]
        VOL: Volatility scalar for each sample
        timestamps: List of timestamps for each sample
    """
    if timestamp_col and timestamp_col in df.columns:
        ts = df[timestamp_col]
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have timestamp column or DatetimeIndex")
        ts = df.index

    # Get all data in interval
    mask = (ts >= start) & (ts < end)
    df_interval = df[mask].copy()
    
    if len(df_interval) < lookback:
        return None, None, None, None

    asset1, asset2 = pair
    
    # Build feature columns
    feature_cols = []
    
    # Add single-asset features for both assets
    for feat in single_asset_features:
        col1 = single_asset_format.format(ASSET=asset1, FEATURE=feat)
        col2 = single_asset_format.format(ASSET=asset2, FEATURE=feat)
        if col1 in df_interval.columns and col2 in df_interval.columns:
            feature_cols.extend([col1, col2])
    
    # Add pair features
    for feat in pair_features:
        col = pair_feature_format.format(ASSET1=asset1, ASSET2=asset2, FEATURE=feat)
        if col in df_interval.columns:
            feature_cols.append(col)
    
    # Extract feature matrix
    feature_matrix = df_interval[feature_cols].values
    
    # Build rolling windows
    n_samples = len(feature_matrix) - lookback + 1
    n_features = len(feature_cols)
    
    X = np.zeros((n_samples, n_features, lookback))
    timestamps = []
    
    for i in range(n_samples):
        X[i] = feature_matrix[i:i+lookback].T
        timestamps.append(df_interval.index[i + lookback - 1])
    
    # Build returns (assuming 'close' feature exists)
    close1_col = single_asset_format.format(ASSET=asset1, FEATURE='close')
    close2_col = single_asset_format.format(ASSET=asset2, FEATURE='close')
    
    if close1_col in df_interval.columns and close2_col in df_interval.columns:
        close1 = df_interval[close1_col].values
        close2 = df_interval[close2_col].values
        
        # Log returns
        ret1 = np.diff(np.log(close1))
        ret2 = np.diff(np.log(close2))
        
        # Align with samples (skip first lookback-1)
        ret1 = ret1[lookback-1:]
        ret2 = ret2[lookback-1:]
        
        R = np.column_stack([ret1, ret2])
    else:
        R = np.zeros((n_samples, 2))
    
    # Build volatility (using pair volatility if available)
    vol_col = pair_feature_format.format(ASSET1=asset1, ASSET2=asset2, FEATURE='volatility')
    if vol_col in df_interval.columns:
        VOL = df_interval[vol_col].values[lookback-1:]
    else:
        VOL = np.ones(n_samples) * 0.01  # Default small volatility
    
    return X, R, VOL, timestamps


class PortfolioWeightsEnvUtility(gym.Env):
    """
    Statistical Arbitrage Environment using Quadratic Utility Function.
    
    Action space: Continuous [-1, 1]
    - action = 1.0: 100% invested (50% long asset1, 50% short asset2, 0% cash)
    - action = -1.0: 100% invested (50% short asset1, 50% long asset2, 0% cash)
    - action = 0.0: 100% cash (0% invested)
    - action = 0.5: 50% invested (25% long asset1, 25% short asset2, 50% cash)
    - action = -0.5: 50% invested (25% short asset1, 25% long asset2, 50% cash)
    
    In other words:
    - abs(action) controls capital allocation (0 = all cash, 1 = fully invested)
    - sign(action) controls position direction (+ = long/short, - = short/long)
    
    Reward function uses quadratic utility to penalize both upside and downside variance:
    R_t = x_t - (lambda/2) * x_t^2
    where x_t = net portfolio return after transaction costs
    """
    metadata = {"render_modes": []}

    def __init__(self, X, R, VOL, tickers, lookback, cfg_env):
        super().__init__()
        self.X = X
        self.R = R
        self.VOL = VOL
        self.tickers = tickers
        self.lookback = lookback
        self.cfg = cfg_env

        self.n_pairs = X.shape[1]
        self.active_pair_idx = 0
        self.n_assets = 2
        self.include_cash = cfg_env["include_cash"]

        n_features = X.shape[2]
        market_obs_dim = n_features * lookback
        position_obs_dim = 3
        obs_dim = market_obs_dim + position_obs_dim
        
        self.observation_space = spaces.Box(
            low=-5, 
            high=5, 
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Transaction costs
        self.taker_fee = cfg_env["transaction_costs"]["taker_bps"] / 1e4
        self.slippage = cfg_env["transaction_costs"]["slippage_bps"] / 1e4
        
        # Quadratic utility parameter
        self.lambda_utility = cfg_env["reward"].get("lambda_utility", 20.0)
        
        # Reward clipping to prevent NaN explosion
        self.reward_clip = cfg_env["reward"].get("reward_clip", 5.0)

        self.reset(seed=cfg_env.get("seed", 42))

    def _continuous_to_weights(self, action: float) -> np.ndarray:
        """
        Convert continuous action to portfolio weights [asset1, asset2, cash].
        
        Action interpretation:
        - abs(action) = capital allocation (0=all cash, 1=fully invested)
        - sign(action) = position direction (+=long/short, -=short/long)
        
        Examples:
        - action=1.0  → [+0.5, -0.5, 0.0] = 50% long asset1, 50% short asset2
        - action=-1.0 → [-0.5, +0.5, 0.0] = 50% short asset1, 50% long asset2
        - action=0.5  → [+0.25, -0.25, 0.5] = 25% long/short, 50% cash
        - action=0.0  → [0.0, 0.0, 1.0] = 100% cash
        """
        action = np.clip(action, -1.0, 1.0)
        
        # abs(action) determines how much capital to deploy
        capital_deployed = abs(action)
        
        # Split deployed capital 50/50 between long and short positions
        position_size = capital_deployed * 0.5
        
        # sign(action) determines which asset is long vs short
        if action >= 0:
            asset1_weight = position_size   # Long asset1
            asset2_weight = -position_size  # Short asset2
        else:
            asset1_weight = -position_size  # Short asset1
            asset2_weight = position_size   # Long asset2
        
        cash_weight = 1.0 - capital_deployed
        
        return np.array([asset1_weight, asset2_weight, cash_weight])
    
    def _to_obs(self, t):
        """Get observation at time t."""
        market_features = self.X[t, self.active_pair_idx, :, :].reshape(-1).astype(np.float32)
        position_features = self.w.astype(np.float32)
        obs = np.concatenate([market_features, position_features])
        return np.clip(obs, -5.0, 5.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.active_pair_idx = self.np_random.integers(0, self.n_pairs)
        self.t = 0
        self.portfolio_value = 1.0
        self.w = np.array([0.0, 0.0, 1.0])  # Start with 100% cash
        self.last_action = 0.0
        obs = self._to_obs(self.t)
        return obs, {}

    def step(self, action):
        # Extract scalar action
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)
        
        # Convert action to weights
        w_target = self._continuous_to_weights(action)
        turnover = np.sum(np.abs(w_target[:2] - self.w[:2]))
        trading_cost = (self.taker_fee + self.slippage) * turnover

        # Update weights
        self.w = w_target

        # Get returns for the active pair
        asset1_ret = self.R[self.t, 0]
        asset2_ret = self.R[self.t, 1]
        
        # Clip returns to prevent numerical instability
        asset1_ret = np.clip(asset1_ret, -0.1, 0.1)
        asset2_ret = np.clip(asset2_ret, -0.1, 0.1)
        
        # Calculate portfolio log return
        portfolio_log_ret = self.w[0] * asset1_ret + self.w[1] * asset2_ret
        
        # Net return after transaction costs
        net_return = portfolio_log_ret - trading_cost
        
        # QUADRATIC UTILITY REWARD FUNCTION
        reward = net_return - (self.lambda_utility / 2.0) * (net_return ** 2)
        
        # Clip reward to prevent extreme values that cause NaN explosion
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        # Update portfolio value
        self.portfolio_value *= math.exp(net_return)
        self.last_action = action
        
        self.t += 1
        terminated = (self.t >= len(self.R)-1)
        truncated = False

        obs = self._to_obs(self.t) if not terminated else self._to_obs(self.t-1)
        
        leverage = np.sum(np.abs(self.w[:2]))
        
        info = {
            "portfolio_value": self.portfolio_value,
            "total_leverage": leverage,
            "turnover": turnover,
            "portfolio_log_ret": portfolio_log_ret,
            "net_return": net_return,
            "utility_reward": reward,
            "action_taken": action
        }
        return obs, reward, terminated, truncated, info
