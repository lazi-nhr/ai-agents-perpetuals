import numpy as np
import pandas as pd
from stable_baselines3 import PPO


"""
STANDALONE INFERENCE FUNCTION
==============================
This function can be copied to a separate Python script for production use.
It takes features for a single timestep and generates a trading signal.

Required imports for standalone use:
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from stable_baselines3 import PPO

Feature structure:
------------------
Single asset features (per asset): 12 features
    ['close', 'closeBias', 'closeEwmaVol', 'closeKalman', 'closeLowerShadow', 
     'closeMacd', 'closeRsi', 'closeSigns', 'closeStochRsi', 'closeUpperShadow', 
     'funding', 'fundingMinutesLeft']

Pair features (for both assets combined): 8 features
    ['alpha', 'beta', 'corr', 'pval', 'spreadNorm', 'spreadNormKalman', 
     'spreadNormMa', 'spreadNormVol']
"""

def continuous_action_to_weights(action: float) -> tuple[float, float]:
    """
    Convert continuous action [-1, 1] to portfolio weights.
    
    Parameters
    ----------
    action : float
        Action value from -1 (short asset1/long asset2) to 1 (long asset1/short asset2)
    
    Returns
    -------
    tuple[float, float]
        (asset1_weight, asset2_weight)
    """
    action = np.clip(action, -1.0, 1.0)
    position_size = action * 0.5
    asset1_weight = position_size
    asset2_weight = -position_size
    return asset1_weight, asset2_weight


def generate_signal_from_features(
    model_path: str,
    asset1: str,
    asset2: str,
    asset1_features: dict, # 'close', 'closeBias', 'closeEwmaVol', 'closeKalman', 'closeLowerShadow', 'closeMacd', 'closeRsi', 'closeSigns', 'closeStochRsi', 'closeUpperShadow', 'funding', 'fundingMinutesLeft'
    asset2_features: dict, # 'close', 'closeBias', 'closeEwmaVol', 'closeKalman', 'closeLowerShadow', 'closeMacd', 'closeRsi', 'closeSigns', 'closeStochRsi', 'closeUpperShadow', 'funding', 'fundingMinutesLeft'
    pair_features: dict, # 'alpha', 'beta', 'corr', 'pval', 'spreadNorm', 'spreadNormKalman', 'spreadNormMa', 'spreadNormVol'
    lookback_window: int = 24,
    timestamp: str = None,
    notional_usd: float = 100,
    bar_timeframe: str = "1h",
    exchange: str = "binance_perpetual",
    fresh_for_seconds: int = 3600,
    version: int = 1
) -> dict:
    """
    Generate a trading signal from features for a single timestep.
    
    This is a STANDALONE function that can be copied to a separate Python script.
    
    Parameters
    ----------
    model_path : str
        Path to the trained PPO model (.zip file)
    asset1 : str
        First asset symbol (e.g., "ETH")
    asset2 : str
        Second asset symbol (e.g., "BTC")
    asset1_features : dict
        Features for asset1 with keys:
        ['close', 'closeBias', 'closeEwmaVol', 'closeKalman', 'closeLowerShadow',
         'closeMacd', 'closeRsi', 'closeSigns', 'closeStochRsi', 'closeUpperShadow',
         'funding', 'fundingMinutesLeft']
        Each value should be a list/array of length lookback_window
    asset2_features : dict
        Features for asset2 (same structure as asset1_features)
    pair_features : dict
        Combined pair features with keys:
        ['alpha', 'beta', 'corr', 'pval', 'spreadNorm', 'spreadNormKalman',
         'spreadNormMa', 'spreadNormVol']
        Each value should be a list/array of length lookback_window
    lookback_window : int
        Number of historical timesteps (default: 24)
    timestamp : str, optional
        ISO format timestamp. If None, uses current time.
    notional_usd : float
        Notional value in USD
    bar_timeframe : str
        Timeframe of bars (e.g., "1h", "1m")
    exchange : str
        Exchange identifier
    fresh_for_seconds : int
        Signal validity duration in seconds
    version : int
        Signal format version
    
    Returns
    -------
    dict
        Trading signal in JSON format
    
    Example
    -------
    >>> signal = generate_signal_from_features(
    ...     model_path="./models/ppo_model.zip",
    ...     asset1="ETH",
    ...     asset2="BTC",
    ...     asset1_features={
    ...         'close': [0.1, 0.2, ...],  # 24 values
    ...         'closeBias': [0.05, -0.03, ...],
    ...         # ... other 10 features
    ...     },
    ...     asset2_features={
    ...         'close': [0.15, 0.18, ...],
    ...         # ... other 11 features
    ...     },
    ...     pair_features={
    ...         'alpha': [0.01, 0.02, ...],  # 24 values
    ...         'beta': [0.98, 0.99, ...],
    ...         # ... other 6 features
    ...     }
    ... )
    """
    
    # Define feature order (must match training order)
    SINGLE_ASSET_FEATURES = [
        'close', 'closeBias', 'closeEwmaVol', 'closeKalman', 'closeLowerShadow',
        'closeMacd', 'closeRsi', 'closeSigns', 'closeStochRsi', 'closeUpperShadow',
        'funding', 'fundingMinutesLeft'
    ]
    
    PAIR_FEATURES = [
        'alpha', 'beta', 'corr', 'pval', 'spreadNorm', 'spreadNormKalman',
        'spreadNormMa', 'spreadNormVol'
    ]
    
    # Validate inputs
    for feat in SINGLE_ASSET_FEATURES:
        if feat not in asset1_features:
            raise ValueError(f"Missing feature '{feat}' in asset1_features")
        if feat not in asset2_features:
            raise ValueError(f"Missing feature '{feat}' in asset2_features")
        if len(asset1_features[feat]) != lookback_window:
            raise ValueError(f"asset1_features['{feat}'] must have length {lookback_window}")
        if len(asset2_features[feat]) != lookback_window:
            raise ValueError(f"asset2_features['{feat}'] must have length {lookback_window}")
    
    for feat in PAIR_FEATURES:
        if feat not in pair_features:
            raise ValueError(f"Missing feature '{feat}' in pair_features")
        if len(pair_features[feat]) != lookback_window:
            raise ValueError(f"pair_features['{feat}'] must have length {lookback_window}")
    
    # Build observation tensor
    # Shape: (n_features * lookback_window,) where n_features = 12 + 12 + 8 = 32
    # But we only use pair features for the model since it's pair trading
    # Total features per pair = 8 pair features
    # Observation shape: (8 features * 24 lookback,) = (192,)
    
    n_pair_features = len(PAIR_FEATURES)
    obs = np.zeros((n_pair_features, lookback_window), dtype=np.float32)
    
    # Fill pair features
    for i, feat in enumerate(PAIR_FEATURES):
        obs[i, :] = pair_features[feat]
    
    # Flatten to (n_features * lookback_window,)
    obs = obs.reshape(-1).astype(np.float32)
    
    # Clip observation values (same as training)
    obs = np.clip(obs, -5.0, 5.0)
    
    # Load model and predict
    model = PPO.load(model_path)
    action, _ = model.predict(obs, deterministic=True)
    
    # Convert action to weights
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
    asset1_weight, asset2_weight = continuous_action_to_weights(action_value)
    
    # Get or generate timestamp
    if timestamp is None:
        timestamp = pd.Timestamp.now(tz='UTC').strftime("%Y-%m-%dT%H:%M:%SZ")
    elif isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Create signal JSON
    signal = {
        "timestamp": timestamp,
        "bar_timeframe": bar_timeframe,
        "pair": {
            "asset1": asset1,
            "asset2": asset2
        },
        "markets": {
            "exchange": exchange,
            "asset1": f"{asset1}-USDT",
            "asset2": f"{asset2}-USDT"
        },
        "weights": {
            "asset1": float(asset1_weight),
            "asset2": float(asset2_weight)
        },
        "notional_usd": notional_usd,
        "fresh_for_seconds": fresh_for_seconds,
        "version": version
    }
    
    return signal
