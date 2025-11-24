CONFIG = {
    "DATA": {
        "forward_fill": True,
        "drop_na_after_ffill": True,
        "cache_dir": "./data_cache",
        "timestamp_format": "%Y-%m-%d %H:%M:%S",
        "asset_price_format": "{ASSET}_{FEATURE}",
        "pair_feature_format": "{ASSET1}_{ASSET2}_{FEATURE}",
        "timestamp_col": "timestamp",
        "sampling": "1m",
        "features": {
            "file_id": "1OCqEkOWV73Z8e-67fpqVL3r3ugVcfml8",
            "file_name": "bin_futures_full_features",
            "type": "csv",
            "separator": ",",
            "index": "datetime",
            "start": "2024-05-01 00:00:00",
            "end": "2025-05-01 00:00:00",
            "individual_identifier": "close",
            "pair_identifier": "beta",
        },
    },
    "ENV": {
        "include_cash": True,
        "trading_window_days": "1D",
        "sliding_window_step": "1D",
        "lookback_window": 180,  # 3 hours context
        "transaction_costs": {
            "commission_bps": 2.5,  # 0.025% per trade (realistic for maker+taker avg), improvement: {"taker": 3.5, "maker": 1},
            "slippage_bps": 1,    # 0.01% slippage for liquid pairs
        },
        "reward": {
            "risk_lambda": 0.005, # for standard environment reward
            "lambda_utility": 5, # for utility-based reward
        },
        "seed": 42,
    },
    "SPLITS": {
        "data_start": "2024-05-01",
        "data_end": "2025-04-30",
        "train": ["2024-05-01 00:00:00", "2024-12-31 23:59:59"],  # 8 months for training
        "val": ["2025-01-01 00:00:00", "2025-02-28 23:59:59"],    # 2 months for validation
        "test": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],   # 2 months for testing
    },
    "RL": {
        "timesteps": 10000, # 1_000_000 - 3_000_000
        "policy": "MlpPolicy",
        "gamma": 0.995,
        "gae_lambda": 0.92, # 0.9 - 0.97
        "clip_range": 0.25, # default 0.2
        "n_steps": 1440,  # 1 day of steps for 1m data
        "batch_size": 360,  # 25% of n_steps
        "learning_rate": 1e-4,
        "ent_coef": 0.01,
        "vf_coef": 0.7,
        "max_grad_norm": 0.5,
        "n_epochs": 10,    # not implemented
    },
    "EVAL": {
        "plots": True,
        "reports_dir": "./reports",
        "frequency": 1000,
    },
    "IO": {
        "models_dir": "./models",
        "tb_logdir": "./tb_logs",
    },
}