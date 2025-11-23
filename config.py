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
            "seperator": ",",
            "index": "datetime",
            "start": "2024-05-01 00:00:00",
            "end": "2025-05-01 00:00:00",
            "individual_identifier": "close",
            "pair_identifier": "beta",
        },
        "prices": {
            "price_folder_id": "1uXEBUyySypdsW_ZqL-RZ3d1bWdIZisij",
            "ADA_1h": "1ydaR3T68ReE_7j5t3wZbj0F-zdRPYoxg",
            "APT_1h": "1CxG9N2bqWPs9fOPOUryYNtHmONXo4SRi",
            "ARB_1h": "136FSMlAW3XHG8WocxxTEcSKiLMUBwWMi",
            "ATOM_1h": "1mhSQgEwRHn3nvu8Qu1ctQGzdW5JuxATR",
            "BTC_1h": "1-sBNQpEFGEpVO3GDFCkSZiV3Iaqp2vB_",
            "DOGE_1h": "14XlkoQMYr8WWecGninAKUavvjB3qNxk0",
            "DOT_1h": "1kCWB4ZZu3FnadbAquTa3Rcdcwkhnq6-s",
            "ENA_1h": "1TYTxexlD24cs7qmhyVoTacX7lqGOsfky",
            "ETC_1h": "1coBd9QiEX03MndMgX5_549mOPyY23ZcI",
            "ETH_1h": "1kj8G1scpFuEYTTXKEUzF9pwgGI2WFFL9",
            "HBAR_1h": "1LVseecBvXKl3Wl9hbPLsROYKR1Gp8zhQ",
            "LINK_1h": "1ZLEraxdV3H8jpf1FmPeVs1ySL7TzMvH5",
            "LTC_1h": "18d3_jD-tuYTQQR2QOwXupckeDgqvAIvx",
            "NEAR_1h": "1PqI2hD2gbDxUaRDPnJpvDNH5wPYv47G6",
            "SOL_1h": "17CjYYSEsTEqBdmm51zGLgmpkslxxjiji",
            "SUI_1h": "1bToOJts-x2Ia48tqXcMs4qFIQ5OV1lAP",
            "TON_1h": "1SARYo5zB6AunG82kw7KGF4Nird3lQ4zB",
            "TRX_1h": "1FlcZo1WRtKFQMbBrsb61Lp3_pplISW4U",
            "UNI_1h": "15L-eKWliyg9MBKuznlZZ-FJzm52Ovt20",
            "WLD_1h": "1XqD1K4-YZzPxYFHKHY3KmKWnnwi3zO20",
            "XLM_1h": "1_3E5-mORLWh3X16Hi0ccHwzVKg5QxoT4",
            "XRP_1h": "1crt2g_t0qpYnaGpcozl35yDeHhd4tmi4"
        }
    },
    "ENV": {
        "include_cash": True,
        "shorting": True,
        "trading_window_days": "1D",
        "sliding_window_step": "1D",
        "lookback_window": 24,
        "transaction_costs": {
            "commission_bps": 0,
            "slippage_bps": 0,
        },
        "reward": {
            "risk_lambda": 0.0001, # Risk aversion parameter for basic reward
            "lambda_utility": 5  # Risk aversion parameter for quadratic utility
        },
        "leverage": {
            "use_leverage": True,
            "long_cap": 1.0,
            "short_cap": 1.0,
            "use_asymmetric": False,
        },
        "constraints": {
            "min_weight": -1.0,
            "max_weight": 1.0,
            "sum_to_one": False
        },
        "seed": 42
    },
    "SPLITS": {
        "data_start": "2024-05-01",
        "data_end": "2025-04-30",
        "train": ["2024-05-01 00:00:00", "2024-12-31 23:59:59"],  # 8 months for training
        "val": ["2025-01-01 00:00:00", "2025-02-28 23:59:59"],    # 2 months for validation
        "test": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],   # 2 months for testing
    },
    "RL": {
        "timesteps": 3e6, # 1e6 - 3e6
        "policy": "MlpPolicy",
        "gamma": 0.995,
        "gae_lambda": 0.92,
        "clip_range": 0.25,
        "n_steps": 1440,  # 1 day of steps for 1m data
        "batch_size": 360,  # 25% of n_steps
        "learning_rate": 1e-4,
        "ent_coef": 0.05,
        "vf_coef": 0.7,
        "max_grad_norm": 0.5,
    },
    "EVAL": {
        "plots": True,
        "reports_dir": "./reports",
        "frequency": 10000
    },
    "IO": {
        "models_dir": "./models",
        "tb_logdir": "./tb_logs"
    }
}