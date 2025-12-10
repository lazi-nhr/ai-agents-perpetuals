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
        "action_space_type": "continuous",  # "continuous" or "discrete"

        # 21 discrete actions with 0.1 steps from -1.0 to 1.0
        "discrete_actions": [-1.0, -0.9, -0.8, -0.7, -0.6, 
                             -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 
                             0.1, 0.2, 0.3, 0.4, 0.5, 
                             0.6, 0.7, 0.8, 0.9, 1.0],
        "trading_window_days": "1D",
        "sliding_window_step": "1D",
        "lookback_window": 30, # context
        "transaction_costs": {
            "taker_bps": 1.44, # fee based on hyperliquid tier system (0.0144% to 0.045%)
            "maker_bps": 1.0, # rebates (not implemented)
            "slippage_bps": 0, # slippage
        },
        "reward": {
            "lambda_basic": 0.001,  # Risk penalty to discourage extreme volatility
            "lambda_utility": 10,  # High value penalizes extreme returns and prevents gradient explosion
            "reward_clip": 5.0,  # Clip rewards to prevent extreme values causing NaN
        },
        "seed": 42,
    },

    "SPLITS": {
        "data_start": "2024-05-01",
        "data_end": "2025-04-30",

        #"train": ["2024-05-01 00:00:00", "2024-12-31 23:59:59"],  # 8 months for training (66.7% of data)
        #"val": ["2025-01-01 00:00:00", "2025-02-28 23:59:59"],    # 2 months for validation (16.7% of data)
        #"test": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],   # 2 months for testing (16.7% of data)

        "test": ["2024-05-01 00:00:00", "2024-06-30 23:59:59"],   # 2 months for testing (16.7% of data)
        "train": ["2024-07-01 00:00:00", "2025-02-28 23:59:59"],  # 8 months for training (66.7% of data)
        "val": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],    # 2 months for validation (16.7% of data)
    },

    "RL": {
        # ===== General (All Algorithms)    =====
        "algorithm": "PPO",  # "SAC", "PPO", or "DQN" (to be matched with action_space_type)
        "timesteps": 4e6,  # Total training timesteps (2e6 - 4e6)
        "policy": "MlpPolicy",  # Policy network architecture
        "gamma": 0.995,  # Discount factor for future rewards
        "learning_rate": 3e-4,  # Learning rate for optimizer (standard for PPO)
        "batch_size": 64,  # Batch size for training updates
        
        # -----         On-Policy           -----

        # The agent learns from data collected by its current policy
        # A policy defines the agent's behavior at a given time

        # =====         PPO (Continuous)    =====
        "gae_lambda": 0.95,  # GAE lambda for advantage estimation
        "clip_range": 0.2,  # PPO clipping range for policy updates (standard default)
        "n_steps": 2048,  # Number of steps to collect before update
        "n_epochs": 10,  # Number of epochs for policy optimization
        "ent_coef": 0.1,  # Entropy coefficient for exploration (PPO-specific, conflicts with SAC)
        "vf_coef": 0.5,  # Value function loss coefficient
        "max_grad_norm": 1.0,  # Maximum gradient norm for clipping (less aggressive)
        
        # -----         Off-Policy          -----
        # It means the agent learns from data collected by a different policy than its current one
        # This allows the use of experience replay buffers to learn from past experiences
        # Off policy means algorithms like SAC and DQN

        "buffer_size": 200000,  # Larger replay buffer for more diverse experiences
        "learning_starts": 5000,  # Collect more random experiences before learning
        "tau": 0.005,  # Soft update coefficient for target networks
        "train_freq": 1,  # Update model every N steps
        "gradient_steps": 1,  # Gradient steps per environment step

        # =====         SAC (Continuous)    =====
        "ent_coef_SAC": "auto",  # Automatic entropy tuning for better exploration
        
        # =====         DQN (Discrete)      =====
        # "buffer_size": 100000,  # Replay buffer size (shared with SAC)
        # "learning_starts": 1000,  # Steps before learning starts (shared with SAC)
        "exploration_fraction": 0.1,  # Fraction of training for epsilon decay
        "exploration_initial_eps": 1.0,  # Initial epsilon for exploration
        "exploration_final_eps": 0.05,  # Final epsilon after decay
        "target_update_interval": 1,  # Update target network every N steps
    },

    "EVAL": {
        "plots": True,
        "reports_dir": "./reports",
        "frequency": 14400,  # Evaluate every 14,400 steps
        "n_eval_episodes": 1,  # Number of episodes to run during evaluation (default: 5)
        "save_freq": 144000,
    },
    
    "IO": {
        "models_dir": "./models",
        "tb_logdir": "./tb_logs",
    },
}