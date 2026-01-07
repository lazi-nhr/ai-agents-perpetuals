CONFIG = {
    "DATA": {
        "forward_fill": True,
        "drop_na_after_ffill": True,
        "cache_data": True,
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
        "lookback_window": 60, # context
        "transaction_costs": {
            "fee_structures": [0.0, 1.44, 4.5], # in bps: 0.0%, 0.0144%, 0.045%
            "taker_bps": 0, # fee based on hyperliquid tier system (0.0144% to 0.045%)
            "maker_bps": 1.0, # rebates (not implemented)
            "slippage_bps": 0, # slippage
        },
        "reward": {
            "lambda_basic": 0.01,  # Risk penalty for basic reward system
            "lambda_utility": 6,  # Risk penalty for utility-based reward system
            "reward_clip": 3.0,  # Clip rewards to prevent extreme values causing NaN
        },
        "seed": 42,
    },

    "SPLITS": {
        "data_start": "2024-05-01",
        "data_end": "2025-04-30",

        "train": ["2024-05-01 00:00:00", "2024-12-31 23:59:59"],  # 8 months for training (66.7% of data)
        "val": ["2025-01-01 00:00:00", "2025-02-28 23:59:59"],    # 2 months for validation (16.7% of data)
        "test": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],   # 2 months for testing (16.7% of data)

        #"test": ["2024-05-01 00:00:00", "2024-06-30 23:59:59"],   # 2 months for testing (16.7% of data)
        #"train": ["2024-07-01 00:00:00", "2025-02-28 23:59:59"],  # 8 months for training (66.7% of data)
        #"val": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],    # 2 months for validation (16.7% of data)
    },

    "RL": {
        # ===== General (All Algorithms)    =====
        "algorithm": "PPO",  # "SAC", "PPO", or "DQN" (to be matched with action_space_type)
        "timesteps": 2e6,  # Total training timesteps (2e6 - 4e6)
        "policy": "MlpPolicy",  # Policy network architecture
        "gamma": 0.99,  # Discount factor for future rewards
        "learning_rate": 3e-4,  # Learning rate for optimizer (standard for PPO)
        "batch_size": 128,  # Batch size for training updates (must be power of 2)
        
        # -----         On-Policy           -----

        # The agent learns from data collected by its current policy
        # A policy defines the agent's behavior at a given time

        # =====         PPO (Continuous)    =====
        "gae_lambda": 0.95,  # GAE lambda for advantage estimation
        "clip_range": 0.2,  # PPO clipping range for policy updates (standard default)
        "n_steps": 2048,  # Number of steps to collect before update (MUST be divisible by batch_size)
        "n_epochs": 10,  # INCREASED: More epochs per batch for better learning
        "ent_coef": 0.1,  # INCREASED: Higher entropy for more exploration
        "vf_coef": 0.5,  # Value function loss coefficient
        "max_grad_norm": 0.5,  # Maximum gradient norm for clipping (reduced from 1.0 to prevent large updates)
        "use_sde": True,  # Use State Dependent Exploration for continuous action noise
        "sde_sample_freq": -1,  # CHANGED: Sample new noise every step for maximum exploration
        
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
        # Plotting and reporting settings
        "plots": True,
        "reports_dir": "./reports",

        # RL evaluation settings
        "frequency": 14400,  # Evaluation frequency in timesteps
        "n_eval_episodes": 3,  # Number of episodes per evaluation
        "save_freq": 144000, # Save model every N timesteps
        "action_std_threshold": 0.01,  # Threshold for action std to consider policy collapsed
        "action_extreme_threshold": 1,  # Threshold for extreme actions to consider policy collapsed
    },
    
    "IO": {
        "models_dir": "./models",
        "tb_logdir": "./tb_logs",
    },
}