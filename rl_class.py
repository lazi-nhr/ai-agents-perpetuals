from config import CONFIG
import os
import json
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# Import functions from rl_utils module
from rl_utils import (
    identify_assets_features_pairs,
    build_state_tensor_for_interval,
    PortfolioWeightsEnvUtility,
    ensure_dir,
    load_csv_to_df,
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# load features
file_name = CONFIG["DATA"]["features"]["file_name"]
cache_dir = CONFIG["DATA"]["cache_dir"]
# Convert to absolute path if relative
if not os.path.isabs(cache_dir):
    cache_dir = os.path.join(SCRIPT_DIR, cache_dir)
    
index = CONFIG["DATA"]["features"]["index"]
sep = CONFIG["DATA"]["features"].get("seperator", ",")
file_path = os.path.join(cache_dir, f"{file_name}.csv")

# Check if file exists
if not os.path.exists(file_path):
    print(f"\n⚠ ERROR: Data file not found: {file_path}")
    print("\nPlease ensure the data file exists or run the data preparation notebook first.")
    print(f"Expected location: {os.path.abspath(file_path)}")
    print("\nYou can either:")
    print("  1. Download the data using the notebook cells")
    print("  2. Update CONFIG['DATA']['cache_dir'] to point to the correct location")
    print("  3. Update CONFIG['DATA']['features']['file_name'] if using a different file\n")
    exit(1)

features_df = load_csv_to_df(file_path, sep, timestamp_index_col=index)

# print dataframe info
print("Features DataFrame Info:")
print(features_df.info())




#######################################################################################




class PPOAgentManager:
    """
    Unified class for training, fine-tuning, and inference with PPO agent.
    
    Features:
    - Load raw CSV data
    - Train new models or fine-tune existing ones
    - Single timestep deterministic prediction
    - Automatic model saving/loading
    - Export predictions to JSON
    
    Example usage:
    ```python
    # Initialize manager
    manager = PPOAgentManager(config=CONFIG)
    
    # Train new model
    manager.train(
        csv_path="data/features.csv",
        train_period=["2024-01-01", "2024-06-30"],
        val_period=["2024-07-01", "2024-07-31"],
        timesteps=1000000
    )
    
    # Fine-tune on new data
    manager.fine_tune(
        csv_path="data/features_new.csv",
        model_path="models/best_model.zip",
        train_period=["2024-08-01", "2024-09-30"],
        timesteps=100000
    )
    
    # Predict single timestep
    result = manager.predict(
        csv_path="data/features_latest.csv",
        model_path="models/best_model.zip",
        timestamp="2024-10-01 12:00:00"
    )
    print(result)  # {"timestamp": "...", "action": 0.85, "confidence": 0.92}
    ```
    """
    
    def __init__(self, config: dict):
        """
        Initialize the PPO Agent Manager.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing all settings
        """
        self.config = config
        self.model = None
        self.env = None
        self.feature_columns = None
        self.assets = None
        
    def _load_and_prepare_data(self, csv_path: str) -> tuple:
        """
        Load CSV data and prepare features.
        
        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing features
            
        Returns
        -------
        tuple
            (features_df, X_tensor, R_tensor, VOL_tensor, timestamps, ticker_order)
        """
        print(f"Loading data from: {csv_path}")
        
        # Load CSV
        features_df = load_csv_to_df(
            csv_path,
            sep=self.config["DATA"]["features"].get("separator", ","),
            timestamp_index_col=self.config["DATA"]["features"]["index"]
        )
        
        print(f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns")
        
        # Identify assets and features
        assets, single_features, pair_features, pairs = identify_assets_features_pairs(
            features_df,
            self.config["DATA"]["asset_price_format"],
            self.config["DATA"]["pair_feature_format"]
        )
        
        self.assets = sorted(list(assets))
        print(f"Detected {len(self.assets)} assets: {self.assets}")
        print(f"Single features: {sorted(single_features)}")
        print(f"Pair features: {sorted(pair_features)}")
        
        # Build state tensor
        X_tensor, R_tensor, VOL_tensor, timestamps, ticker_order = build_state_tensor_for_interval(
            features_df,
            self.assets,
            sorted(single_features),
            sorted(pair_features),
            lookback=self.config["ENV"]["lookback_window"]
        )
        
        print(f"Built tensors: X{X_tensor.shape}, R{R_tensor.shape}, VOL{VOL_tensor.shape}")
        print(f"Timestamps: {len(timestamps)} samples from {timestamps.min()} to {timestamps.max()}")
        
        return features_df, X_tensor, R_tensor, VOL_tensor, timestamps, ticker_order
    
    def _create_time_mask(self, timestamps: pd.DatetimeIndex, period: list) -> np.ndarray:
        """
        Create boolean mask for time period.
        
        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            All available timestamps
        period : list
            [start_date, end_date] as strings
            
        Returns
        -------
        np.ndarray
            Boolean mask
        """
        start = pd.to_datetime(period[0]).tz_localize('UTC')
        end = pd.to_datetime(period[1]).tz_localize('UTC')
        
        # Ensure timestamps are UTC
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')
        elif timestamps.tz != pytz.UTC:
            timestamps = timestamps.tz_convert('UTC')
        
        mask = (timestamps >= start) & (timestamps <= end)
        print(f"Time mask: {mask.sum()} / {len(mask)} samples in period {period[0]} to {period[1]}")
        
        return mask
    
    def _create_env(self, X: np.ndarray, R: np.ndarray, VOL: np.ndarray, 
                    ticker_order: list, name: str = "env") -> gym.Env:
        """
        Create environment instance.
        
        Parameters
        ----------
        X, R, VOL : np.ndarray
            State tensors
        ticker_order : list
            List of asset tickers
        name : str
            Environment name for monitoring
            
        Returns
        -------
        gym.Env
            Wrapped environment
        """
        env = PortfolioWeightsEnvUtility(
            X, R, VOL, ticker_order,
            self.config["ENV"]["lookback_window"],
            self.config["ENV"]
        )
        env = Monitor(env, filename=None)
        return env
    
    def train(self, csv_path: str, train_period: list, val_period: list,
              timesteps: int = None, save_path: str = None) -> dict:
        """
        Train a new PPO model from scratch.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with features
        train_period : list
            [start_date, end_date] for training
        val_period : list
            [start_date, end_date] for validation
        timesteps : int, optional
            Number of training timesteps (uses config if None)
        save_path : str, optional
            Where to save the model (uses config if None)
            
        Returns
        -------
        dict
            Training metrics and paths
        """
        print("\n" + "="*70)
        print("TRAINING NEW MODEL")
        print("="*70)
        
        # Load and prepare data
        _, X_all, R_all, VOL_all, timestamps, ticker_order = self._load_and_prepare_data(csv_path)
        
        # Create time masks
        train_mask = self._create_time_mask(timestamps, train_period)
        val_mask = self._create_time_mask(timestamps, val_period)
        
        # Slice data
        X_train = X_all[train_mask]
        R_train = R_all[train_mask]
        VOL_train = VOL_all[train_mask]
        
        X_val = X_all[val_mask]
        R_val = R_all[val_mask]
        VOL_val = VOL_all[val_mask]
        
        # Create environments
        train_env = self._create_env(X_train, R_train, VOL_train, ticker_order, "train")
        val_env = self._create_env(X_val, R_val, VOL_val, ticker_order, "val")
        
        vec_train = DummyVecEnv([lambda: train_env])
        vec_val = DummyVecEnv([lambda: val_env])
        
        # Create model
        print("\nInitializing PPO model...")
        self.model = PPO(
            policy=self.config["RL"]["policy"],
            env=vec_train,
            gamma=self.config["RL"]["gamma"],
            gae_lambda=self.config["RL"]["gae_lambda"],
            clip_range=self.config["RL"]["clip_range"],
            n_steps=self.config["RL"]["n_steps"],
            batch_size=self.config["RL"]["batch_size"],
            learning_rate=self.config["RL"]["learning_rate"],
            ent_coef=self.config["RL"]["ent_coef"],
            vf_coef=self.config["RL"]["vf_coef"],
            max_grad_norm=self.config["RL"]["max_grad_norm"],
            tensorboard_log=self.config["IO"]["tb_logdir"],
            device="cpu",
            verbose=0
        )
        
        # Setup callbacks
        save_path = save_path or self.config["IO"]["models_dir"]
        ensure_dir(save_path)
        
        eval_callback = EvalCallback(
            vec_val,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=self.config["EVAL"]["frequency"],
            n_eval_episodes=self.config["EVAL"]["n_eval_episodes"],
            deterministic=True,
            render=False,
            verbose=0
        )
        
        # Train
        timesteps = timesteps or int(self.config["RL"]["timesteps"])
        print(f"\nStarting training for {timesteps:,} timesteps...")
        print("Monitor progress: tensorboard --logdir=./tb_logs\n")
        
        self.model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
        
        # Save final model
        final_path = os.path.join(save_path, "final_model.zip")
        self.model.save(final_path)
        print(f"\n✓ Training complete! Model saved to: {final_path}")
        
        return {
            "final_model_path": final_path,
            "best_model_path": os.path.join(save_path, "best_model.zip"),
            "timesteps": timesteps,
            "train_period": train_period,
            "val_period": val_period
        }
    
    def fine_tune(self, csv_path: str, model_path: str, train_period: list,
                  val_period: list = None, timesteps: int = None, 
                  save_path: str = None) -> dict:
        """
        Fine-tune an existing model on new data (transfer learning).
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with new features
        model_path : str
            Path to pre-trained model
        train_period : list
            [start_date, end_date] for fine-tuning
        val_period : list, optional
            [start_date, end_date] for validation
        timesteps : int, optional
            Number of fine-tuning timesteps (default: 10% of original training)
        save_path : str, optional
            Where to save fine-tuned model
            
        Returns
        -------
        dict
            Fine-tuning metrics and paths
        """
        print("\n" + "="*70)
        print("FINE-TUNING EXISTING MODEL")
        print("="*70)
        
        # Load pre-trained model
        print(f"Loading pre-trained model from: {model_path}")
        self.model = PPO.load(model_path)
        
        # Load and prepare new data
        _, X_all, R_all, VOL_all, timestamps, ticker_order = self._load_and_prepare_data(csv_path)
        
        # Create time masks
        train_mask = self._create_time_mask(timestamps, train_period)
        
        # Slice data
        X_train = X_all[train_mask]
        R_train = R_all[train_mask]
        VOL_train = VOL_all[train_mask]
        
        # Create training environment
        train_env = self._create_env(X_train, R_train, VOL_train, ticker_order, "finetune_train")
        vec_train = DummyVecEnv([lambda: train_env])
        
        # Update model's environment
        self.model.set_env(vec_train)
        
        # Setup validation if provided
        vec_val = None
        if val_period:
            val_mask = self._create_time_mask(timestamps, val_period)
            X_val = X_all[val_mask]
            R_val = R_all[val_mask]
            VOL_val = VOL_all[val_mask]
            val_env = self._create_env(X_val, R_val, VOL_val, ticker_order, "finetune_val")
            vec_val = DummyVecEnv([lambda: val_env])
        
        # Setup callbacks
        save_path = save_path or self.config["IO"]["models_dir"]
        ensure_dir(save_path)
        
        callbacks = []
        if vec_val:
            eval_callback = EvalCallback(
                vec_val,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=self.config["EVAL"]["frequency"],
                n_eval_episodes=self.config["EVAL"]["n_eval_episodes"],
                deterministic=True,
                render=False,
                verbose=0
            )
            callbacks.append(eval_callback)
        
        # Fine-tune
        timesteps = timesteps or int(self.config["RL"]["timesteps"] * 0.1)  # 10% of original
        print(f"\nFine-tuning for {timesteps:,} timesteps...")
        
        callback = CallbackList(callbacks) if callbacks else None
        self.model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True, reset_num_timesteps=False)
        
        # Save fine-tuned model
        finetuned_path = os.path.join(save_path, "finetuned_model.zip")
        self.model.save(finetuned_path)
        print(f"\n✓ Fine-tuning complete! Model saved to: {finetuned_path}")
        
        return {
            "finetuned_model_path": finetuned_path,
            "timesteps": timesteps,
            "train_period": train_period,
            "val_period": val_period
        }
    
    def predict(self, model_path: str, current_features: dict,
                current_position: list = None, output_json: str = None) -> dict:
        """
        Predict next action based on current market features.
        
        This is the ONLINE inference function - provide current market state
        and get the next trading action immediately.
        
        Parameters
        ----------
        model_path : str
            Path to trained model
        current_features : dict
            Current market features for the pair. Keys:
            - 'X': np.ndarray of shape (n_features, lookback) - market features
            - 'timestamp': str or pd.Timestamp - current time (optional)
            - 'pair_name': str - trading pair name (optional)
        current_position : list, optional
            Current portfolio weights [asset1_weight, asset2_weight, cash_weight]
            Default: [0.0, 0.0, 1.0] (100% cash)
        output_json : str, optional
            Path to save prediction as JSON
            
        Returns
        -------
        dict
            Prediction result with action and interpretation:
            {
                'timestamp': current time,
                'action': raw action value in [-1, 1],
                'capital_deployed': abs(action) - % of capital to deploy,
                'position_direction': 'long/short' or 'short/long',
                'portfolio_weights': [asset1, asset2, cash],
                'interpretation': human-readable description
            }
        
        Example
        -------
        >>> # Get current market data from your live system
        >>> current_X = get_latest_features()  # shape (15, 30)
        >>> result = manager.predict(
        ...     model_path='models/best_model.zip',
        ...     current_features={'X': current_X, 'timestamp': '2024-10-01 12:00:00'}
        ... )
        >>> print(f"Deploy {result['capital_deployed']*100:.1f}% capital")
        >>> print(f"Position: {result['interpretation']}")
        """
        print("\n" + "="*70)
        print("ONLINE PREDICTION - Current Market State")
        print("="*70)
        
        # Load model
        if self.model is None or model_path:
            print(f"Loading model from: {model_path}")
            self.model = PPO.load(model_path)
        
        # Extract features
        X_current = current_features['X']
        timestamp = current_features.get('timestamp', datetime.now().isoformat())
        pair_name = current_features.get('pair_name', 'Unknown')
        
        # Validate shape
        if len(X_current.shape) != 2:
            raise ValueError(f"X must be 2D (n_features, lookback), got shape {X_current.shape}")
        
        n_features, lookback = X_current.shape
        print(f"✓ Features: {n_features} features × {lookback} timesteps")
        print(f"✓ Pair: {pair_name}")
        print(f"✓ Timestamp: {timestamp}")
        
        # Default position if not provided
        if current_position is None:
            current_position = [0.0, 0.0, 1.0]  # 100% cash
        
        # Build observation (market features + current position)
        market_obs = X_current.reshape(-1).astype(np.float32)
        position_obs = np.array(current_position, dtype=np.float32)
        obs = np.concatenate([market_obs, position_obs])
        obs = np.clip(obs, -5.0, 5.0)
        
        print(f"✓ Observation built: {obs.shape[0]} dims (market + position)")
        
        # Predict (deterministic)
        action, _states = self.model.predict(obs, deterministic=True)
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        print(f"\n🤖 Model Prediction: action = {action_value:+.4f}")
        
        # Interpret action using the same logic as environment
        capital_deployed = abs(action_value)
        position_size = capital_deployed * 0.5
        
        if action_value >= 0:
            asset1_weight = position_size
            asset2_weight = -position_size
            direction = "LONG asset1 / SHORT asset2"
        else:
            asset1_weight = -position_size
            asset2_weight = position_size
            direction = "SHORT asset1 / LONG asset2"
        
        cash_weight = 1.0 - capital_deployed
        
        # Create human-readable interpretation
        if capital_deployed < 0.05:
            interpretation = "STAY IN CASH (no position)"
        elif capital_deployed < 0.3:
            interpretation = f"SMALL POSITION: {capital_deployed*100:.1f}% deployed, {direction}"
        elif capital_deployed < 0.7:
            interpretation = f"MEDIUM POSITION: {capital_deployed*100:.1f}% deployed, {direction}"
        else:
            interpretation = f"LARGE POSITION: {capital_deployed*100:.1f}% deployed, {direction}"
        
        # Build result
        result = {
            "timestamp": str(timestamp),
            "pair": pair_name,
            "action": action_value,
            "capital_deployed": capital_deployed,
            "position_direction": direction,
            "portfolio_weights": {
                "asset1": asset1_weight,
                "asset2": asset2_weight,
                "cash": cash_weight
            },
            "interpretation": interpretation,
            "model_path": model_path,
            "current_position": current_position,
            "prediction_time": datetime.now().isoformat()
        }
        
        # Save to JSON if requested
        if output_json:
            ensure_dir(os.path.dirname(output_json))
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Prediction saved to: {output_json}")
        
        print("\n" + "="*70)
        print("📊 TRADING SIGNAL")
        print("="*70)
        print(f"Capital Deployment: {capital_deployed*100:.1f}%")
        print(f"Position Direction: {direction}")
        print("Portfolio Allocation:")
        print(f"  • Asset 1: {asset1_weight*100:+.1f}%")
        print(f"  • Asset 2: {asset2_weight*100:+.1f}%")
        print(f"  • Cash:    {cash_weight*100:+.1f}%")
        print(f"\n💡 {interpretation}")
        print("="*70)
        
        return result
    
    def batch_predict(self, csv_path: str, model_path: str, 
                     start_time: str, end_time: str,
                     output_json: str = None) -> list:
        """
        Predict actions for multiple timesteps.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with features
        model_path : str
            Path to trained model
        start_time, end_time : str
            Time range for predictions
        output_json : str, optional
            Path to save predictions as JSON
            
        Returns
        -------
        list
            List of prediction dictionaries
        """
        print("\n" + "="*70)
        print("BATCH PREDICTION")
        print("="*70)
        
        # Load model
        if self.model is None or model_path:
            print(f"Loading model from: {model_path}")
            self.model = PPO.load(model_path)
        
        # Load and prepare data
        _, X_all, R_all, VOL_all, timestamps, ticker_order = self._load_and_prepare_data(csv_path)
        
        # Create time mask
        period = [start_time, end_time]
        time_mask = self._create_time_mask(timestamps, period)
        
        # Get indices
        indices = np.where(time_mask)[0]
        print(f"Predicting for {len(indices)} timesteps...")
        
        results = []
        for idx in indices:
            state = X_all[idx]
            action, _ = self.model.predict(state, deterministic=True)
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            
            result = {
                "timestamp": str(timestamps[idx]),
                "action": action_value,
                "returns": R_all[idx].tolist() if isinstance(R_all[idx], np.ndarray) else [float(R_all[idx])],
                "volatility": VOL_all[idx].tolist() if isinstance(VOL_all[idx], np.ndarray) else [float(VOL_all[idx])]
            }
            results.append(result)
        
        # Save to JSON if requested
        if output_json:
            ensure_dir(os.path.dirname(output_json))
            output = {
                "model_path": model_path,
                "period": period,
                "assets": ticker_order,
                "num_predictions": len(results),
                "predictions": results
            }
            with open(output_json, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"✓ Batch predictions saved to: {output_json}")
        
        return results

def create_synthetic_state_tensor(n_samples=1000, n_pairs=5, n_features=15, lookback=30):
    """
    Create synthetic state tensors for testing/demo purposes.
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    n_pairs : int
        Number of asset pairs
    n_features : int
        Number of features per pair
    lookback : int
        Lookback window size
        
    Returns
    -------
    tuple
        (X, R, VOL, timestamps, ticker_order)
    """
    print(f"\n🔧 Creating synthetic data: {n_samples} samples, {n_pairs} pairs, {n_features} features, lookback={lookback}")
    
    # Generate synthetic data with realistic patterns
    np.random.seed(42)
    
    # X: (n_samples, n_pairs, n_features, lookback)
    X = np.random.randn(n_samples, n_pairs, n_features, lookback).astype(np.float32)
    
    # Add some trend and mean-reversion patterns
    for i in range(n_pairs):
        # Add autocorrelation (mean reversion)
        for t in range(1, n_samples):
            X[t, i] = 0.9 * X[t-1, i] + 0.1 * np.random.randn(n_features, lookback).astype(np.float32)
    
    # Normalize to reasonable range
    X = np.clip(X, -5.0, 5.0)
    
    # R: (n_samples, 2) - returns for asset1 and asset2
    # Generate correlated returns with some noise
    base_returns = np.random.randn(n_samples, 1) * 0.001  # Market factor
    R = np.concatenate([
        base_returns + np.random.randn(n_samples, 1) * 0.0005,  # Asset 1
        base_returns + np.random.randn(n_samples, 1) * 0.0005   # Asset 2
    ], axis=1).astype(np.float32)
    
    # VOL: (n_samples, 1) - volatility
    VOL = np.abs(np.random.randn(n_samples, 1) * 0.02 + 0.01).astype(np.float32)
    
    # Create timestamps (1-minute intervals)
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='1min')
    
    # Create ticker order (pair names)
    ticker_order = [f"PAIR_{i}" for i in range(n_pairs)]
    
    print("✓ Synthetic tensors created:")
    print(f"  X shape: {X.shape}")
    print(f"  R shape: {R.shape}")
    print(f"  VOL shape: {VOL.shape}")
    print(f"  Timestamps: {len(timestamps)} from {timestamps[0]} to {timestamps[-1]}")
    print(f"  Tickers: {ticker_order}")
    
    return X, R, VOL, timestamps, ticker_order



if __name__ == "__main__":
    print("\n" + "="*70)
    print("PPOAgentManager Demo with Synthetic Data")
    print("="*70)
    
    # Create synthetic data for testing
    X_synthetic, R_synthetic, VOL_synthetic, timestamps_synthetic, tickers_synthetic = create_synthetic_state_tensor(
        n_samples=500,   # Small dataset for quick training
        n_pairs=3,       # 3 asset pairs
        n_features=15,   # 15 features per pair
        lookback=30      # 30-period lookback
    )
    
    # Split into train/val/test
    n_train = int(0.7 * len(X_synthetic))
    n_val = int(0.15 * len(X_synthetic))
    
    train_mask = np.zeros(len(X_synthetic), dtype=bool)
    val_mask = np.zeros(len(X_synthetic), dtype=bool)
    test_mask = np.zeros(len(X_synthetic), dtype=bool)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    
    print("\n📊 Data splits:")
    print(f"  Train: {train_mask.sum()} samples ({train_mask.sum()/len(X_synthetic)*100:.1f}%)")
    print(f"  Val: {val_mask.sum()} samples ({val_mask.sum()/len(X_synthetic)*100:.1f}%)")
    print(f"  Test: {test_mask.sum()} samples ({test_mask.sum()/len(X_synthetic)*100:.1f}%)")
    
    # Create manager
    manager = PPOAgentManager(config=CONFIG)
    
    # Prepare output directory
    demo_models_dir = os.path.join(SCRIPT_DIR, "models", "demo_synthetic")
    ensure_dir(demo_models_dir)
    
    print("\n🚀 Starting training with synthetic data...")
    print("   Training for 10,000 timesteps (fast demo)")
    
    # Train on synthetic data
    try:
        # Create environments directly from synthetic tensors
        X_train = X_synthetic[train_mask]
        R_train = R_synthetic[train_mask]
        VOL_train = VOL_synthetic[train_mask]
        
        X_val = X_synthetic[val_mask]
        R_val = R_synthetic[val_mask]
        VOL_val = VOL_synthetic[val_mask]
        
        print(f"\n✓ Training data ready: X{X_train.shape}, R{R_train.shape}, VOL{VOL_train.shape}")
        print(f"✓ Validation data ready: X{X_val.shape}, R{R_val.shape}, VOL{VOL_val.shape}")
        
        # Create environments
        train_env = manager._create_env(X_train, R_train, VOL_train, tickers_synthetic, "demo_train")
        val_env = manager._create_env(X_val, R_val, VOL_val, tickers_synthetic, "demo_val")
        
        vec_train = DummyVecEnv([lambda: train_env])
        vec_val = DummyVecEnv([lambda: val_env])
        
        # Create and train PPO model
        print("\nInitializing PPO model...")
        from stable_baselines3 import PPO
        
        model = PPO(
            policy="MlpPolicy",
            env=vec_train,
            gamma=0.99,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.01,
            tensorboard_log=None,
            device="cpu",
            verbose=0
        )
        
        # Setup eval callback
        eval_callback = EvalCallback(
            vec_val,
            best_model_save_path=demo_models_dir,
            log_path=demo_models_dir,
            eval_freq=2000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=0
        )
        
        # Train
        print("Training for 10,000 timesteps...")
        model.learn(total_timesteps=10_000, callback=eval_callback, progress_bar=True)
        
        # Save final model
        final_model_path = os.path.join(demo_models_dir, "final_demo_model.zip")
        model.save(final_model_path)
        
        print("\n✅ Training complete!")
        print(f"   Model saved to: {demo_models_dir}")
        
        # Test inference on a single observation
        print("\n🔮 Testing inference...")
        
        # Get test environment to properly format observation
        X_test = X_synthetic[test_mask]
        R_test = R_synthetic[test_mask]
        VOL_test = VOL_synthetic[test_mask]
        
        test_env = manager._create_env(X_test, R_test, VOL_test, tickers_synthetic, "demo_test")
        
        # Reset environment to get a properly formatted observation
        test_obs, _ = test_env.reset()
        test_timestamp = timestamps_synthetic[test_mask][0]
        
        action, _ = model.predict(test_obs, deterministic=True)
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        print("\n📈 Inference Result:")
        print(f"   Timestamp: {test_timestamp}")
        print(f"   Action: {action_value:.4f}")
        print(f"   Expected returns: {R_synthetic[test_mask][0]}")
        print(f"   Volatility: {VOL_synthetic[test_mask][0][0]:.6f}")
        
        # Interpret action
        if action_value > 0.5:
            position = "LONG asset1 / SHORT asset2"
        elif action_value < -0.5:
            position = "SHORT asset1 / LONG asset2"
        else:
            position = "NEUTRAL (mostly cash)"
        print(f"   Position: {position}")
        
        print("\n" + "="*70)
        print("✅ Demo completed successfully!")
        print("="*70)
        print("\n💡 Next steps:")
        print("   1. Replace synthetic data with real data using manager.train()")
        print("   2. Use manager.predict() for single-timestamp inference")
        print("   3. Use manager.batch_predict() for multiple predictions")
        print("   4. Check the notebook for full training pipeline examples")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis is expected if environment setup needs adjustment.")