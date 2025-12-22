from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from stable_baselines3 import PPO
except Exception as e:  # pragma: no cover
    PPO = None  # type: ignore


PAIR_FEATURES_ORDER = [
    "alpha",
    "beta",
    "corr",
    "pval",
    "spreadNorm",
    "spreadNormKalman",
    "spreadNormMa",
    "spreadNormVol",
]


def continuous_action_to_weights(action: float) -> tuple[float, float]:
    action = float(np.clip(action, -1.0, 1.0))
    position_size = action * 0.5
    return position_size, -position_size


@dataclass
class PPOModelRunner:
    model_path: Path
    _model: Optional[object] = None

    def load(self) -> None:
        if PPO is None:
            raise RuntimeError("stable-baselines3 is not installed. Install requirements.txt.")
        if self._model is None:
            self._model = PPO.load(str(self.model_path))

    def predict_weights(self, pair_features: dict[str, list[float]], lookback: int) -> tuple[float, float, float]:
        """Returns (w1, w2, raw_action)."""
        self.load()
        obs = np.zeros((len(PAIR_FEATURES_ORDER), lookback), dtype=np.float32)
        for i, feat in enumerate(PAIR_FEATURES_ORDER):
            obs[i, :] = np.asarray(pair_features[feat], dtype=np.float32)
        obs = obs.reshape(-1).astype(np.float32)
        obs = np.clip(obs, -5.0, 5.0)
        action, _ = self._model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        w1, w2 = continuous_action_to_weights(a)
        return float(w1), float(w2), a
