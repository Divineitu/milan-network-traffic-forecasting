"""Model 2 (machine learning): LightGBM on lag and calendar features.

The exploratory analysis showed strongly nonlinear calendar effects: Brera's weekend
traffic is a third of its weekday traffic while the Duomo's is higher, and holidays behave
like Sundays. Gradient-boosted trees capture such interactions (hour x day-of-week x area
behaviour) without them being specified in advance, and were the strongest family in the
M5 competition (Makridakis et al., 2022). LightGBM is histogram-based, so it trains in
seconds on a CPU (Ke et al., 2017), which matters on this hardware.

Input representation: one row per time step t, predicting t+1, with
  * lags 1, 2, 3 (PACF significant at 1-2), 6 (one hour), 144 (a day), 1008 (a week),
  * rolling mean/std over 1 hour and 6 hours,
  * the first difference,
  * calendar columns (slot of day, weekday, weekend/holiday flags, cyclical encodings).
No scaling is applied: trees split on order, so monotone rescaling cannot change them.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import SEED
from models.dataset import supervised_frame

STEP = pd.Timedelta("10min")


class LightGbmForecaster:
    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.05,
                 n_estimators: int = 400, min_child_samples: int = 20,
                 feature_fraction: float = 0.9) -> None:
        self.params = dict(num_leaves=num_leaves, learning_rate=learning_rate,
                           n_estimators=n_estimators, min_child_samples=min_child_samples,
                           feature_fraction=feature_fraction)
        self.name = f"LightGBM(leaves {num_leaves}, lr {learning_rate}, trees {n_estimators})"
        self._model: lgb.LGBMRegressor | None = None
        self._features: list[str] = []

    def fit(self, history: pd.Series) -> "LightGbmForecaster":
        frame = supervised_frame(history).dropna()
        self._features = [c for c in frame.columns if c != "target"]
        self._model = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **self.params)
        self._model.fit(frame[self._features], frame["target"])
        return self

    def predict(self, series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
        """Predict each target timestamp from features known one step earlier."""
        if self._model is None:
            raise RuntimeError("fit() must be called first")
        frame = supervised_frame(series)
        rows = frame.loc[target_index - STEP, self._features]   # features at time t -> target t+1
        return self._model.predict(rows)

    @property
    def importances(self) -> pd.Series:
        """Which inputs the model actually used (gain-based), for the discussion."""
        return pd.Series(self._model.booster_.feature_importance("gain"),
                         index=self._features).sort_values(ascending=False)
