"""Naive reference forecasters.

These have no parameters to fit, so they cost nothing and cannot overfit. They exist to
give every reported error a meaning: with a lag-1 autocorrelation of 0.987, persistence is
already a strong forecaster, so a model is only worth its complexity if it beats these.
"""
import numpy as np
import pandas as pd

from config import SLOTS_PER_DAY, SLOTS_PER_WEEK


class NaiveForecaster:
    """Predict x(t+1) as the observation `lag` steps before t+1."""

    def __init__(self, lag: int = 1, name: str | None = None) -> None:
        self.lag = lag
        self.name = name or f"naive(lag {lag})"

    def fit(self, history: pd.Series) -> "NaiveForecaster":
        return self                      # nothing to learn

    def predict(self, series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
        return series.shift(self.lag).loc[target_index].to_numpy()


def persistence() -> NaiveForecaster:
    """x̂(t+1) = x(t): the hardest baseline to beat at 10-minute resolution."""
    return NaiveForecaster(lag=1, name="Persistence")


def seasonal_naive_daily() -> NaiveForecaster:
    """x̂(t+1) = the value at the same time yesterday."""
    return NaiveForecaster(lag=SLOTS_PER_DAY, name="Seasonal naive (day)")


def seasonal_naive_weekly() -> NaiveForecaster:
    """x̂(t+1) = the value at the same time last week."""
    return NaiveForecaster(lag=SLOTS_PER_WEEK, name="Seasonal naive (week)")
