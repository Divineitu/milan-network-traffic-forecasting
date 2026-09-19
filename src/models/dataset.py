"""Turning one area's traffic series into the inputs each model needs.

All three models forecast one step ahead from the *true* history: at time t they see
observations up to and including t, and predict t+1. The three input representations
differ, which is the point of the comparison:

  * ARIMA + Fourier : the series itself, plus deterministic seasonal terms
  * LightGBM        : a table of lagged values and calendar features
  * LSTM            : a window of the last 144 scaled observations, plus calendar features
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (FEATURE_LAGS, HOLIDAYS, ROLLING_WINDOWS, SLOTS_PER_DAY,
                    SLOTS_PER_WEEK, TEST_END, TEST_START, TRAIN_START, VAL_START)


@dataclass
class Split:
    """One area's series, cut chronologically into train / validation / test."""
    square_id: int
    train: pd.Series
    val: pd.Series
    test: pd.Series

    @property
    def train_and_val(self) -> pd.Series:
        """Used to refit a tuned model before forecasting the test week."""
        return pd.concat([self.train, self.val])


def _window(series: pd.Series, start: str, end: str) -> pd.Series:
    """Half-open slice [start, end). Explicit boundaries avoid pandas' string-slicing rule
    that a date label includes the whole day, which silently overlapped the splits."""
    timezone = series.index.tz
    lower = pd.Timestamp(start, tz=timezone)
    upper = pd.Timestamp(end, tz=timezone)
    return series[(series.index >= lower) & (series.index < upper)]


def split_series(series: pd.Series, square_id: int) -> Split:
    return Split(
        square_id=square_id,
        train=_window(series, TRAIN_START, VAL_START),
        val=_window(series, VAL_START, TEST_START),
        test=_window(series, TEST_START, TEST_END),
    )


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Time-of-day / day-of-week / holiday information, as plain numeric columns.

    Hour and weekday are also encoded as sine/cosine pairs so that models treating them
    as numbers see 23:50 and 00:00 as adjacent rather than 143 steps apart.
    """
    holidays = pd.to_datetime(HOLIDAYS).date
    slot_of_day = index.hour * SLOTS_PER_HOUR_FACTOR + index.minute // 10
    frame = pd.DataFrame(index=index)
    frame["slot_of_day"] = slot_of_day
    frame["day_of_week"] = index.dayofweek
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    frame["is_holiday"] = pd.Series(index.date, index=index).isin(holidays).astype(int)
    frame["day_sin"] = np.sin(2 * np.pi * slot_of_day / SLOTS_PER_DAY)
    frame["day_cos"] = np.cos(2 * np.pi * slot_of_day / SLOTS_PER_DAY)
    frame["week_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    frame["week_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    return frame


SLOTS_PER_HOUR_FACTOR = 6   # 10-minute slots per hour, used above


def supervised_frame(series: pd.Series) -> pd.DataFrame:
    """Lag/rolling/calendar features for tree models, aligned so that row t predicts t+1.

    The target column is the value at t+1; every feature column is known at time t, which
    is what makes this a genuine one-step-ahead setup with no leakage from the future.
    """
    frame = calendar_features(series.index)
    for lag in FEATURE_LAGS:
        frame[f"lag_{lag}"] = series.shift(lag - 1)      # lag_1 = the value at time t
    for window in ROLLING_WINDOWS:
        frame[f"roll_mean_{window}"] = series.shift(0).rolling(window).mean()
        frame[f"roll_std_{window}"] = series.shift(0).rolling(window).std()
    frame["diff_1"] = series.diff()
    frame["target"] = series.shift(-1)                   # the value at t+1
    return frame


def fourier_terms(index: pd.DatetimeIndex, harmonics_day: int, harmonics_week: int) -> pd.DataFrame:
    """Deterministic seasonal regressors for the ARIMA model.

    Long seasonal periods (144, 1008) make a classical SARIMA state space impractically
    large, so seasonality is supplied as a handful of sine/cosine pairs instead
    (dynamic harmonic regression, Hyndman & Athanasopoulos 2021, ch. 12).
    """
    step = np.arange(len(index))
    terms = {}
    for k in range(1, harmonics_day + 1):
        terms[f"day_sin_{k}"] = np.sin(2 * np.pi * k * step / SLOTS_PER_DAY)
        terms[f"day_cos_{k}"] = np.cos(2 * np.pi * k * step / SLOTS_PER_DAY)
    for k in range(1, harmonics_week + 1):
        terms[f"week_sin_{k}"] = np.sin(2 * np.pi * k * step / SLOTS_PER_WEEK)
        terms[f"week_cos_{k}"] = np.cos(2 * np.pi * k * step / SLOTS_PER_WEEK)
    return pd.DataFrame(terms, index=index)


class MinMaxScaler:
    """Scale to [0, 1] using the training range only.

    Fitting the scaler on all the data would leak information about the future
    (the maximum of the test week) into training, which inflates results.
    """

    def __init__(self) -> None:
        self.low = 0.0
        self.high = 1.0

    def fit(self, values: np.ndarray) -> "MinMaxScaler":
        self.low = float(np.min(values))
        self.high = float(np.max(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.low) / (self.high - self.low)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * (self.high - self.low) + self.low


def windowed(series: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Sliding windows for the LSTM: X[i] = the `window` values ending at t, y[i] = t+1.

    Returned alongside the timestamps of the predicted values, so predictions can be
    lined up with the evaluation period.
    """
    values = series.to_numpy(dtype=np.float32)
    n = len(values) - window
    X = np.lib.stride_tricks.sliding_window_view(values, window)[:n]
    y = values[window:]
    return X, y, series.index[window:]
