"""Forecast accuracy metrics.

MAE and RMSE are in the dataset's own (scaled) traffic units, so they are only
comparable *within* one area. MAPE is unit-free and therefore comparable across areas,
but it is unstable when actual values approach zero and it penalises over- and
under-prediction asymmetrically (Hyndman & Koehler, 2006), so all three are reported
together rather than any one alone.
"""
import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Slots with actual == 0 are skipped."""
    usable = actual != 0
    return float(np.mean(np.abs((actual[usable] - predicted[usable]) / actual[usable])) * 100)


def evaluate(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    return {"MAE": mae(actual, predicted),
            "RMSE": rmse(actual, predicted),
            "MAPE": mape(actual, predicted)}


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Tidy table of one row per (area, model) with metrics and timings."""
    return pd.DataFrame(rows).sort_values(["square_id", "MAE"]).reset_index(drop=True)
