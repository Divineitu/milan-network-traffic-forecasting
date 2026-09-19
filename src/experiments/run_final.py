"""Final evaluation: forecast the test week (Dec 16-22) for the three busiest areas.

Protocol, fixed before any test data is touched:
  * each model uses the configuration chosen on the validation week (see BEST_CONFIGS);
  * it is refitted on train + validation (Nov 1 - Dec 15), so the forecast uses all data
    available before the test week begins;
  * forecasts are one step ahead from the true history: predicting x(t+1) uses observations
    up to t, never the model's own earlier predictions;
  * the same three areas, the same metrics and the same timing method apply to every model.

Outputs (figures/ and experiments/results/):
  fig6_forecast_<square>_<model>.png   9 plots: actual vs predicted per model per area
  fig7_forecast_overview.png           all nine panels in one figure, for the report
  final_metrics.csv                    MAE / RMSE / MAPE per area and model
  final_timing.csv                     training and inference time per model
  predictions_<square>.csv             raw predictions, for the failure analysis

Usage:
    python src/experiments/run_final.py
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loading                                   # noqa: E402
from eda import plotting                                   # noqa: E402
from evaluation.metrics import evaluate                    # noqa: E402
from models.arima_fourier import ArimaFourierForecaster    # noqa: E402
from models import baselines                               # noqa: E402
from models.dataset import split_series                    # noqa: E402
from models.lgbm import LightGbmForecaster                 # noqa: E402
from models.lstm import LstmForecaster                     # noqa: E402

RESULTS = Path("experiments/results")
TIMING_REPEATS = 5        # inference is fast, so it is timed repeatedly and averaged

# Chosen on the validation week; see experiments/results/tuning_*.csv and experiments/log.md.
BEST_CONFIGS = {
    # Validation MAE 101.26; p=3 is a genuine optimum (p=4,5,6,8 were all slightly worse).
    "ARIMA+Fourier": dict(order=(3, 0, 0), harmonics_day=6, harmonics_week=1),
    # Validation MAE 98.49; round 1 showed overfitting, so round 2 searched smaller models.
    "LightGBM": dict(num_leaves=15, learning_rate=0.03, n_estimators=300,
                     min_child_samples=40, feature_fraction=0.7),
    # Validation MAE 98.32; the delta target beat the level target by 24 MAE (103.27 vs 127.41).
    "LSTM": dict(target="delta", hidden=64, layers=1, dropout=0.0, learning_rate=3e-3),
}


def build_models() -> dict:
    """Fresh, unfitted model instances (one set per area, so no state is shared)."""
    return {
        "Persistence": baselines.persistence(),
        "Seasonal naive (week)": baselines.seasonal_naive_weekly(),
        "ARIMA+Fourier": ArimaFourierForecaster(**BEST_CONFIGS["ARIMA+Fourier"]),
        "LightGBM": LightGbmForecaster(**BEST_CONFIGS["LightGBM"]),
        "LSTM": LstmForecaster(**BEST_CONFIGS["LSTM"]),
    }


MAIN_MODELS = ["ARIMA+Fourier", "LightGBM", "LSTM"]     # the three compared models


def run_area(square: int, processed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every model on train+validation and forecast the test week for one area."""
    series = loading.get_series(square, processed)[square].interpolate()
    split = split_series(series, square)
    history = split.train_and_val
    actual = split.test.to_numpy()

    metrics, predictions = [], {"actual": pd.Series(actual, index=split.test.index)}
    for name, model in build_models().items():
        started = time.perf_counter()
        model.fit(history)
        fit_seconds = time.perf_counter() - started

        predicted = model.predict(series, split.test.index)
        started = time.perf_counter()
        for _ in range(TIMING_REPEATS):
            model.predict(series, split.test.index)
        predict_seconds = (time.perf_counter() - started) / TIMING_REPEATS

        predictions[name] = pd.Series(predicted, index=split.test.index)
        metrics.append({"square_id": square, "model": name, **evaluate(actual, predicted),
                        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                        "config": getattr(model, "name", name)})
        print(f"  {name:22s} MAE {metrics[-1]['MAE']:7.2f}  RMSE {metrics[-1]['RMSE']:7.2f}  "
              f"MAPE {metrics[-1]['MAPE']:5.2f}%  fit {fit_seconds:6.1f}s", flush=True)

    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def plot_single(square: int, model: str, frame: pd.DataFrame, metrics: pd.Series) -> None:
    """One of the nine required plots: observed and predicted traffic over the test week."""
    fig, axis = plt.subplots(figsize=(10, 3.2))
    axis.plot(frame.index, frame["actual"], color=plotting.INK, linewidth=1.0, label="Actual")
    axis.plot(frame.index, frame[model], color=plotting.PALETTE[1], linewidth=1.0,
              alpha=0.85, label="Predicted")
    axis.set_title(f"Square {square} — {model}   "
                   f"(MAE {metrics['MAE']:.1f}, RMSE {metrics['RMSE']:.1f}, "
                   f"MAPE {metrics['MAPE']:.2f}%)", loc="left")
    axis.set_xlabel("Milan local time, 16–22 December 2013")
    axis.set_ylabel("Internet traffic")
    axis.legend(ncol=2)
    plotting.save(fig, f"fig6_forecast_{square}_{model.split('+')[0].replace(' ', '_')}")


def plot_overview(predictions: dict[int, pd.DataFrame], metrics: pd.DataFrame) -> None:
    """All nine model/area combinations in one figure, for the report."""
    squares = list(predictions)
    fig, axes = plt.subplots(len(squares), len(MAIN_MODELS), figsize=(13, 8),
                             sharex=True)
    for row, square in enumerate(squares):
        frame = predictions[square]
        for column, model in enumerate(MAIN_MODELS):
            axis = axes[row, column]
            axis.plot(frame.index, frame["actual"], color=plotting.INK, linewidth=0.7)
            axis.plot(frame.index, frame[model], color=plotting.PALETTE[column],
                      linewidth=0.7, alpha=0.85)
            score = metrics[(metrics.square_id == square) & (metrics.model == model)].iloc[0]
            axis.set_title(f"{square} — {model} (MAE {score['MAE']:.0f})", loc="left", fontsize=9)
    for axis in axes[-1]:
        axis.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Actual (black) vs predicted, 16–22 December 2013", y=0.995, fontsize=11)
    plotting.save(fig, "fig7_forecast_overview")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", default="data/processed", type=Path)
    parser.add_argument("--squares", type=int, nargs="*", help="default: the three busiest areas")
    args = parser.parse_args()

    plotting.use_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    squares = args.squares or loading.top_squares(3, args.processed)

    all_metrics, all_predictions = [], {}
    for square in squares:
        print(f"\n== Square {square} ==", flush=True)
        metrics, predictions = run_area(square, args.processed)
        all_metrics.append(metrics)
        all_predictions[square] = predictions
        predictions.to_csv(RESULTS / f"predictions_{square}.csv")

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(RESULTS / "final_metrics.csv", index=False)

    for square, frame in all_predictions.items():
        for model in MAIN_MODELS:
            score = metrics[(metrics.square_id == square) & (metrics.model == model)].iloc[0]
            plot_single(square, model, frame, score)
    plot_overview(all_predictions, metrics)

    # Timing: averaged over the three areas, so one number represents the model, not an area.
    timing = (metrics.groupby("model")[["fit_seconds", "predict_seconds"]]
              .agg(["mean", "std"]).round(3))
    timing.to_csv(RESULTS / "final_timing.csv")

    print("\n== Test-week results (Dec 16-22) ==")
    for square in squares:
        table = metrics[metrics.square_id == square][["model", "MAE", "RMSE", "MAPE"]]
        print(f"\nSquare {square}:\n{table.to_string(index=False, float_format=lambda v: f'{v:,.2f}')}")
    print(f"\n== Timing (mean over {len(squares)} areas, seconds) ==\n{timing.to_string()}")
    print(f"\nSaved -> {RESULTS}")


if __name__ == "__main__":
    main()
