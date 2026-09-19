"""Phase 5: where, when and why the models fail on the test week.

Reads the predictions written by run_final.py and produces the evidence for the
comparative analysis and the failure analysis:

  1. Skill against persistence per area  -> does the ranking change with area character?
  2. Error by day of the test week        -> does accuracy degrade as Christmas approaches?
  3. Error by hour of day                 -> which parts of the daily cycle are hard?
  4. The worst 6-hour window per area     -> a concrete failure case to show and explain
  5. Diebold-Mariano tests                -> are differences between models statistically real?

Usage:
    python src/experiments/analyse_errors.py
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loading                                   # noqa: E402
from eda import plotting                                   # noqa: E402

RESULTS = Path("experiments/results")
MODELS = ["ARIMA+Fourier", "LightGBM", "LSTM"]
WORST_WINDOW = 36                     # 6 hours of 10-minute slots


def load_predictions(square: int) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / f"predictions_{square}.csv", index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert("Europe/Rome")
    return frame


def skill_table(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """MAE skill score vs persistence: 1 - MAE_model / MAE_persistence (higher is better).

    Raw MAE cannot be compared across areas with different traffic levels; a skill score can,
    because each area is measured against its own naive benchmark.
    """
    rows = []
    for square, frame in frames.items():
        reference = (frame["actual"] - frame["Persistence"]).abs().mean()
        for model in MODELS:
            mae = (frame["actual"] - frame[model]).abs().mean()
            rows.append({"square_id": square, "model": model, "MAE": mae,
                         "persistence_MAE": reference, "skill_vs_persistence": 1 - mae / reference})
    return pd.DataFrame(rows)


def daily_errors(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """MAE per calendar day of the test week, to expose drift towards Christmas."""
    rows = []
    for square, frame in frames.items():
        for day, block in frame.groupby(frame.index.date):
            row = {"square_id": square, "day": pd.Timestamp(day).strftime("%a %d"),
                   "mean_traffic": block["actual"].mean()}
            for model in MODELS + ["Persistence"]:
                row[model] = (block["actual"] - block[model]).abs().mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_daily_errors(daily: pd.DataFrame) -> None:
    squares = daily["square_id"].unique()
    fig, axes = plt.subplots(1, len(squares), figsize=(12, 3.4), sharey=False)
    for axis, square in zip(np.atleast_1d(axes), squares):
        block = daily[daily["square_id"] == square]
        for model, colour in zip(MODELS, plotting.PALETTE):
            axis.plot(block["day"], block[model], marker="o", markersize=4, color=colour, label=model)
        axis.plot(block["day"], block["Persistence"], linestyle=":", color=plotting.MUTED,
                  label="Persistence")
        axis.set_title(f"Square {square}", loc="left")
        axis.set_ylabel("MAE")
        axis.tick_params(axis="x", labelrotation=45)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Error by day of the test week", y=1.02, fontsize=11)
    plotting.save(fig, "fig8_error_by_day")


def plot_hourly_errors(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Absolute error by hour of day, pooled over the three areas after normalising by area mean."""
    pooled = []
    for square, frame in frames.items():
        scale = frame["actual"].mean()
        errors = pd.DataFrame({m: (frame["actual"] - frame[m]).abs() / scale
                               for m in MODELS + ["Persistence"]}, index=frame.index)
        errors["hour"] = frame.index.hour
        pooled.append(errors)
    by_hour = pd.concat(pooled).groupby("hour").mean() * 100

    fig, axis = plt.subplots(figsize=(8, 3.4))
    for model, colour in zip(MODELS, plotting.PALETTE):
        axis.plot(by_hour.index, by_hour[model], color=colour, marker="o", markersize=3, label=model)
    axis.plot(by_hour.index, by_hour["Persistence"], color=plotting.MUTED, linestyle=":",
              label="Persistence")
    axis.set_xticks(range(0, 24, 2))
    axis.set_xlabel("Hour of day (Milan local time)")
    axis.set_ylabel("Mean absolute error (% of area mean)")
    axis.set_title("When in the day are forecasts hardest? (three areas pooled)", loc="left")
    axis.legend(ncol=2, fontsize=8)
    plotting.save(fig, "fig9_error_by_hour")
    return by_hour


def worst_window(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """For each area, the 6-hour stretch with the largest error of the best overall model."""
    rows = []
    for square, frame in frames.items():
        mean_errors = {m: (frame["actual"] - frame[m]).abs().mean() for m in MODELS}
        best = min(mean_errors, key=mean_errors.get)
        rolling = (frame["actual"] - frame[best]).abs().rolling(WORST_WINDOW).mean()
        end = rolling.idxmax()
        start = frame.index[frame.index.get_loc(end) - WORST_WINDOW + 1]
        window = frame.loc[start:end]
        rows.append({"square_id": square, "best_model": best, "start": start, "end": end,
                     **{f"{m}_MAE": (window["actual"] - window[m]).abs().mean()
                        for m in MODELS + ["Persistence"]}})
    return pd.DataFrame(rows)


def plot_worst_window(frames: dict[int, pd.DataFrame], worst: pd.DataFrame) -> None:
    """Zoom on the worst window of the busiest area, with a day of context either side."""
    case = worst.iloc[0]
    frame = frames[case["square_id"]]
    context = pd.Timedelta(hours=9)
    view = frame.loc[case["start"] - context: case["end"] + context]

    fig, axis = plt.subplots(figsize=(10, 3.6))
    axis.axvspan(case["start"], case["end"], color=plotting.GRID, alpha=0.6, linewidth=0,
                 label="worst 6-hour window")
    axis.plot(view.index, view["actual"], color=plotting.INK, linewidth=1.4, label="Actual")
    for model, colour in zip(MODELS, plotting.PALETTE):
        axis.plot(view.index, view[model], color=colour, linewidth=1.0, label=model)
    axis.set_title(f"Failure case: square {case['square_id']}, "
                   f"{case['start']:%a %d %b %H:%M} – {case['end']:%H:%M}", loc="left")
    axis.set_ylabel("Internet traffic")
    axis.legend(ncol=3, fontsize=8)
    plotting.save(fig, "fig10_failure_case")


def diebold_mariano(actual: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss, with a Newey-West variance.

    H0: the two forecasts are equally accurate. The loss differential of consecutive
    10-minute forecasts is autocorrelated, so a plain t-test would overstate significance;
    the Newey-West estimator corrects for that.
    """
    differential = (actual - first) ** 2 - (actual - second) ** 2
    n = differential.size
    lag = int(np.floor(n ** (1 / 3)))
    centred = differential - differential.mean()
    variance = centred @ centred / n
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        variance += 2 * weight * (centred[k:] @ centred[:-k]) / n
    statistic = differential.mean() / np.sqrt(variance / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
    return float(statistic), float(p_value)


def significance_table(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for square, frame in frames.items():
        actual = frame["actual"].to_numpy()
        for first, second in combinations(MODELS + ["Persistence"], 2):
            statistic, p_value = diebold_mariano(actual, frame[first].to_numpy(),
                                                 frame[second].to_numpy())
            better = second if statistic > 0 else first
            rows.append({"square_id": square, "comparison": f"{first} vs {second}",
                         "DM_statistic": statistic, "p_value": p_value,
                         "significant_5pct": p_value < 0.05,
                         "more_accurate": better if p_value < 0.05 else "no significant difference"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--squares", type=int, nargs="*", help="default: the three busiest areas")
    args = parser.parse_args()

    plotting.use_style()
    squares = args.squares or loading.top_squares(3)
    frames = {square: load_predictions(square) for square in squares}
    fmt = lambda v: f"{v:,.3f}"                            # noqa: E731

    skill = skill_table(frames)
    skill.to_csv(RESULTS / "analysis_skill.csv", index=False)
    print("== Skill vs persistence (1 - MAE_model / MAE_persistence) ==")
    print(skill.pivot(index="model", columns="square_id", values="skill_vs_persistence")
          .to_string(float_format=fmt))

    daily = daily_errors(frames)
    daily.to_csv(RESULTS / "analysis_daily_errors.csv", index=False)
    print("\n== MAE by day of the test week ==")
    print(daily.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    plot_daily_errors(daily)

    hourly = plot_hourly_errors(frames)
    hourly.to_csv(RESULTS / "analysis_hourly_errors.csv")

    worst = worst_window(frames)
    worst.to_csv(RESULTS / "analysis_worst_windows.csv", index=False)
    print("\n== Worst 6-hour window per area ==")
    print(worst.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    plot_worst_window(frames, worst)

    significance = significance_table(frames)
    significance.to_csv(RESULTS / "analysis_diebold_mariano.csv", index=False)
    print("\n== Diebold-Mariano tests (squared error; H0 = equal accuracy) ==")
    print(significance.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
