"""Time-series characterisation of the busiest area, to inform the forecasting design.

Covers Task 2's two additional analyses, on the area with the highest total traffic:
  A. Autocorrelation (ACF/PACF): which past observations carry information about the
     next one, which determines the input window the models are given.
  B. Decomposition and stationarity (MSTL, ADF, KPSS): how much of the signal is
     seasonal structure versus residual noise, and what transformations a model that
     assumes stationarity would need.

It also quantifies how predictable each area is, using the size of one-step changes,
which is the evidence behind "which area is hardest to forecast".

Usage:
    python src/eda/temporal_analysis.py
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loading                      # noqa: E402
from eda import plotting                      # noqa: E402

SLOTS_PER_DAY = 144          # 10-minute slots
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY
REFERENCE_SQUARES = [4159, 4556]
RESULTS = Path("experiments/results")


def fill_gaps(series: pd.Series) -> pd.Series:
    """Fill the few missing slots by time interpolation, so tests needing a gap-free series can run.

    Kept explicit (rather than hidden in preprocessing) because interpolation invents data:
    it is acceptable for 0.04% of slots, but the choice must be visible and reported.
    """
    missing = int(series.isna().sum())
    if missing:
        print(f"  interpolated {missing} missing slots ({missing / len(series) * 100:.3f}%)")
    return series.interpolate(method="time").bfill().ffill()


def plot_autocorrelation(series: pd.Series, square: int) -> pd.DataFrame:
    """Figure 4: ACF over two weeks of lags, a zoom on short lags, and the PACF."""
    long_lags = 2 * SLOTS_PER_WEEK
    acf_long = acf(series, nlags=long_lags, fft=True)
    acf_short = acf(series, nlags=36, fft=True)
    pacf_short = pacf(series, nlags=36, method="ywm")
    confidence = 1.96 / np.sqrt(len(series))   # 95% band for "no correlation"

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    lags_hours = np.arange(len(acf_long)) / 6
    axes[0].plot(lags_hours, acf_long, color=plotting.PALETTE[0], linewidth=0.9)
    for day in range(1, 15):
        axes[0].axvline(day * 24, color=plotting.MUTED, linestyle=":", linewidth=0.6)
    axes[0].set_xlabel("Lag (hours)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_title("(a) ACF over 14 days of lags")

    axes[1].stem(np.arange(len(acf_short)) * 10, acf_short, basefmt=" ",
                 linefmt=plotting.PALETTE[0], markerfmt="o")
    axes[1].axhspan(-confidence, confidence, color=plotting.GRID, alpha=0.7, linewidth=0)
    axes[1].set_xlabel("Lag (minutes)")
    axes[1].set_title("(b) ACF, first 6 hours")

    axes[2].stem(np.arange(len(pacf_short)) * 10, pacf_short, basefmt=" ",
                 linefmt=plotting.PALETTE[1], markerfmt="o")
    axes[2].axhspan(-confidence, confidence, color=plotting.GRID, alpha=0.7, linewidth=0)
    axes[2].set_xlabel("Lag (minutes)")
    axes[2].set_title("(c) PACF, first 6 hours")
    fig.suptitle(f"Square {square}: temporal dependence", y=1.02, fontsize=11)
    plotting.save(fig, "fig4_autocorrelation")

    summary = pd.DataFrame({
        "lag_label": ["10 min (1 step)", "20 min", "1 hour", "12 hours",
                      "1 day (144)", "1 week (1008)", "2 weeks (2016)"],
        "lag_steps": [1, 2, 6, 72, SLOTS_PER_DAY, SLOTS_PER_WEEK, 2 * SLOTS_PER_WEEK],
    })
    summary["autocorrelation"] = [acf_long[k] for k in summary["lag_steps"]]
    return summary


def plot_decomposition(series: pd.Series, square: int) -> pd.Series:
    """Figure 5: MSTL split into trend, daily season, weekly season and remainder."""
    result = MSTL(series, periods=(SLOTS_PER_DAY, SLOTS_PER_WEEK)).fit()
    parts = {
        "observed": series,
        "trend": result.trend,
        f"daily season ({SLOTS_PER_DAY})": result.seasonal.iloc[:, 0],
        f"weekly season ({SLOTS_PER_WEEK})": result.seasonal.iloc[:, 1],
        "remainder": result.resid,
    }
    fig, axes = plt.subplots(len(parts), 1, figsize=(10, 9), sharex=True)
    for axis, (name, values), colour in zip(axes, parts.items(), plotting.PALETTE):
        axis.plot(values.index, values, color=colour, linewidth=0.7)
        axis.set_title(name, loc="left")
    axes[-1].set_xlabel("Milan local time")
    fig.suptitle(f"Square {square}: MSTL decomposition", y=0.995, fontsize=11)
    plotting.save(fig, "fig5_decomposition")

    # Share of variance explained by each component: how much structure is learnable.
    total = series.var()
    return pd.Series({
        "var_trend_share": result.trend.var() / total,
        "var_daily_share": result.seasonal.iloc[:, 0].var() / total,
        "var_weekly_share": result.seasonal.iloc[:, 1].var() / total,
        "var_remainder_share": result.resid.var() / total,
        "remainder_std": result.resid.std(),
        "remainder_std_over_mean": result.resid.std() / series.mean(),
    })


def stationarity_table(series: pd.Series) -> pd.DataFrame:
    """ADF and KPSS on the raw series and on two differenced versions.

    The two tests ask opposite questions, so they are reported together:
      ADF  H0 = the series has a unit root (non-stationary); small p-value => stationary.
      KPSS H0 = the series is stationary;                    small p-value => non-stationary.
    """
    variants = {
        "raw": series,
        "first difference (lag 1)": series.diff().dropna(),
        "seasonal difference (lag 144)": series.diff(SLOTS_PER_DAY).dropna(),
    }
    rows = []
    for name, values in variants.items():
        with warnings.catch_warnings():
            # KPSS warns when the statistic falls outside its p-value lookup table (p is then
            # capped at 0.10); statsmodels also warns about a planned change to its return type.
            warnings.simplefilter("ignore")
            adf_stat, adf_p = adfuller(values, autolag="AIC")[:2]
            kpss_stat, kpss_p = kpss(values, regression="c", nlags="auto")[:2]
        rows.append({
            "series": name,
            "adf_stat": adf_stat, "adf_p": adf_p,
            "adf_says": "stationary" if adf_p < 0.05 else "non-stationary",
            "kpss_stat": kpss_stat, "kpss_p": kpss_p,
            "kpss_says": "non-stationary" if kpss_p < 0.05 else "stationary",
        })
    return pd.DataFrame(rows).set_index("series")


def predictability_table(squares: list[int], processed: Path) -> pd.DataFrame:
    """How hard is each area to forecast, before any model is trained?

    The size of a one-step change is what a persistence forecast would get wrong, so
    it measures the unpredictable part of the signal, unlike the raw standard deviation
    which is dominated by the (highly predictable) daily cycle.
    """
    frame = loading.get_series(squares, processed)
    rows = []
    for square in squares:
        values = fill_gaps(frame[square])
        changes = values.diff().abs().dropna()
        rows.append({
            "square_id": square,
            "mean": values.mean(),
            "std": values.std(),
            "coef_variation": values.std() / values.mean(),
            "mean_abs_change": changes.mean(),
            "mean_abs_change_pct": changes.mean() / values.mean() * 100,
            "persistence_over_daily_swing": changes.mean() / (values.max() - values.min()) * 100,
        })
    return pd.DataFrame(rows).set_index("square_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", default="data/processed", type=Path)
    args = parser.parse_args()

    plotting.use_style()
    RESULTS.mkdir(parents=True, exist_ok=True)

    top3 = loading.top_squares(3, args.processed)
    target = top3[0]
    print(f"== Time-series analysis of square {target} (highest total traffic) ==")
    series = fill_gaps(loading.get_series(target, args.processed)[target])

    acf_summary = plot_autocorrelation(series, target)
    print("\n-- Autocorrelation at key lags --")
    print(acf_summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    acf_summary.to_csv(RESULTS / "eda_autocorrelation.csv", index=False)

    variance = plot_decomposition(series, target)
    print("\n-- Variance explained by each component --")
    print(variance.to_string(float_format=lambda v: f"{v:,.3f}"))
    variance.to_csv(RESULTS / "eda_decomposition.csv", header=["value"])

    stationarity = stationarity_table(series)
    print("\n-- Stationarity tests --")
    print(stationarity.to_string(float_format=lambda v: f"{v:,.4f}"))
    stationarity.to_csv(RESULTS / "eda_stationarity.csv")

    predictability = predictability_table(top3 + REFERENCE_SQUARES, args.processed)
    print("\n-- Predictability of each area (whole period) --")
    print(predictability.to_string(float_format=lambda v: f"{v:,.2f}"))
    predictability.to_csv(RESULTS / "eda_predictability.csv")


if __name__ == "__main__":
    main()
