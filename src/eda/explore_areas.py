"""Exploratory analysis of the spatial and temporal structure of Milan internet traffic.

Covers Task 2, items 1 and 2:
  1. how total traffic is distributed across the 10,000 areas;
  2. the time series of the three busiest areas plus squares 4159 and 4556
     during the first two weeks, and how their temporal dynamics compare.

Outputs figures to figures/ and summary statistics to experiments/results/.

Usage:
    python src/eda/explore_areas.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loading                      # noqa: E402
from eda import plotting                      # noqa: E402

REFERENCE_SQUARES = [4159, 4556]   # required by the assignment
WEEKS_SHOWN = 2
RESULTS = Path("experiments/results")


def gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = every area identical, 1 = all traffic in one area."""
    ordered = np.sort(values)
    n = ordered.size
    rank = np.arange(1, n + 1)
    return float((2 * (rank * ordered).sum()) / (n * ordered.sum()) - (n + 1) / n)


def describe_distribution(totals: pd.DataFrame) -> pd.Series:
    """Headline statistics of the across-area traffic distribution."""
    values = totals["total_traffic"].to_numpy()
    ordered = np.sort(values)[::-1]
    share = ordered.cumsum() / ordered.sum()
    n = values.size
    return pd.Series({
        "areas": n,
        "mean": values.mean(),
        "median": np.median(values),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "max_over_median": values.max() / np.median(values),
        "skewness": pd.Series(values).skew(),
        "share_top_1pct": share[int(0.01 * n) - 1],
        "share_top_5pct": share[int(0.05 * n) - 1],
        "share_top_10pct": share[int(0.10 * n) - 1],
        "gini": gini(values),
        "total_missing_slots": totals["missing_slots"].sum(),
        "areas_with_missing": int((totals["missing_slots"] > 0).sum()),
    })


def plot_distribution(totals: pd.DataFrame) -> None:
    """Figure 1: how unevenly traffic is spread over the city."""
    values = totals["total_traffic"].to_numpy()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: magnitudes span orders of magnitude, so bin on a log scale.
    left.hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), 60),
              color=plotting.PALETTE[0], edgecolor="white", linewidth=0.3)
    left.set_xscale("log")
    left.set_xlabel("Total internet traffic over 62 days (scaled units, log scale)")
    left.set_ylabel("Number of areas")
    left.set_title("(a) Distribution across the 10,000 areas")
    left.axvline(np.median(values), color=plotting.INK, linestyle="--", linewidth=1)
    left.annotate(f"median = {np.median(values):,.0f}",
                  xy=(np.median(values), left.get_ylim()[1] * 0.9),
                  xytext=(6, 0), textcoords="offset points", color=plotting.INK)

    # Right: concentration. How much of the city's traffic do the busiest areas carry?
    ordered = np.sort(values)[::-1]
    share = ordered.cumsum() / ordered.sum()
    rank_pct = np.arange(1, ordered.size + 1) / ordered.size * 100
    right.plot(rank_pct, share * 100, color=plotting.PALETTE[1])
    right.plot([0, 100], [0, 100], color=plotting.MUTED, linestyle=":", linewidth=1)
    for mark in (1, 5, 10):
        y = share[int(mark / 100 * ordered.size) - 1] * 100
        right.plot([mark], [y], "o", color=plotting.PALETTE[1], markersize=5)
        right.annotate(f"top {mark}% of areas\ncarry {y:.0f}% of traffic",
                       xy=(mark, y), xytext=(14, -6), textcoords="offset points",
                       color=plotting.INK, fontsize=8)
    right.set_xlabel("Areas ranked by traffic (%)")
    right.set_ylabel("Cumulative share of city traffic (%)")
    right.set_title("(b) Concentration of traffic")
    plotting.save(fig, "fig1_traffic_distribution")


def plot_area_series(series: pd.DataFrame, labels: dict[int, str]) -> None:
    """Figure 2: the raw two-week series, one panel per area (scales differ hugely)."""
    fig, axes = plt.subplots(len(series.columns), 1, figsize=(10, 9), sharex=True)
    for axis, (square, colour) in zip(axes, zip(series.columns, plotting.PALETTE)):
        axis.plot(series.index, series[square], color=colour, linewidth=0.8)
        axis.set_title(f"Square {square} — {labels[square]}", loc="left")
        axis.set_ylabel("traffic")
        # Shade weekends: the weekly rhythm is easier to see than to describe.
        for day in pd.date_range(series.index[0].normalize(), series.index[-1], freq="D"):
            if day.dayofweek >= 5:
                axis.axvspan(day, day + pd.Timedelta(days=1),
                             color=plotting.GRID, alpha=0.45, linewidth=0)
    axes[-1].set_xlabel("Milan local time (grey bands = weekends)")
    fig.suptitle(f"Internet traffic, first {WEEKS_SHOWN} weeks", y=0.995, fontsize=11)
    plotting.save(fig, "fig2_area_series")


def plot_shape_comparison(series: pd.DataFrame) -> pd.DataFrame:
    """Figure 3: average daily and weekly shapes, normalised so areas can be compared."""
    normalised = series / series.mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    minutes = normalised.index.hour * 60 + normalised.index.minute
    daily = normalised.groupby(minutes).mean()
    for (square, colour) in zip(normalised.columns, plotting.PALETTE):
        left.plot(daily.index / 60, daily[square], color=colour, label=f"{square}")
    left.set_xticks(range(0, 25, 4))
    left.set_xlabel("Hour of day (Milan local time)")
    left.set_ylabel("Traffic / area's own mean")
    left.set_title("(a) Average daily shape")
    left.legend(title="Square", ncol=2, fontsize=8)

    weekly = normalised.groupby(normalised.index.dayofweek).mean()
    for (square, colour) in zip(normalised.columns, plotting.PALETTE):
        right.plot(weekly.index, weekly[square], color=colour, marker="o", markersize=4)
    right.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    right.set_ylabel("Traffic / area's own mean")
    right.set_title("(b) Average weekly shape")
    plotting.save(fig, "fig3_area_shapes")
    return daily


def area_characteristics(series: pd.DataFrame) -> pd.DataFrame:
    """Numbers behind the visual comparison: level, variability, peak time, weekend effect."""
    minutes = series.index.hour * 60 + series.index.minute
    is_weekend = series.index.dayofweek >= 5
    rows = []
    for square in series.columns:
        values = series[square]
        daily_shape = values.groupby(minutes).mean()
        peak_minute = daily_shape.idxmax()
        rows.append({
            "square_id": square,
            "mean": values.mean(),
            "std": values.std(),
            "coef_variation": values.std() / values.mean(),
            "peak_hour": f"{peak_minute // 60:02d}:{peak_minute % 60:02d}",
            "trough_hour": f"{daily_shape.idxmin() // 60:02d}:{daily_shape.idxmin() % 60:02d}",
            "peak_over_trough": daily_shape.max() / daily_shape.min(),
            "weekend_over_weekday": values[is_weekend].mean() / values[~is_weekend].mean(),
        })
    return pd.DataFrame(rows).set_index("square_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", default="data/processed", type=Path)
    args = parser.parse_args()

    plotting.use_style()
    RESULTS.mkdir(parents=True, exist_ok=True)

    totals = loading.load_square_totals(args.processed)
    stats = describe_distribution(totals)
    print("\n== Distribution of total traffic across areas ==")
    print(stats.to_string(float_format=lambda v: f"{v:,.3f}"))
    stats.to_csv(RESULTS / "eda_distribution_stats.csv", header=["value"])
    plot_distribution(totals)

    top3 = loading.top_squares(3, args.processed)
    print("\n== Ten busiest areas ==")
    print(totals.nlargest(10, "total_traffic").to_string(index=False,
                                                         float_format=lambda v: f"{v:,.0f}"))

    squares = top3 + REFERENCE_SQUARES
    labels = {square: ("top-3 traffic" if square in top3 else "assignment reference area")
              for square in squares}
    print(f"\nTop-3 areas: {top3}; reference areas: {REFERENCE_SQUARES}")

    full = loading.get_series(squares, args.processed)
    two_weeks = full.loc[full.index[0]:full.index[0] + pd.Timedelta(weeks=WEEKS_SHOWN)]
    plot_area_series(two_weeks, labels)
    plot_shape_comparison(two_weeks)

    characteristics = area_characteristics(two_weeks)
    print("\n== Temporal characteristics (first two weeks) ==")
    print(characteristics.to_string(float_format=lambda v: f"{v:,.2f}"))
    characteristics.to_csv(RESULTS / "eda_area_characteristics.csv")

    pd.Series({"top1": top3[0], "top2": top3[1], "top3": top3[2]}).to_csv(
        RESULTS / "selected_areas.csv", header=["square_id"])


if __name__ == "__main__":
    main()
