"""Hyperparameter experiments, scored on the validation week (Dec 9-15).

The test week is never used here. Each run records its parameters, validation error and
training time to experiments/results/tuning_<model>.csv, which is the evidence behind the
"iterative experimentation" section of the report.

Search strategy differs per model, by design:
  * arima : staged search (seasonal harmonics first, then ARIMA orders), because fitting
            a state-space model is slow and the two choices are largely independent.
  * lgbm  : grid search, since each fit takes seconds.
  * lstm  : a hand-built sequence of configurations, each motivated by the previous result
            (the reasoning is in the `rationale` column).

Usage:
    python src/experiments/tune.py --model arima
    python src/experiments/tune.py --model lgbm
    python src/experiments/tune.py --model lstm
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import loading                                   # noqa: E402
from evaluation.metrics import evaluate                    # noqa: E402
from models.arima_fourier import ArimaFourierForecaster    # noqa: E402
from models.baselines import persistence                   # noqa: E402
from models.dataset import split_series                    # noqa: E402
from models.lgbm import LightGbmForecaster                 # noqa: E402
from models.lstm import LstmForecaster                     # noqa: E402

RESULTS = Path("experiments/results")


def score(model, split, series, rationale: str = "") -> dict:
    """Fit on the training data, score one-step-ahead forecasts on the validation week."""
    started = time.perf_counter()
    model.fit(split.train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predicted = model.predict(series, split.val.index)
    predict_seconds = time.perf_counter() - started

    row = {"model": model.name, **evaluate(split.val.to_numpy(), predicted),
           "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
           "rationale": rationale}
    print(f"{row['model']:62s} MAE {row['MAE']:7.2f}  RMSE {row['RMSE']:7.2f}  "
          f"MAPE {row['MAPE']:5.2f}%  fit {fit_seconds:6.1f}s", flush=True)
    return row


def tune_arima(split, series) -> list[dict]:
    rows = [score(persistence(), split, series, "reference: no parameters")]

    # Stage 1: how much seasonal detail is needed? ARIMA order held at a simple AR(2).
    best_harmonics, best_mae = (4, 2), float("inf")
    for harmonics_day in (3, 4, 6):
        for harmonics_week in (1, 2):
            model = ArimaFourierForecaster((2, 0, 0), harmonics_day, harmonics_week)
            row = score(model, split, series,
                        "stage 1: seasonal detail, ARIMA order fixed at (2,0,0)")
            rows.append(row)
            if row["MAE"] < best_mae:
                best_mae, best_harmonics = row["MAE"], (harmonics_day, harmonics_week)
    print(f"-> best harmonics {best_harmonics} (MAE {best_mae:.2f})\n")

    # Stage 2: with seasonality fixed, how much short-term structure does the ARIMA need?
    for order in [(1, 0, 0), (2, 0, 0), (3, 0, 0), (2, 0, 1), (3, 0, 1), (2, 1, 1)]:
        model = ArimaFourierForecaster(order, *best_harmonics)
        rows.append(score(model, split, series,
                          f"stage 2: ARIMA order with harmonics {best_harmonics}"))

    # Stage 3: stage 2 improved monotonically up to AR(3), the edge of that grid, so the
    # search is extended upwards to find where the gain actually stops.
    for order in [(4, 0, 0), (5, 0, 0), (6, 0, 0), (8, 0, 0)]:
        model = ArimaFourierForecaster(order, *best_harmonics)
        rows.append(score(model, split, series,
                          "stage 3: higher AR orders, since stage 2 was still improving at p=3"))
    return rows


def tune_lgbm(split, series) -> list[dict]:
    rows = []
    # Round 1: a coarse grid over capacity and step size.
    for learning_rate in (0.03, 0.05, 0.1):
        for num_leaves in (15, 31, 63):
            for n_estimators in (200, 400):
                model = LightGbmForecaster(num_leaves=num_leaves, learning_rate=learning_rate,
                                           n_estimators=n_estimators)
                rows.append(score(model, split, series,
                                  "round 1: coarse grid over capacity (leaves, trees) and step size"))

    # Round 2: round 1 showed error rising with capacity (more leaves and more trees were
    # consistently worse), i.e. overfitting. So search *below* the previous best rather
    # than around it, and add stronger regularisation.
    for num_leaves in (7, 15):
        for n_estimators in (100, 200, 300):
            model = LightGbmForecaster(num_leaves=num_leaves, learning_rate=0.03,
                                       n_estimators=n_estimators, min_child_samples=40,
                                       feature_fraction=0.7)
            rows.append(score(model, split, series,
                              "round 2: smaller capacity + stronger regularisation, "
                              "because round 1 showed overfitting"))
    return rows


def tune_lstm(split, series) -> list[dict]:
    """Manual, iterative search: each configuration answers a question raised by the last.

    Background (diagnostic run, see experiments/log.md): predicting the *level* with 40 full
    epochs gave validation MAE 123.4, worse than persistence (116.8), and the internal early-
    stopping loss was noisy, so short patience stopped training around epoch 10.
    """
    rows = []

    def run(settings: dict, rationale: str) -> dict:
        model = LstmForecaster(**settings)
        row = score(model, split, series, rationale)
        row.update(epochs_run=model.epochs_run, best_epoch=model.best_epoch,
                   parameters=model.n_parameters)
        rows.append(row)
        return row

    # Stage 1: what should the network predict? This is the largest design decision.
    base = dict(hidden=64, layers=1, dropout=0.0, learning_rate=1e-3)
    level = run({**base, "target": "level"},
                "stage 1: predict the level x(t+1) directly (the original formulation)")
    delta = run({**base, "target": "delta"},
                "stage 1: predict the change x(t+1)-x(t), anchored on persistence")
    target = "delta" if delta["MAE"] <= level["MAE"] else "level"
    print(f"-> target formulation: {target}\n")

    # Stage 2: capacity, depth, step size and window, each varied from the stage-1 winner.
    plan = [
        (dict(hidden=32), "capacity down: is 64 units more than the data supports?"),
        (dict(hidden=128), "capacity up: does more width still help?"),
        (dict(layers=2, dropout=0.2), "depth instead of width, with dropout against overfitting"),
        (dict(learning_rate=3e-3), "faster learning rate: better convergence within budget?"),
        (dict(learning_rate=3e-4), "slower learning rate: is the default overshooting?"),
        (dict(window=36), "shorter window (6 h): the PACF says recent steps dominate"),
        (dict(window=288), "longer window (48 h): does a second daily cycle help?"),
    ]
    for change, rationale in plan:
        run({**base, "target": target, **change}, f"stage 2 ({target}): {rationale}")
    return rows


TUNERS = {"arima": tune_arima, "lgbm": tune_lgbm, "lstm": tune_lstm}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=list(TUNERS), required=True)
    parser.add_argument("--square", type=int, help="default: the busiest area")
    args = parser.parse_args()

    square = args.square or loading.top_squares(1)[0]
    series = loading.get_series(square)[square].interpolate()
    split = split_series(series, square)
    print(f"Tuning {args.model} on square {square}; validation = "
          f"{split.val.index[0]:%Y-%m-%d} to {split.val.index[-1]:%Y-%m-%d}\n")

    rows = TUNERS[args.model](split, series)
    table = pd.DataFrame(rows).sort_values("MAE")
    table.insert(0, "square_id", square)

    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / f"tuning_{args.model}.csv"
    table.to_csv(destination, index=False)
    print(f"\nBest configuration:\n{table.iloc[0].to_string()}")
    print(f"\nSaved {len(table)} runs -> {destination}")


if __name__ == "__main__":
    main()
