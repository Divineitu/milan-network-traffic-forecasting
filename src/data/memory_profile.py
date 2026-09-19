"""Measure the memory cost of loading one daily file: naive vs optimised.

This produces the before/after evidence for the data-handling section of the report.

Each variant is measured in a *fresh subprocess*, because memory freed by Python is
not reliably returned to the operating system. Measuring both variants in one process
would make whichever ran second look artificially cheap.

Usage:
    python src/data/memory_profile.py                     # run both variants, print table
    python src/data/memory_profile.py --mode naive        # internal: one variant, prints JSON
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

# The files have no header row; these are the columns documented in Barlacchi et al. (2015).
COLUMNS = ["square_id", "time", "country_code",
           "sms_in", "sms_out", "call_in", "call_out", "internet"]
KEEP = ["square_id", "time", "internet"]          # the only columns this study needs
DTYPES = {"square_id": "int16",                   # 1..10000 fits in int16 (max 32767)
          "time": "int64",                        # epoch milliseconds needs 64 bits
          "internet": "float32"}                  # ~7 significant digits is ample here
N_DAYS = 62                                       # for extrapolating to the full dataset
CHUNK_ROWS = 500_000                              # rows parsed at a time in the chunked variant


def load_naive(path: Path) -> pd.DataFrame:
    """What pandas does by default: all 8 columns, 64-bit types everywhere."""
    return pd.read_csv(path, sep="\t", header=None, names=COLUMNS)


def load_optimised(path: Path) -> pd.DataFrame:
    """Read only the needed columns with compact dtypes, then sum away the country codes."""
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS,
                     usecols=KEEP, dtype=DTYPES)
    # Several country codes share one (square, time) slot: sum them into total traffic.
    # NaN (no activity recorded) is treated as 0 by sum(), which is what we want.
    return (df.groupby(["square_id", "time"], sort=False)["internet"]
              .sum()
              .reset_index())


def load_chunked(path: Path) -> pd.DataFrame:
    """Same result as load_optimised, but never holding the whole file in memory.

    Each chunk is reduced to per-(square, slot) totals immediately and then discarded,
    so peak memory is governed by CHUNK_ROWS rather than by the size of the file.
    """
    totals = []
    reader = pd.read_csv(path, sep="	", header=None, names=COLUMNS,
                         usecols=KEEP, dtype=DTYPES, chunksize=CHUNK_ROWS)
    for chunk in reader:
        totals.append(chunk.groupby(["square_id", "time"], sort=False)["internet"].sum())
    # A (square, slot) pair can be split across two chunks, so sum the partial totals again.
    return (pd.concat(totals).groupby(level=["square_id", "time"], sort=False)
              .sum()
              .reset_index())


LOADERS = {"naive": load_naive, "optimised": load_optimised, "chunked": load_chunked}


def measure(mode: str, path: Path) -> dict:
    """Load the file one way and report how much time and memory it cost."""
    process = psutil.Process()
    # Memory already held before any data is read: the interpreter plus imported libraries.
    baseline_mb = process.memory_info().rss / 2**20
    start = time.perf_counter()
    df = LOADERS[mode](path)
    seconds = time.perf_counter() - start

    info = process.memory_info()
    peak_mb = getattr(info, "peak_wset", info.rss) / 2**20
    return {
        "mode": mode,
        "rows": len(df),
        "columns": len(df.columns),
        "baseline_mb": baseline_mb,
        # Memory attributable to the data itself, with the interpreter's own footprint removed.
        "data_peak_mb": peak_mb - baseline_mb,
        # Size of the DataFrame itself; deep=True also counts memory held outside the table.
        "dataframe_mb": df.memory_usage(deep=True).sum() / 2**20,
        # Peak memory the whole process ever held, including pandas' parsing buffers.
        "peak_process_mb": peak_mb,
        "seconds": seconds,
    }


def run_variant(mode: str, path: Path) -> dict:
    """Run one variant in a clean subprocess and return its measurements."""
    result = subprocess.run(
        [sys.executable, __file__, "--mode", mode, "--file", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default="data/raw/sms-call-internet-mi-2013-11-01.txt")
    parser.add_argument("--mode", choices=list(LOADERS),
                        help="internal: measure a single variant and print JSON")
    parser.add_argument("--out", default="experiments/results/memory_profile.csv")
    args = parser.parse_args()
    path = Path(args.file)

    if args.mode:  # child process: measure one variant, print JSON, exit
        print(json.dumps(measure(args.mode, path)))
        return

    results = [run_variant(mode, path) for mode in LOADERS]
    table = pd.DataFrame(results)
    table["full_dataset_gb"] = table["dataframe_mb"] * N_DAYS / 1024
    table["file"] = path.name

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(f"\nOne daily file: {path.name} ({path.stat().st_size / 2**20:,.0f} MB on disk)\n")
    print(table.to_string(index=False))

    naive = results[0]
    for other in results[1:]:
        print(f"\n{other['mode']} vs naive: "
              f"DataFrame {naive['dataframe_mb'] / other['dataframe_mb']:.1f}x smaller, "
              f"peak process {naive['peak_process_mb'] / other['peak_process_mb']:.1f}x smaller, "
              f"{other['seconds'] / naive['seconds']:.2f}x the time")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
