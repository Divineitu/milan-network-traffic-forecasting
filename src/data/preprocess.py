"""Turn the 62 raw daily files (19.4 GB of text) into one compact traffic matrix.

Strategy (justified by the measurements in src/data/memory_profile.py):
  * read one daily file at a time, so RAM never scales with the full dataset;
  * read it in chunks, so peak memory depends on the chunk size, not the file size;
  * keep only square_id / time / internet, with int16 and float32 dtypes;
  * sum the per-country rows into one total per (square, 10-minute slot);
  * write the result straight into a preallocated float32 matrix.

Outputs (data/processed/):
  traffic_matrix.npy    float32 [n_slots x 10000], NaN where no data was recorded.
                        Memory-mappable, so later steps can read one area without
                        loading all 357 MB.
  time_index.parquet    the timestamp of every row, in Milan local time.
  square_totals.parquet per-square total/mean traffic and missing-slot count,
                        which is what the exploratory analysis needs.

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --limit 3     # quick trial run on the first 3 days
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

COLUMNS = ["square_id", "time", "country_code",
           "sms_in", "sms_out", "call_in", "call_out", "internet"]
KEEP = ["square_id", "time", "internet"]
DTYPES = {"square_id": "int16", "time": "int64", "internet": "float32"}

CHUNK_ROWS = 500_000
SUMMARY_BLOCK = 500          # squares summarised at a time, to avoid whole-matrix temporaries
N_SQUARES = 10_000
SLOT = "10min"
TIMEZONE = "Europe/Rome"   # Milan local time; CET (UTC+1) throughout Nov-Dec 2013


def daily_totals(path: Path) -> pd.DataFrame:
    """Read one daily file in chunks and return total internet traffic per (square, slot)."""
    partials = []
    reader = pd.read_csv(path, sep="\t", header=None, names=COLUMNS,
                         usecols=KEEP, dtype=DTYPES, chunksize=CHUNK_ROWS)
    for chunk in reader:
        # Rows differ only by country code here, so summing gives total traffic per slot.
        # NaN means "no activity recorded" and sum() treats it as 0, which is what we want.
        partials.append(chunk.groupby(["square_id", "time"], sort=False)["internet"].sum())
    # A (square, slot) pair can straddle two chunks, so the partial sums are combined again.
    return pd.concat(partials).groupby(level=["square_id", "time"], sort=False).sum().reset_index()


def build_time_index(files: list[Path]) -> pd.DatetimeIndex:
    """The complete 10-minute grid, in Milan local time, covering every day present."""
    days = pd.to_datetime([f.stem.replace("sms-call-internet-mi-", "") for f in files])
    start = days.min()
    end = days.max() + pd.Timedelta(days=1)
    return pd.date_range(start, end, freq=SLOT, inclusive="left", tz=TIMEZONE)


def process(raw_dir: Path, out_dir: Path, limit: int | None = None) -> None:
    files = sorted(raw_dir.glob("sms-call-internet-mi-*.txt"))[:limit]
    if not files:
        raise FileNotFoundError(f"No raw daily files found in {raw_dir}")

    time_index = build_time_index(files)
    # NaN (not 0) marks slots with no recorded data, so the analysis can tell them apart.
    matrix = np.full((len(time_index), N_SQUARES), np.nan, dtype=np.float32)
    slot_position = pd.Series(np.arange(len(time_index)), index=time_index)

    process_handle = psutil.Process()
    started = time.perf_counter()
    for n, path in enumerate(files, 1):
        file_started = time.perf_counter()
        totals = daily_totals(path)

        # Epoch milliseconds are UTC; convert once here so everything downstream is local time.
        stamps = pd.to_datetime(totals["time"], unit="ms", utc=True).dt.tz_convert(TIMEZONE)
        rows = slot_position.reindex(stamps).to_numpy()
        cols = totals["square_id"].to_numpy() - 1          # square IDs are 1-based
        matrix[rows, cols] = totals["internet"].to_numpy()

        # Release each day's intermediates explicitly. Without this, peak memory grew by
        # ~11 MB per file (1.27 GB over 62 days) because the collector was not reclaiming
        # the large temporaries between iterations; with it, memory stays flat.
        del totals, stamps, rows, cols
        gc.collect()

        rss_mb = process_handle.memory_info().rss / 2**20
        print(f"[{n:2d}/{len(files)}] {path.name}  "
              f"{time.perf_counter() - file_started:5.1f}s  RSS {rss_mb:6.0f} MB", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "traffic_matrix.npy", matrix)
    pd.DataFrame({"timestamp": time_index}).to_parquet(out_dir / "time_index.parquet", index=False)

    # Per-square summaries: what the exploratory analysis needs, in a file of a few hundred kB.
    # Computed one column-block at a time: np.nansum(matrix) over the whole array would
    # allocate a temporary copy of all 341 MB, which measurably doubled peak memory.
    totals = np.empty(N_SQUARES, dtype=np.float64)
    missing = np.empty(N_SQUARES, dtype=np.int32)
    for start in range(0, N_SQUARES, SUMMARY_BLOCK):
        block = matrix[:, start:start + SUMMARY_BLOCK]
        totals[start:start + SUMMARY_BLOCK] = np.nansum(block, axis=0)
        missing[start:start + SUMMARY_BLOCK] = np.isnan(block).sum(axis=0)

    pd.DataFrame({
        "square_id": np.arange(1, N_SQUARES + 1, dtype=np.int16),
        "total_traffic": totals,
        "mean_traffic": totals / (len(time_index) - missing),
        "missing_slots": missing,
    }).to_parquet(out_dir / "square_totals.parquet", index=False)

    peak_mb = getattr(process_handle.memory_info(), "peak_wset", 0) / 2**20
    print(f"\nProcessed {len(files)} files in {time.perf_counter() - started:.0f}s")
    print(f"Matrix {matrix.shape} = {matrix.nbytes / 2**20:.0f} MB, "
          f"{missing.sum() / matrix.size * 100:.2f}% missing")
    print(f"Peak process memory: {peak_mb:.0f} MB")
    print(f"Saved -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--limit", type=int, help="process only the first N days (for testing)")
    args = parser.parse_args()
    process(Path(args.raw), Path(args.out), args.limit)


if __name__ == "__main__":
    main()
