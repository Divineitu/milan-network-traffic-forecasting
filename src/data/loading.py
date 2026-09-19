"""Shared access to the processed dataset.

The traffic matrix is opened with mmap_mode="r", so NumPy reads only the parts of the
341 MB file that are actually touched. Extracting a few areas therefore costs a few MB,
not 341 MB, which is what makes the later analysis and model training comfortable on a
16 GB laptop.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")


def load_matrix(processed: Path = PROCESSED) -> np.memmap:
    """Traffic matrix [n_slots x 10000], float32, NaN where nothing was recorded."""
    return np.load(processed / "traffic_matrix.npy", mmap_mode="r")


def load_time_index(processed: Path = PROCESSED) -> pd.DatetimeIndex:
    """Timestamp of every row of the matrix, in Milan local time.

    The 10-minute frequency is restored explicitly: Parquet does not store it, and
    statsmodels refuses to extend a model over an index with no declared frequency.
    """
    stamps = pd.read_parquet(processed / "time_index.parquet")["timestamp"]
    return pd.DatetimeIndex(stamps, freq="10min")


def load_square_totals(processed: Path = PROCESSED) -> pd.DataFrame:
    """Per-square total/mean traffic and missing-slot count (a few hundred kB)."""
    return pd.read_parquet(processed / "square_totals.parquet")


def get_series(square_ids: int | list[int], processed: Path = PROCESSED) -> pd.DataFrame:
    """Time series for one or more squares as a DataFrame indexed by local time."""
    ids = [square_ids] if isinstance(square_ids, int) else list(square_ids)
    matrix = load_matrix(processed)
    columns = np.asarray(ids) - 1                      # square IDs are 1-based
    data = np.asarray(matrix[:, columns], dtype=np.float32)   # copies only these columns
    return pd.DataFrame(data, index=load_time_index(processed), columns=ids)


def top_squares(n: int = 3, processed: Path = PROCESSED) -> list[int]:
    """The n squares with the highest total internet traffic over the whole period."""
    totals = load_square_totals(processed)
    return totals.nlargest(n, "total_traffic")["square_id"].astype(int).tolist()
