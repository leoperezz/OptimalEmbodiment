import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class ArrayStats:
    """Simple numeric summary for a single NumPy array."""

    shape: tuple
    dtype: str
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]


@dataclass
class NpzFileReport:
    """Summary of the contents of one .npz file."""

    path: str
    keys: List[str]
    numeric_arrays: Dict[str, ArrayStats]
    non_numeric_keys: List[str]
    load_error: Optional[str] = None


def _iter_npz_files(root_dir: str) -> Iterable[str]:
    """Yield all .npz files under a directory tree."""
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".npz"):
                yield os.path.join(dirpath, name)


def _array_stats(arr: np.ndarray) -> ArrayStats:
    """Compute basic stats for numeric arrays; skip for non-numeric."""
    if not np.issubdtype(arr.dtype, np.number):
        return ArrayStats(shape=arr.shape, dtype=str(arr.dtype), min=None, max=None, mean=None)

    # Use float64 for stable statistics; guard against empty arrays.
    if arr.size == 0:
        return ArrayStats(shape=arr.shape, dtype=str(arr.dtype), min=None, max=None, mean=None)

    arr_float = arr.astype(np.float64)
    return ArrayStats(
        shape=arr.shape,
        dtype=str(arr.dtype),
        min=float(arr_float.min()),
        max=float(arr_float.max()),
        mean=float(arr_float.mean()),
    )


def analyze_npz_file(path: str) -> NpzFileReport:
    """
    Load a single .npz file (with allow_pickle=True) and compute:
    - list of keys
    - per‑key numeric statistics (shape, dtype, min, max, mean)
    - which keys are non‑numeric or object arrays
    """
    numeric_arrays: Dict[str, ArrayStats] = {}
    non_numeric_keys: List[str] = []

    try:
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            for key in keys:
                arr = data[key]
                if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                    numeric_arrays[key] = _array_stats(arr)
                else:
                    non_numeric_keys.append(key)

        return NpzFileReport(
            path=path,
            keys=keys,
            numeric_arrays=numeric_arrays,
            non_numeric_keys=non_numeric_keys,
            load_error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return NpzFileReport(
            path=path,
            keys=[],
            numeric_arrays={},
            non_numeric_keys=[],
            load_error=str(exc),
        )


def analyze_dataset(
    root_dir: str = "data",
    max_files: Optional[int] = 50,
) -> List[NpzFileReport]:
    """
    Analyze many .npz files under a root directory.

    Parameters
    ----------
    root_dir:
        Directory under which to search for .npz files (e.g. 'data' or 'data/ACCAD').
    max_files:
        Optional cap on the number of files to inspect, to avoid
        scanning the entire AMASS dataset in one go.
    """
    reports: List[NpzFileReport] = []
    for idx, path in enumerate(_iter_npz_files(root_dir)):
        if max_files is not None and idx >= max_files:
            break
        reports.append(analyze_npz_file(path))
    return reports


def print_summary(
    reports: List[NpzFileReport],
) -> None:
    """Pretty‑print a human‑readable summary of the analysis."""
    total_files = len(reports)
    failed = [r for r in reports if r.load_error is not None]
    print(f"Analyzed {total_files} .npz files.")
    if failed:
        print(f"  - {len(failed)} files could not be loaded.")

    for report in reports:
        print("=" * 80)
        print(f"File: {report.path}")
        if report.load_error:
            print(f"  ERROR loading file: {report.load_error}")
            continue

        print(f"  Keys: {', '.join(report.keys)}")
        if report.numeric_arrays:
            print("  Numeric arrays:")
            for key, stats in report.numeric_arrays.items():
                stats_dict = asdict(stats)
                print(f"    - {key}: {stats_dict}")
        if report.non_numeric_keys:
            print(f"  Non‑numeric / object keys: {', '.join(report.non_numeric_keys)}")


if __name__ == "__main__":
    # Example usage:
    # python test/data/testing_dataset.py
    reports = analyze_dataset(root_dir="data", max_files=50)
    print_summary(reports)

