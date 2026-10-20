import sys
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm
import sklearn.model_selection
import sklearn.utils


DIR_DATA_RAW = Path("data/raw")
DIR_DATA_PROCESSED = Path("data/processed")
DIR_MODELS = Path("models")
DIR_OUTPUTS = Path("outputs")


class Bounds:
    def __init__(self, bounds: list[tuple[float | None, float | None]], lb: np.ndarray | None = None, ub: np.ndarray | None = None):
        self.seq = bounds

        if lb is not None and ub is not None:
            self.lb, self.ub = lb, ub
        else:
            self.lb = np.array([lb if lb is not None else float('-inf') for lb, _ in bounds])
            self.ub = np.array([ub if ub is not None else float('inf') for _, ub in bounds])

    def __len__(self):
        return len(self.seq)


@dataclass
class Metadata:
    bounds: Bounds = None  # type: ignore[assignment]
    classification: bool = False
    scaler: Any = None
    error_scaling: float = 1.0
    sl_data: slice = field(default_factory=lambda: slice(1, None))
    sl_target: slice = field(default_factory=lambda: slice(0, 1))

    def fit_bounds(self, data_full: np.ndarray, padding: float = 0.05):
        data = data_full[:, self.sl_data]
        mins, maxs = np.min(data, axis=0), np.max(data, axis=0)
        ranges = maxs - mins
        lower_bounds = mins - (ranges * padding)
        upper_bounds = maxs + (ranges * padding)
        bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
        self.bounds = Bounds(bounds, lower_bounds, upper_bounds)

    def remove_outliers(self, data_full: np.ndarray):
        if self.bounds is None:
            raise AttributeError('bounds not fitted')

        data = data_full[:, self.sl_data]
        valid_cells = (data >= self.bounds.lb) & (data <= self.bounds.ub)
        valid_rows = valid_cells.all(axis=1)
        print(f'removed {(~valid_rows).sum()} outliers', file=sys.stderr)
        return data_full[valid_rows]

    def to_dict(self) -> dict:
        return {
            'error_scaling': self.error_scaling,
            'bounds': getattr(self.bounds, 'seq', None),
            'scales': getattr(self.scaler, 'scale_', np.empty(0)).tolist(),
        }


@dataclass
class DataSplit:
    train: Dataset
    dev: Dataset
    test: Dataset
    metadata: Metadata


def get_samples(dataset: Dataset, items: str = '') -> list[tuple[int, torch.Tensor]]:
    start, stop = 0, len(dataset)  # type: ignore[arg-type]

    if ':' in items:
        a, b = items.split(':', 2)
        if a:
            start = max(start, int(a))
        if b:
            stop = min(stop, int(b))
    elif items:
        # single item
        start = int(items)
        stop = start + 1

    # clone to break connection with Storage
    return [(i, dataset[i][0].clone(), dataset[i][1]) for i in range(start, stop)]


def idx_split(dataset, **kwargs):
    return sklearn.model_selection.train_test_split(
        np.arange(len(dataset), dtype=np.int32),
        **kwargs,
    )


def idx_shuffle(dataset, **kwargs):
    return sklearn.utils.shuffle(
        np.arange(len(dataset), dtype=np.int32),
        **kwargs,
    )


def load_or_create_split_parquet(split_parquet_path, splits: dict[str, np.ndarray]):
    if split_parquet_path.exists():
        df = pl.read_parquet(split_parquet_path)

        for key in splits.keys():
            splits[key] = df.filter(pl.col("split") == key)["idx"].to_numpy()
    else:
        df = pl.concat([
            pl.DataFrame({"idx": indices, "split": name})
            for name, indices in splits.items()
        ])
        df.write_parquet(split_parquet_path)

        print(f"dataset shuffled, indices stored in {split_parquet_path}",
              "(if you are trying to replicate published results,",
              "you need to replace this file by the split published on Zenodo)")


def progress(iterable, main=False, **kwargs):
    if not main:
        kwargs['leave'] = False
        kwargs['disable'] = None

    return tqdm.tqdm(iterable, **kwargs)
