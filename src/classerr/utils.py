import sys
from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import tqdm


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


def save_split_backup(dataset: str, idx_train, idx_dev, idx_test):
    with open(f'datasets/split_{dataset}.json', 'w') as f:
        json.dump({
            'train': idx_train.tolist() if hasattr(idx_train, 'tolist') else idx_train,
            'dev': idx_dev.tolist() if hasattr(idx_dev, 'tolist') else idx_dev,
            'test': idx_test.tolist() if hasattr(idx_test, 'tolist') else idx_test,
        }, f)


def load_split_backup(dataset: str):
    with open(f'datasets/split_{dataset}.json', 'r') as f:
        return json.load(f)


def progress(iterable, main=False, **kwargs):
    if not main:
        kwargs['leave'] = False
        kwargs['disable'] = None

    return tqdm.tqdm(iterable, **kwargs)
