import torch
from torch import nn
import torchvision
import torchmetrics
import polars as pl
import numpy as np

import classerr.trainable
import classerr.utils


class MnistSplit(classerr.utils.DataSplit):
    def __init__(self):
        cache_path = classerr.utils.DIR_DATA_PROCESSED / "mnist_splits.parquet"

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        params = {"root": classerr.utils.DIR_DATA_RAW, "download": True, "transform": transform}
        train_dev = torchvision.datasets.MNIST(train=True, **params)
        test_raw = torchvision.datasets.MNIST(train=False, **params)

        if cache_path.exists():
            df = pl.read_parquet(cache_path)
            train_idx = df.filter(pl.col("split") == "train")["idx"].to_numpy()
            dev_idx = df.filter(pl.col("split") == "dev")["idx"].to_numpy()
            test_idx = df.filter(pl.col("split") == "test")["idx"].to_numpy()
        else:
            rng = np.random.default_rng(seed=26)

            # shuffle and split
            train_dev_perm = rng.permutation(len(train_dev)).astype(np.int32)
            split_point = int(len(train_dev) * 0.8)
            train_idx = train_dev_perm[:split_point]
            dev_idx = train_dev_perm[split_point:]

            # shuffle test set
            test_idx = rng.permutation(len(test_raw)).astype(np.int32)

            df = pl.DataFrame({
                "idx": np.concatenate([train_idx, dev_idx, test_idx]),
                "split": (
                    ["train"] * len(train_idx)
                    + ["dev"] * len(dev_idx)
                    + ["test"] * len(test_idx)
                ),
            })

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache_path)

            print(f"dataset shuffled, indices stored in {cache_path}",
                  "(if you are trying to replicate published results,",
                  "you need to replace this file by the split published on Zenodo)")

        self.train = torch.utils.data.Subset(train_dev, train_idx)
        self.dev = torch.utils.data.Subset(train_dev, dev_idx)
        self.test = torch.utils.data.Subset(test_raw, test_idx)
        bounds = classerr.utils.Bounds([(-0.5, 3.0)] * (28*28))
        self.metadata = classerr.utils.Metadata(bounds=bounds)


class SmallDenseNet(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 10)
            )
        )
        self.configure(
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.parameters(), lr=0.001),
            metric_fn=torchmetrics.Accuracy('multiclass', num_classes=10),
        )

    def callback_stopping(self, epoch, metric_dev):
        return metric_dev > 0.9
