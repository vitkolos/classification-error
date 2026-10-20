import torch
from torch import nn
import torchvision
import torchmetrics

import classerr.trainable
import classerr.utils


class MnistSplit(classerr.utils.DataSplit):
    def __init__(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        params = {"root": classerr.utils.DIR_DATA_RAW, "download": True, "transform": transform}
        train_dev = torchvision.datasets.MNIST(train=True, **params)
        test_raw = torchvision.datasets.MNIST(train=False, **params)

        splits = {}
        splits['train'], splits['dev'] = classerr.utils.idx_split(train_dev, test_size=1/5, random_state=126)
        splits['test'] = classerr.utils.idx_shuffle(test_raw, random_state=127)
        split_parquet_path = classerr.utils.DIR_DATA_PROCESSED / "mnist_splits.parquet"
        classerr.utils.load_or_create_split_parquet(split_parquet_path, splits)

        self.train = torch.utils.data.Subset(train_dev, splits['train'])
        self.dev = torch.utils.data.Subset(train_dev, splits['dev'])
        self.test = torch.utils.data.Subset(test_raw, splits['test'])
        bounds = classerr.utils.Bounds([(-0.5, 3.0)] * (28*28))
        self.metadata = classerr.utils.Metadata(bounds=bounds, classification=True)


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
        return metric_dev > 0.98
