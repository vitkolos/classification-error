import torch
from torch import nn
import torchmetrics
import sklearn.datasets
import sklearn.preprocessing
import numpy as np

import classerr.trainable
import classerr.utils


class CaliforniaHousingDataset(classerr.utils.Dataset):
    """https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset"""

    def __init__(self, data: np.ndarray, metadata: classerr.utils.Metadata):
        if metadata.scaler is None:
            metadata.scaler = sklearn.preprocessing.StandardScaler()
            metadata.scaler.fit(data)
            metadata.error_scaling = metadata.scaler.scale_[0].item()

        # standardize
        data = metadata.scaler.transform(data)

        if metadata.bounds is None:
            metadata.fit_bounds(data)
        else:
            data = metadata.remove_outliers(data)

        self.data = torch.from_numpy(data[:, metadata.sl_data]).to(dtype=torch.get_default_dtype())
        self.target = torch.from_numpy(data[:, metadata.sl_target]).to(dtype=torch.get_default_dtype())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class CaliforniaHousingSplit(classerr.utils.DataSplit):
    def __init__(self):
        data, target = sklearn.datasets.fetch_california_housing(data_home=classerr.utils.DIR_DATA_RAW, return_X_y=True)
        data = np.column_stack((target, data))

        splits = {}
        splits['train_dev'], splits['test'] = classerr.utils.idx_split(data, test_size=1/8, random_state=128)
        data_train_dev = data[splits['train_dev']]
        data_test = data[splits['test']]
        splits['train'], splits['dev'] = classerr.utils.idx_split(data_train_dev, test_size=1/7, random_state=129)
        split_parquet_path = classerr.utils.DIR_DATA_PROCESSED / "california_splits.parquet"
        classerr.utils.load_or_create_split_parquet(split_parquet_path, splits)

        self.metadata = classerr.utils.Metadata()
        train_dev = CaliforniaHousingDataset(data_train_dev, self.metadata)
        self.test = CaliforniaHousingDataset(data_test, self.metadata)
        self.train = torch.utils.data.Subset(train_dev, splits['train'])
        self.dev = torch.utils.data.Subset(train_dev, splits['dev'])


class HousingMLP(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(8, 256),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(128, 64),
                nn.ReLU(),

                nn.Linear(64, 1)
            )
        )
        self.configure(
            loss_fn=nn.HuberLoss(),
            optimizer=torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5),
            metric_fn=torchmetrics.MeanSquaredError(),
            epochs=50,
        )

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                module.bias.data.fill_(0.01)

        self.apply(init_weights)
