import sys

import click
import torch
from torch import nn
import torchmetrics

import classerr.applications
import classerr.quantization
import classerr.utils


@click.command()
@click.argument('dataset')
def main(dataset):
    bundle = classerr.applications.DataBundle(dataset)
    data_split = bundle.data_split
    model = bundle.model_class()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    print('accelerator', device)
    model.to(device)

    model.fit(data_split.train, data_split.dev)
    model.cpu()
    model.save(bundle.model_file)


class BaseModel(nn.Module):
    """base for a single-stream model"""

    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def callback_stopping(self, epoch: int, metric_dev: float):
        return False

    def save(self, path):
        data = self.state_dict()
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path, weights_only=True, map_location=self.device)
        self.load_state_dict(data)
        return self

    def round(self, **kwargs):
        """modifies the network in-place (converts the network to its approximation)"""
        return classerr.quantization.lower_precision(self, **kwargs)


class TrainableModel(BaseModel):
    def __init__(self, layers: nn.Sequential):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        return self.layers(x)

    def configure(
        self,
        loss_fn,
        optimizer,
        metric_fn: torchmetrics.Metric,
        scheduler=None,
        epochs=20,
        batch_size=64,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(
        self,
        data_train: classerr.utils.Dataset,
        data_dev: classerr.utils.Dataset,
    ):
        loader_train = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size)
        loader_dev = torch.utils.data.DataLoader(data_dev, batch_size=self.batch_size)

        for epoch in range(1, self.epochs+1):
            loss_train, metric_train = self.train_epoch(loader_train)
            loss_dev, metric_dev = self.evaluate(loader_dev)
            print(f"epoch {epoch} done | train: loss {loss_train:.2f}, metric {metric_train:.2f}",
                  f"| dev: loss {loss_dev:.2f}, metric {metric_dev:.2f}")
            is_stopping = self.callback_stopping(epoch, metric_dev)

            if is_stopping:
                print("stopping")
                return

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        self.train()
        return self._execute_epoch(loader, train=True)

    def evaluate(self, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        with torch.no_grad():
            self.eval()
            return self._execute_epoch(loader, train=False)

    def quality(self, metric_name: str, data_test: classerr.utils.Dataset, error_scaling: float) -> float:
        with torch.no_grad():
            self.eval()
            loader = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size)
            metric_fn: torchmetrics.Metric | None = None

            match metric_name.lower().split('/'):
                case ['rmse']:
                    metric_fn = torchmetrics.MeanSquaredError(False)
                case ['mae']:
                    metric_fn = torchmetrics.MeanAbsoluteError()
                case ['accuracy', classes]:
                    metric_fn = torchmetrics.Accuracy(task='multiclass', num_classes=int(classes))
                case _:
                    raise NotImplementedError(f"'{metric_name}' metric is not available")

            loss, metric = self._execute_epoch(loader, train=False, metric_fn=metric_fn)
            return metric * error_scaling

    def _execute_epoch(self, loader: torch.utils.data.DataLoader, train: bool, metric_fn: torchmetrics.Metric | None = None) -> tuple[float, float]:
        device = self.device
        loss_sum = 0.0
        total_samples = 0

        if metric_fn is None:
            metric_fn = self.metric_fn

        metric_fn = metric_fn.to(device)
        metric_fn.reset()

        for X, y in classerr.utils.progress(loader):
            X, y = X.to(device), y.to(device)
            N = X.shape[0]
            pred = self(X)
            loss = self.loss_fn(pred, y)
            loss_sum += loss.item()*N
            total_samples += N
            metric_fn.update(pred, y)

            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler is not None and self.scheduler.step()

        loss = loss_sum / total_samples
        metric = metric_fn.compute().item()
        return loss, metric

    def subset(self, dataset: classerr.utils.Dataset, classification: bool = False) -> classerr.utils.Dataset:
        """creates a subset of the dataset containing only items where the model is accurate enough"""
        predictions_list = []
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        with torch.no_grad():
            self.eval()
            device = self.device

            for X, y in classerr.utils.progress(loader):
                X, y = X.to(device), y.to(device)
                predictions_list.append(self(X))

        predictions = torch.cat(predictions_list)

        if classification:
            targets = torch.tensor([target for point, target in dataset])
            accurate_enough = targets == predictions.argmax(dim=1)
        else:
            targets = dataset[:][1]
            accurate_enough = (targets - predictions).abs() < 1.0

        accurate_count = accurate_enough.sum()
        indices = torch.nonzero(accurate_enough, as_tuple=True)[0].tolist()
        removed_count = len(dataset)-accurate_count  # type: ignore[arg-type]
        print(
            f'subset contains only {accurate_count} data points ({removed_count} removed, model was too inaccurate)', file=sys.stderr)
        return torch.utils.data.Subset(dataset, indices)


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


if __name__ == '__main__':
    main()
