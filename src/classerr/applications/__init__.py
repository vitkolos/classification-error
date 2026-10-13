import torch

from classerr.applications import mnist
import classerr.trainable
import classerr.utils


class DataBundle:
    def __init__(self, dataset: str):
        self.model_class: type[classerr.trainable.BaseModel]
        self.data_split: classerr.utils.DataSplit

        match dataset.lower():
            case 'mnist':
                self.model_file = classerr.utils.DIR_MODELS / "mnist_dense_net.pt"
                self.model_class = mnist.SmallDenseNet
                self.data_split = mnist.MnistSplit()
            case _:
                raise NotImplementedError(f"'{dataset}' dataset is not available")

    def load_model(self) -> torch.nn.Module:
        model = self.model_class()
        model.load(self.model_file).eval()
        return model
