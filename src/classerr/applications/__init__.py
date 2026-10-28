import json
import torch

from classerr.applications import mnist, california_housing
import classerr.trainable
import classerr.utils


class DataBundle:
    def __init__(self, dataset: str):
        dataset = dataset.lower()
        self.model_class: type[classerr.trainable.TrainableModel]
        self.data_split: classerr.utils.DataSplit

        classerr.utils.DIR_DATA_RAW.mkdir(parents=True, exist_ok=True)
        classerr.utils.DIR_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        classerr.utils.DIR_MODELS.mkdir(parents=True, exist_ok=True)

        match dataset:
            case 'mnist':
                self.model_file = classerr.utils.DIR_MODELS / "mnist_conv_net.pt"
                self.model_class = mnist.SmallConvNet
                self.data_split = mnist.MnistSplit()
            case 'california':
                self.model_file = classerr.utils.DIR_MODELS / "california_mlp.pt"
                self.model_class = california_housing.HousingMLP
                self.data_split = california_housing.CaliforniaHousingSplit()
            case _:
                raise NotImplementedError(f"'{dataset}' dataset is not available")

        # save metadata
        with open(classerr.utils.DIR_DATA_PROCESSED / f'{dataset}_metadata.json', 'w') as file_json:
            json.dump(self.data_split.metadata.to_dict(), file_json)

    def load_model(self) -> classerr.trainable.TrainableModel:
        model = self.model_class()
        model.load(self.model_file).eval()
        return model
