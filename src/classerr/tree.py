import torch
from torch import nn
import torch.nn.functional as F

import classerr.utils
from classerr.utils import len_ds


class BuildingNode:
    def __init__(self, dataset: classerr.utils.Dataset, model: torch.nn.Sequential, data_idx: list[int]):
        self.dataset = dataset
        self.model = model
        self.data_idx = data_idx

        self.input_weight: list[torch.Tensor] = []
        self.input_bias: list[torch.Tensor] = []
        self.prev_mask: list[torch.Tensor] = []
        self.num_saturated: list[torch.Tensor] = []

        self.decision_weight: torch.Tensor | None = None
        self.decision_bias: torch.Tensor | None = None

        self.left: BuildingNode | None = None
        self.right: BuildingNode | None = None

    def build(self):
        # 1. compute num_saturated
        # 2. reduce (as long as there are reducible hidden neurons in the first layer)
        # 3. if there's at least one hidden neuron, we branch:
        # 4. find the branching neuron, extract its affine projection (shortcut weights)
        # 5. create children + split data_idx and assign them to the children according to the decision criterion
        ...

    def compute_saturations(self):
        for s in self.num_saturated:
            s.zero_()

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.model.batch_size)

        for points, _ in loader:
            self.forward(points)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        xs = inputs
        i = 0

        for j, module in enumerate(self.model):
            match module:
                case nn.ReLU():
                    xs = self.apply_linearity(i, xs, inputs, self.model[:j] if i == 0 else None)
                    xs = self.apply_relu(i, xs)
                    i += 1
                case _:
                    xs = module(xs)

        # last layer
        xs = self.apply_linearity(i, xs, inputs)
        return xs

    def apply_linearity(
        self, i: int, xs: torch.Tensor, inputs: torch.Tensor, modules: nn.Sequential | None = None
    ) -> torch.Tensor:
        if i == len(self.prev_mask):
            single_input_shape = torch.Size(1, *inputs.shape[1:])
            single_output_shape = torch.Size(1, *xs.shape[1:])
            self.prev_mask.append(torch.ones(single_output_shape, dtype=bool))
            self.input_weight.append(torch.zeros((single_output_shape.numel(), single_input_shape.numel())))
            self.input_bias.append(torch.zeros(single_output_shape.numel()))

            if i == 0:
                assert modules is not None
                self.prev_mask[0].zero_()
                self.input_bias[0] = modules(torch.zeros(single_input_shape))
                eye = torch.eye(single_input_shape.numel()).reshape((-1, *xs.shape[1:]))
                self.input_weight[0] = modules(eye) - self.input_bias[0]

        xs *= self.prev_mask[i]
        xs += F.linear(inputs.flatten(1), self.input_weight[i], self.input_bias[i]).reshape(xs.shape)

    def apply_relu(self, i: int, xs: torch.Tensor) -> torch.Tensor:
        ...


def build_tree(model: torch.nn.Sequential, dataset: classerr.utils.Dataset):
    assert model is torch.nn.Sequential
    root = BuildingNode(model, list(range(len_ds(dataset))))

    root.build()
