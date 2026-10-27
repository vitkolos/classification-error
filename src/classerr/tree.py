import torch
from torch import nn
import torch.nn.functional as F

import classerr.trainable
import classerr.utils
from classerr.utils import len_ds


class BuildingNode:
    def __init__(self, dataset: classerr.utils.Dataset, model: classerr.trainable.TrainableModel, data_idx: list[int]):
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

    @property
    def num_hidden_layers(self):
        return len(self.num_saturated)

    @torch.no_grad()
    def build(self):
        # 1. compute num_saturated
        self.compute_saturations()
        # 2. reduce (as long as there are reducible hidden neurons in the first layer)
        # TODO: self.reduce()
        # 3. if there's at least one hidden neuron, we branch:
        # 4. find the branching neuron, extract its affine projection (shortcut weights)
        # 5. create children + split data_idx and assign them to the children according to the decision criterion
        breakpoint()

    def compute_saturations(self):
        for s in self.num_saturated:
            s.zero_()

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.model.batch_size)

        for points, _ in loader:
            self.forward_count(points)

    def forward_count(self, inputs: torch.Tensor) -> torch.Tensor:
        xs = inputs
        i = 0

        for j, module in enumerate(self.model.layers):
            match module:
                case nn.ReLU():
                    xs = self.apply_linearity(i, xs, inputs, modules=(self.model.layers[:j] if i == 0 else None))
                    self.count_saturations(i, xs)
                    i += 1

            xs = module(xs)

        # last layer
        xs = self.apply_linearity(i, xs, inputs)
        return xs

    def apply_linearity(
        self, i: int, xs: torch.Tensor, inputs: torch.Tensor, modules: nn.Sequential | None = None
    ) -> torch.Tensor:
        if i == len(self.prev_mask):
            single_input_shape = torch.Size((1, *inputs.shape[1:]))
            single_output_shape = torch.Size((1, *xs.shape[1:]))
            w_shape = (single_output_shape.numel(), single_input_shape.numel())
            b_shape = (single_output_shape.numel(),)
            self.prev_mask.append(torch.ones(single_output_shape, dtype=torch.bool))
            self.input_weight.append(torch.zeros(w_shape))
            self.input_bias.append(torch.zeros(b_shape))

            if i == 0:
                assert modules is not None
                self.prev_mask[0].zero_()
                bias = modules(torch.zeros(single_input_shape))
                eye = torch.eye(single_input_shape.numel()).reshape((-1, *single_input_shape[1:]))
                weight = modules(eye) - bias
                # reshape and add
                self.input_weight[0] += weight.movedim(0, -1).reshape(w_shape)
                self.input_bias[0] += bias.reshape(b_shape)

        xs *= self.prev_mask[i]
        xs += F.linear(inputs.flatten(1), self.input_weight[i], self.input_bias[i]).reshape(xs.shape)
        return xs

    def count_saturations(self, i: int, xs: torch.Tensor) -> torch.Tensor:
        if i == len(self.num_saturated):
            self.num_saturated.append(torch.zeros(xs.shape[1:], dtype=torch.int))

        self.num_saturated[i] += (xs < 0).sum(dim=0)

    def neurons_first_hidden(self) -> list:
        neurons = []

        for i in range(self.num_hidden_layers):
            if (self.input_weight[i].any() or self.input_bias[i].any()) and not self.prev_mask[i].all():
                for j in range(self.input_bias[i].numel()):
                    ...

        return neurons


def build_tree(model: classerr.trainable.TrainableModel, dataset: classerr.utils.Dataset):
    root = BuildingNode(dataset, model, list(range(len_ds(dataset))))
    root.build()
