import torch
from torch import nn
import torch.nn.functional as F
import torch.func

import classerr.trainable
import classerr.utils
from classerr.utils import len_ds

Neurons = list[tuple[int, torch.Tensor]]


class BuildingNode:
    def __init__(self, dataset: classerr.utils.Dataset, model: classerr.trainable.TrainableModel, data_idx: torch.Tensor):
        self.dataset = dataset
        self.model = model

        self.linear_seqs: list[torch.nn.Sequential] = []
        self.output_shape: torch.Size | None = None

        self.data_idx = data_idx
        self.subset = torch.utils.data.Subset(dataset, data_idx)

        self.input_weight: list[torch.Tensor] = []
        self.input_bias: list[torch.Tensor] = []
        self.prev_mask: list[torch.Tensor] = []

        self.num_saturated: list[torch.Tensor] = []

        self.decision_weight: torch.Tensor | None = None
        self.decision_bias: torch.Tensor | None = None

        self.left: BuildingNode | None = None
        self.right: BuildingNode | None = None

    def make_child(self, data_idx: torch.Tensor) -> 'BuildingNode':
        child = BuildingNode(self.dataset, self.model, data_idx)
        child.linear_seqs = self.linear_seqs
        child.output_shape = self.output_shape
        # if we were tight on memory, we could reuse the parent's tensors in one of the children
        child.input_weight = [t.clone() for t in self.input_weight]
        child.input_bias = [t.clone() for t in self.input_bias]
        child.prev_mask = [t.clone() for t in self.prev_mask]
        return child

    @property
    def num_hidden_layers(self):
        return len(self.num_saturated)

    def num_neurons_in_layer(self, i: int):
        return self.input_bias[i].numel()

    @torch.no_grad()
    def build(self):
        # compute num_saturated
        self.compute_saturations()
        # reduce (as long as there are reducible hidden neurons in the first layer)
        neurons_first_hidden = self.reduce()  # TODO

        # if there's at least one hidden neuron, we branch:
        if len(neurons_first_hidden) > 0:
            # find the branching neuron, extract its affine projection (shortcut weights)
            self.set_branching_criterion(neurons_first_hidden)
            # create children + split data_idx and assign them to the children according to the decision criterion
            self.left, self.right = self.create_children()

    def compute_saturations(self):
        if self.num_saturated:
            raise classerr.utils.AppError("saturations have already been computed")

        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.model.batch_size)

        for inputs, _ in classerr.utils.progress(loader):
            self.forward_count(inputs)

    def forward_count(self, inputs: torch.Tensor) -> torch.Tensor:
        xs = inputs
        last_nonlinearity = -1
        i = 0
        xs = self.disable_connections(i, xs)

        for j, module in enumerate(self.model.layers):
            match module:
                case nn.ReLU():
                    self.store_linear_seq(i, last_nonlinearity, j)
                    xs = self.apply_linearity(i, xs, inputs)
                    self.increment_saturation_counters(i, xs)
                    xs = module(xs)
                    i += 1
                    xs = self.disable_connections(i, xs)
                case _:
                    xs = module(xs)

        # last layer
        xs = self.apply_linearity(i, xs, inputs)
        self.store_output_shape(xs.shape)
        return xs

    def disable_connections(self, i: int, xs: torch.Tensor) -> torch.Tensor:
        if i == len(self.prev_mask):
            self.prev_mask.append(torch.ones((1, *xs.shape[1:]), dtype=torch.bool))

        return xs * self.prev_mask[i]

    def store_linear_seq(self, i, last_nonlinearity, j):
        if len(self.linear_seqs) == i:
            self.linear_seqs.append(self.model.layers[last_nonlinearity+1:j])

    def store_output_shape(self, shape: torch.Size):
        if self.output_shape is None:
            self.output_shape = torch.Size((1, *shape[1:]))

    def apply_linearity(self, i: int, xs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if i == len(self.input_weight):
            single_input_shape = torch.Size((1, *inputs.shape[1:]))
            single_output_shape = torch.Size((1, *xs.shape[1:]))
            w_shape = (single_output_shape.numel(), single_input_shape.numel())
            b_shape = (single_output_shape.numel(),)
            self.input_weight.append(torch.zeros(w_shape))
            self.input_bias.append(torch.zeros(b_shape))

            if i == 0:
                bias = self.linear_seqs[0](torch.zeros(single_input_shape))
                eye = torch.eye(single_input_shape.numel()).reshape((-1, *single_input_shape[1:]))
                weight = self.linear_seqs[0](eye) - bias
                # reshape and add
                self.input_weight[0] += weight.movedim(0, -1).reshape(w_shape)
                self.input_bias[0] += bias.reshape(b_shape)
                # disable original connections to the first layer
                self.prev_mask[0].zero_()

        return xs + F.linear(inputs.flatten(1), self.input_weight[i], self.input_bias[i]).reshape(xs.shape)

    def increment_saturation_counters(self, i: int, xs: torch.Tensor):
        if i == len(self.num_saturated):
            self.num_saturated.append(torch.zeros(xs.shape[1:], dtype=torch.int))

        self.num_saturated[i] += (xs < 0).sum(dim=0)

    def reduce(self) -> Neurons:
        """generates and modifies (not necessarily reduces) a list of hidden neurons"""
        # TODO
        breakpoint()
        return self.find_neurons_first_hidden()

    def find_neurons_first_hidden(self) -> Neurons:
        neurons: Neurons = []

        for i in range(self.num_hidden_layers):
            if self.input_weight[i].any() or self.input_bias[i].any():
                # some neurons in this layer are connected directly to the input layer (so they may be in the first hidden layer)
                # note that this implies there are some False values in self.prev_mask[i]

                if not self.prev_mask[i].any():
                    # if the self.prev_mask[i] is entirely False, every neuron is either disabled or in the first hidden layer
                    neurons_mask = (self.input_weight[i].any(dim=1) | (self.input_bias[i] != 0))  # False means disabled
                    neurons_indices = neurons_mask.nonzero(as_tuple=True)[0]
                    neurons.append((i, neurons_indices))
                else:
                    # otherwise, we need to check which neurons are not connected to the previous layers anymore
                    raise NotImplementedError

        return neurons

    def set_branching_criterion(self, neurons_first_hidden: Neurons):
        data_len = torch.tensor(len(self.data_idx))
        candidates = []

        for i, neurons_indices in neurons_first_hidden:
            saturated = self.num_saturated[i].flatten()[neurons_indices]
            minimum = torch.maximum(saturated, data_len-saturated).min(0)
            candidates.append((minimum.values, i, minimum.indices))

        _, layer, neuron = min(candidates)
        self.decision_weight = self.input_weight[layer][neuron]
        self.decision_bias = self.input_bias[layer][neuron]

    def create_children(self) -> tuple['BuildingNode', 'BuildingNode']:
        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.model.batch_size)
        mask_list = []

        for inputs, _ in classerr.utils.progress(loader):
            sat_cond = inputs.flatten(1) @ self.decision_weight + self.decision_bias < 0
            mask_list.append(sat_cond)

        mask = torch.cat(mask_list)
        left = self.make_child(self.data_idx[mask])
        right = self.make_child(self.data_idx[~mask])
        return left, right


def build_tree(model: classerr.trainable.TrainableModel, dataset: classerr.utils.Dataset):
    root = BuildingNode(dataset, model, torch.arange(len_ds(dataset)))
    stack = [root]

    while stack:
        node = stack.pop()
        node.build()

        if node.left is not None:
            stack.append(node.left)
            stack.append(node.right)
