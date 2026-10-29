import torch
from torch import nn
import torch.nn.functional as F
import torch.func

import classerr.trainable
import classerr.utils
from classerr.utils import len_ds

Neurons = list[tuple[int, torch.Tensor]]


class BuildingNode:
    def __init__(
        self,
        dataset: classerr.utils.Dataset,
        model: torch.nn.Sequential,
        batch_size: int,
        data_idx: torch.Tensor
    ):
        self.name = ""
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size

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
        child = BuildingNode(self.dataset, self.model, self.batch_size, data_idx)
        child.linear_seqs = self.linear_seqs
        child.output_shape = self.output_shape
        # if we were tight on memory, we could reuse the parent's tensors in one of the children
        child.input_weight = [t.clone() for t in self.input_weight]
        child.input_bias = [t.clone() for t in self.input_bias]
        child.prev_mask = [t.clone() for t in self.prev_mask]
        return child

    @property
    def num_hidden_layers(self):
        """number of hidden layers before reduction"""
        return len(self.num_saturated)

    def num_neurons_in_layer(self, i: int):
        return self.input_bias[i].numel()

    @torch.no_grad()
    def build(self):
        print("building", self.name)
        # compute num_saturated
        self.compute_saturations()
        # reduce (as long as there are reducible neurons in the first hidden layer)
        neurons_first_hidden = self.reduce()

        # if there's at least one hidden neuron, we branch:
        if len(neurons_first_hidden) > 0:
            # find the branching neuron, extract its affine projection (shortcut weights)
            self.set_branching_criterion(neurons_first_hidden)
            # create children + split data_idx and assign them to the children according to the decision criterion
            self.left, self.right = self.create_children()

    def compute_saturations(self):
        if self.num_saturated:
            raise classerr.utils.AppError("saturations have already been computed")

        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.batch_size)

        for inputs, _ in classerr.utils.progress(loader):
            self.forward_count(inputs)

    def forward_count(self, inputs: torch.Tensor) -> torch.Tensor:
        xs = inputs
        last_nonlinearity = -1
        i = 0
        xs = self.disable_connections(i, xs)

        for j, module in enumerate(self.model):
            match module:
                case nn.ReLU():
                    self.store_linear_seq(i, last_nonlinearity, j)
                    xs = self.apply_linearity(i, xs, inputs)
                    self.increment_saturation_counters(i, xs)
                    xs = module(xs)
                    i += 1
                    last_nonlinearity = j
                    xs = self.disable_connections(i, xs)
                case _:
                    xs = module(xs)

        if last_nonlinearity == j:
            raise classerr.utils.AppError("model ends with a nonlinear layer")

        # last layer
        self.store_linear_seq(i, last_nonlinearity, j+1)
        xs = self.apply_linearity(i, xs, inputs)
        self.store_output_shape(xs.shape)

        return xs

    def disable_connections(self, i: int, xs: torch.Tensor) -> torch.Tensor:
        if i == len(self.prev_mask):
            self.prev_mask.append(torch.ones((1, *xs.shape[1:]), dtype=torch.bool))

        return xs * self.prev_mask[i]

    def store_linear_seq(self, i, last_nonlinearity, j):
        if len(self.linear_seqs) == i:
            self.linear_seqs.append(self.model[last_nonlinearity+1:j])

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
                # FIXME: the line below causes the bias to be added twice (we cannot disable bias for an active neuron)
                # FIXME: branching_criterion & reduction
                self.input_bias[0] += bias.reshape(b_shape)
                # disable original connections to the first layer
                self.prev_mask[0].zero_()

            return xs
        else:
            return xs + F.linear(inputs.flatten(1), self.input_weight[i], self.input_bias[i]).reshape(xs.shape)

    def increment_saturation_counters(self, i: int, xs: torch.Tensor):
        if i == len(self.num_saturated):
            self.num_saturated.append(torch.zeros(xs.shape[1:], dtype=torch.int))

        self.num_saturated[i] += (xs < 0).sum(dim=0)

    def reduce(self) -> Neurons:
        """returns a list of neurons in the first hidden layer (after reduction)"""
        neurons_first_hidden = self.find_neurons_first_hidden(range(self.num_hidden_layers))
        data_len = torch.tensor(len(self.data_idx))
        any_reducible = False

        for i, neurons_indices in neurons_first_hidden:
            num_saturated = self.num_saturated[i].flatten()[neurons_indices]
            linear = (num_saturated == 0)
            dead = (num_saturated == data_len)
            reducible = linear | dead
            layer_reducible = reducible.any()
            any_reducible |= layer_reducible

            if layer_reducible:
                self.reduce_layer(i, neurons_indices[linear], neurons_indices[dead])

        if any_reducible:
            # we reduced some neurons
            # we need to check again (for simplicity, we start all over again)
            return self.reduce()
        else:
            # no reduction was performed in this run, this is the final list
            return neurons_first_hidden

    def reduce_layer(self, i: int, idx_linear: torch.Tensor, idx_dead: torch.Tensor):
        if idx_linear.numel() > 0:
            coords_linear = torch.unravel_index(idx_linear, self.prev_mask[i+1].shape)
            zero_vectors = torch.zeros(len(idx_linear), *self.prev_mask[i+1].shape[1:])
            unit_vectors = zero_vectors.clone()
            unit_vectors[coords_linear] = 1
            coeffs = self.linear_seqs[i+1](unit_vectors) - self.linear_seqs[i+1](zero_vectors)  # w[j,k]
            weight = self.input_weight[i][idx_linear]  # w[j,i]
            zeros_current = torch.zeros(self.prev_mask[i].shape)
            breakpoint()
            bias = self.input_bias[i][idx_linear] + self.linear_seqs[i](zeros_current).flatten()[idx_linear]  # b[j]
            # for every disabled neuron j, for every input neuron i, for every successor k: w_ki += w_kj * w_ji
            self.input_weight[i+1] += torch.einsum('jk,ji->ki', coeffs, weight)
            # for every disabled neuron j, for every successory k: b_k += w_kj * b_j
            self.input_bias[i+1] += torch.einsum('jk,j->k', coeffs, bias)

        for idx in [idx_linear, idx_dead]:
            self.input_weight[i][idx] = 0
            self.input_bias[i][idx] = 0
            coords = torch.unravel_index(idx, self.prev_mask[i+1].shape)
            self.prev_mask[i+1][coords] = 0

        print(f"from {i} removed", idx_linear, "<- linear | dead ->", idx_dead)

    def find_neurons_first_hidden(self, layers: list[int]) -> Neurons:
        neurons: Neurons = []

        for i in layers:
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

                    def transform_fn(x):
                        return self.linear_seqs[i](x * self.prev_mask[i])

                    zeros = torch.zeros(self.prev_mask[i].shape)
                    bias, vjp_fn = torch.func.vjp(transform_fn, zeros)
                    diff_total = torch.zeros_like(bias)

                    for _ in range(3):
                        rand_input = torch.randn(self.prev_mask[i].shape)
                        diff_total += torch.abs(transform_fn(rand_input) - bias)

                    EPS1 = 1e-5
                    likely_disconnected = diff_total < EPS1

                    if likely_disconnected.any():
                        sus_indices_flat = likely_disconnected.flatten().nonzero(as_tuple=True)[0]
                        N = len(sus_indices_flat)
                        unit_vectors = torch.zeros(N, bias.numel())
                        unit_vectors[torch.arange(N), sus_indices_flat] = 1
                        unit_vectors = unit_vectors.reshape(N, *bias.shape)
                        grads = torch.func.vmap(vjp_fn)(unit_vectors)[0]
                        EPS2 = 1e-8
                        disconnected_mask = (torch.abs(grads) < EPS2).flatten(1).any(dim=1)
                        disconnected_mask2 = (grads == 0).flatten(1).any(dim=1)
                        print("masks", disconnected_mask.sum(), disconnected_mask2.sum())
                        disconnected_indices = disconnected_mask.nonzero(as_tuple=True)[0]

                        if disconnected_indices.numel() > 0:
                            neurons.append((i, disconnected_indices))
            else:
                # input weights and biases are blank
                # which means that all neurons in this layer have been reduced (and disabled)
                # or no neurons in the previous layer have been reduced
                # => there are definitely no "first hidden" neurons in this layer
                pass

        return neurons

    def set_branching_criterion(self, neurons_first_hidden: Neurons):
        data_len = torch.tensor(len(self.data_idx))
        candidates = []

        for i, neurons_indices in neurons_first_hidden:
            num_saturated = self.num_saturated[i].flatten()[neurons_indices]
            minimum = torch.maximum(num_saturated, data_len-num_saturated).min(0)
            # print("min", minimum.values)
            global_index = neurons_indices[minimum.indices]  # we used neurons_indices to select a subset of neurons
            candidates.append((minimum.values, i, global_index))

        _, layer, neuron = min(candidates)
        self.decision_weight = self.input_weight[layer][neuron]
        self.decision_bias = self.input_bias[layer][neuron]
        print(min(candidates))

    def create_children(self) -> tuple['BuildingNode', 'BuildingNode']:
        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.batch_size)
        mask_list = []

        for inputs, _ in classerr.utils.progress(loader):
            sat_cond = inputs.flatten(1) @ self.decision_weight + self.decision_bias < 0
            mask_list.append(sat_cond)

        mask = torch.cat(mask_list)

        if mask.all() or not mask.any():
            breakpoint()

        left = self.make_child(self.data_idx[mask])
        left.name = self.name + "l"
        right = self.make_child(self.data_idx[~mask])
        right.name = self.name + "r"
        return left, right


def build_tree(model: classerr.trainable.TrainableModel, dataset: classerr.utils.Dataset):
    root = BuildingNode(dataset, model.layers, model.batch_size, torch.arange(len_ds(dataset)))
    stack = [root]

    while stack:
        # print(stack)
        node = stack.pop()
        node.build()

        if node.left is not None:
            stack.append(node.left)
            stack.append(node.right)
