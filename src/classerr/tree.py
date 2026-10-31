import typing
import torch
from torch import nn
import torch.nn.functional as F
import torch.func
import tqdm

import classerr.trainable
import classerr.utils
from classerr.utils import len_ds

Neurons = dict[int, torch.Tensor]

LAYERS_NONLINEAR = [
    nn.ReLU,
]

LAYERS_LINEAR = [
    # actual linear projections
    nn.Linear, nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm,
    # other kinds of layers not violating linearity
    nn.Flatten, nn.Dropout,
]


class BuildingNode:
    def __init__(
        self,
        dataset: classerr.utils.Dataset,
        model: torch.nn.Sequential,
        batch_size: int,
        data_idx: torch.Tensor
    ):
        if model.training:
            raise classerr.utils.AppError("model is in the training mode; you should call model.eval()")

        self.name = ""
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size

        self.linear_seqs: list[torch.nn.Sequential] = []
        self.output_shape: torch.Size | None = None

        self.data_idx = data_idx
        self.subset = torch.utils.data.Subset(dataset, typing.cast(typing.Sequence[int], data_idx))

        self.input_weight: list[torch.Tensor] = []
        self.input_bias_init: list[torch.Tensor] = []  # initial bias (present in the original model)
        self.input_bias_addit: list[torch.Tensor] = []  # additional bias (collected from the ancestors)
        self.prev_mask: list[torch.Tensor] = []

        self.num_active: list[torch.Tensor] = []

        self.decision_weight: torch.Tensor | None = None
        self.decision_bias: torch.Tensor | None = None
        self.children: list[BuildingNode] = []

    def make_child(self, data_idx: torch.Tensor, name_suffix: str = "") -> 'BuildingNode':
        child = BuildingNode(self.dataset, self.model, self.batch_size, data_idx)
        child.name = self.name + name_suffix
        child.linear_seqs = self.linear_seqs
        child.output_shape = self.output_shape
        # if we were tight on memory, we could reuse the parent's tensors in one of the children
        child.input_weight = [t.clone() for t in self.input_weight]
        child.input_bias_init = [t.clone() for t in self.input_bias_init]
        child.input_bias_addit = [t.clone() for t in self.input_bias_addit]
        child.prev_mask = [t.clone() for t in self.prev_mask]
        return child

    @property
    def num_hidden_layers(self):
        """number of hidden layers before reduction"""
        return len(self.num_active)

    def num_neurons_in_layer(self, i: int):
        return self.input_bias_init[i].numel()

    @torch.no_grad()
    def build(self):
        # compute num_active
        self.compute_activity()
        # reduce (as long as there are reducible neurons in the first hidden layer)
        neurons_first_hidden = self.reduce()

        # if there's at least one hidden neuron, we branch:
        if len(neurons_first_hidden) > 0:
            # find the branching neuron, extract its affine projection (shortcut weights)
            self.set_branching_criterion(neurons_first_hidden)
            # create children + split data_idx and assign them to the children according to the decision criterion
            self.children = self.create_children()

    def compute_activity(self):
        if self.num_active:
            raise classerr.utils.AppError("activity has already been computed")

        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.batch_size)

        for inputs, _ in classerr.utils.progress(loader):
            self.forward_count(inputs)

    def forward_count(self, inputs: torch.Tensor) -> torch.Tensor:
        xs = inputs
        last_nonlinearity = -1
        i = 0
        xs = self.disable_connections(i, xs)

        for j, module in enumerate(self.model):
            if any(isinstance(module, t) for t in LAYERS_NONLINEAR):
                self.store_linear_seq(i, last_nonlinearity, j)
                xs = self.apply_linearity(i, xs, inputs)
                self.increment_activity_counters(i, xs, module)
                xs = module(xs)
                i += 1
                last_nonlinearity = j
                xs = self.disable_connections(i, xs)
            elif any(isinstance(module, t) for t in LAYERS_LINEAR):
                xs = module(xs)
            else:
                raise NotImplementedError(
                    f"BuildTree.forward_count is not implemented for '{type(module).__name__}' module")

        if last_nonlinearity == j:
            raise classerr.utils.AppError("model ends with a nonlinear layer")

        # last layer
        self.store_linear_seq(i, last_nonlinearity, j+1)
        xs = self.apply_linearity(i, xs, inputs)
        self.store_output_shape(xs.shape)

        torch.testing.assert_close(xs, self.model(inputs))
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
            self.input_bias_addit.append(torch.zeros(b_shape))

            bias_init = self.linear_seqs[i](torch.zeros(self.prev_mask[i].shape))
            self.input_bias_init.append(bias_init.reshape(b_shape))

            if i == 0:
                eye = torch.eye(single_input_shape.numel()).reshape((-1, *single_input_shape[1:]))
                weight = self.linear_seqs[0](eye) - bias_init
                # reshape and add
                self.input_weight[0] += weight.movedim(0, -1).reshape(w_shape)
                # disable original connections to the first layer
                self.prev_mask[0].zero_()

            return xs  # all the transformations have already been done (or there are none)
        else:
            return xs + F.linear(inputs.flatten(1), self.input_weight[i], self.input_bias_addit[i]).reshape(xs.shape)

    def increment_activity_counters(self, i: int, xs: torch.Tensor, module: torch.nn.Module):
        if i == len(self.num_active):
            self.num_active.append(torch.zeros(xs.shape[1:].numel(), dtype=torch.int))

        # MAX: we need to somehow apply the transformation that module does (but in the arg-way)
        # MAX: we are interested in the number of "activations" (argmax + bincount)
        self.num_active[i] += (xs < 0).sum(dim=0).flatten()

    def reduce(self) -> Neurons:
        """returns lists of neurons in the first hidden layer (FHL) after reduction"""
        data_len = torch.tensor(len(self.data_idx))
        to_be_rescanned = set(range(self.num_hidden_layers))
        neurons_first_hidden: Neurons = {}

        while to_be_rescanned:
            self.find_neurons_first_hidden(to_be_rescanned, neurons_first_hidden)
            to_be_rescanned.clear()

            for i, neurons_indices in neurons_first_hidden.items():
                num_active = self.num_active[i][neurons_indices]
                # MAX: reducible neurons have all zeros except for one
                linear = (num_active == 0)
                dead = (num_active == data_len)
                reducible = linear | dead

                if reducible.any():
                    # MAX: for each FHL neuron, we should be able to get all the w&b in a reasonable shape
                    reducible_idx = neurons_indices[reducible]
                    linear_idx = neurons_indices[linear]
                    linear_weight = self.input_weight[i][linear_idx]
                    linear_bias = self.input_bias_init[i][linear_idx] + self.input_bias_addit[i][linear_idx]
                    self.reduce_layer(i, reducible_idx, linear_idx, linear_weight, linear_bias)
                    to_be_rescanned.add(i)  # remove the reduced neurons from the list

                    if i+1 < self.num_hidden_layers:  # check if there's a next layer
                        # now, the neurons in the next layer may belong in the FHL
                        to_be_rescanned.add(i+1)

        return neurons_first_hidden

    def reduce_layer(
        self,
        i: int,
        reducible_idx: torch.Tensor,
        linear_idx: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor
    ):
        if linear_idx.numel() > 0:
            for idx, w, b in zip(
                linear_idx.split(self.batch_size),
                linear_weight.split(self.batch_size),
                linear_bias.split(self.batch_size)
            ):
                self.reduce_linear_neurons(i, w, b, idx)

        for idx in reducible_idx:
            coords = torch.unravel_index(idx, self.prev_mask[i+1].shape)
            self.prev_mask[i+1][coords] = 0

    def reduce_linear_neurons(self, i: int, in_weight: torch.Tensor, in_bias: torch.Tensor, out_idx: torch.Tensor):
        neurons_arranged_shape = self.prev_mask[i+1].shape[1:]
        coords_linear = torch.unravel_index(out_idx, neurons_arranged_shape)
        unit_vectors = torch.zeros(len(out_idx), *neurons_arranged_shape)
        zero_vector = unit_vectors[0:1].clone()
        unit_vectors[(torch.arange(len(out_idx)), *coords_linear)] = 1
        assert unit_vectors.flatten(1).sum(dim=1).eq(1).all()

        coeffs = self.linear_seqs[i+1](unit_vectors) - self.linear_seqs[i+1](zero_vector)  # w[j,k]
        coeffs = coeffs.flatten(1)
        # for every disabled neuron j, for every input neuron i, for every successor k: w_ki += w_kj * w_ji
        self.input_weight[i+1] += torch.einsum('jk,ji->ki', coeffs, in_weight)
        # for every disabled neuron j, for every successory k: b_k += w_kj * b_j
        self.input_bias_addit[i+1] += torch.einsum('jk,j->k', coeffs, in_bias)

    def find_neurons_first_hidden(self, layers: typing.Iterable[int], neurons: Neurons):
        """
        A neuron is in the first hidden layer iff none of its ancestors is a hidden neuron.
        In our model, it suffices to verify that the neuron:
        1) is not disabled,
        2) is in the first layer or has no connections with the previous layer.
        """
        for i in layers:
            found = self.find_neurons_first_hidden_in_layer(i)
            neurons[i] = found

            if found is None:
                del neurons[i]

        return neurons

    def find_neurons_first_hidden_in_layer(self, i: int) -> torch.Tensor | None:
        if not self.prev_mask[i].any():
            # the self.prev_mask[i] is entirely False => there are no active connections to the previous layer
            # note that this also works for the neurons in the first layer (the original connections were disabled)
            neurons_mask = self.prev_mask[i+1].flatten()  # False means disabled

            if neurons_mask.any():
                neurons_indices = neurons_mask.nonzero(as_tuple=True)[0]
                # all non-disabled neurons definitely are in the first hidden layer
                return neurons_indices
        else:
            # otherwise, we need to check which neurons are not connected to the previous layers anymore
            def transform_fn(x):
                return self.linear_seqs[i](x * self.prev_mask[i])

            zeros = torch.zeros(self.prev_mask[i].shape)
            bias_init, vjp_fn = torch.func.vjp(transform_fn, zeros)  # type: ignore[misc]
            diff_total = torch.zeros_like(bias_init)

            for _ in range(3):
                rand_input = torch.randn(self.prev_mask[i].shape)
                diff_total += torch.abs(transform_fn(rand_input) - bias_init)

            EPS = 1e-5
            likely_disconnected = diff_total < EPS
            likely_disconnected &= self.prev_mask[i+1]  # ignore disabled neurons

            if likely_disconnected.any():
                sus_indices_flat = likely_disconnected.flatten().nonzero(as_tuple=True)[0]
                N = len(sus_indices_flat)
                unit_vectors = torch.zeros(N, bias_init.numel())
                unit_vectors[torch.arange(N), sus_indices_flat] = 1
                unit_vectors = unit_vectors.reshape(N, *bias_init.shape)
                grads = torch.func.vmap(vjp_fn)(unit_vectors)[0]
                disconnected_mask = (grads == 0).flatten(1).any(dim=1)

                if disconnected_mask.any():
                    disconnected_indices = sus_indices_flat[disconnected_mask]
                    return disconnected_indices

    def set_branching_criterion(self, neurons_first_hidden: Neurons):
        # MAX: branching criterion has to be an argmax (parametrized by several linear transformations)
        data_len = torch.tensor(len(self.data_idx))
        candidates = []

        # MAX: best candidate minimizes the distance between min and max num of activations
        for i, neurons_indices in neurons_first_hidden.items():
            num_active = self.num_active[i][neurons_indices]
            minimum = torch.maximum(num_active, data_len-num_active).min(0)
            global_index = neurons_indices[minimum.indices]  # we used neurons_indices to select a subset of neurons
            candidates.append((minimum.values, i, global_index))

        _, layer, neuron = min(candidates)
        self.decision_weight = self.input_weight[layer][neuron]
        self.decision_bias = self.input_bias_init[layer][neuron] + self.input_bias_addit[layer][neuron]

    def create_children(self) -> list['BuildingNode']:
        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.batch_size)
        branch_list = []
        num_branches = 2

        for inputs, _ in classerr.utils.progress(loader):
            # MAX: modify criterion
            sat_cond = inputs.flatten(1) @ self.decision_weight + self.decision_bias >= 0
            branch_list.append(sat_cond)

        branching = torch.cat(branch_list)
        children = []

        for b in range(num_branches):
            children.append(self.make_child(self.data_idx[branching == b], str(b)))

        return children


class Node:
    def __init__(self, building_node: BuildingNode):
        self.is_leaf = False
        self.output_shape = building_node.output_shape

        # inner nodes
        self.decision_weight = building_node.decision_weight
        self.decision_bias = building_node.decision_bias
        self.children: list[Node] = []

        # leaves
        self.input_weight: torch.Tensor | None = None
        self.input_bias: torch.Tensor | None = None

        if building_node.children:
            self.children = [Node(child) for child in building_node.children]
        else:
            self.is_leaf = True
            self.input_weight = building_node.input_weight[-1]
            self.input_bias = building_node.input_bias_init[-1] + building_node.input_bias_addit[-1]

    def __call__(self, xs: torch.Tensor):
        assert xs.shape[0] == 1, "batch size has to be one"

        if self.is_leaf:
            return F.linear(xs.flatten(1), self.input_weight, self.input_bias).reshape(self.output_shape)

        sat_cond = xs.flatten(1) @ self.decision_weight + self.decision_bias >= 0
        idx = int(sat_cond)
        return self.children[idx](xs)


def build_tree(model: classerr.trainable.TrainableModel, dataset: classerr.utils.Dataset):
    used_length = min(len_ds(dataset), 50)
    root = BuildingNode(dataset, model.layers, model.batch_size, torch.arange(used_length))
    stack = [root]

    with tqdm.tqdm(total=used_length-1) as pbar:  # to split N items, we will make at most N-1 splits
        while stack:
            node = stack.pop()
            node.build()

            if node.children:
                stack.extend(reversed(node.children))
                pbar.update(1)

    tree = Node(root)

    loader = torch.utils.data.DataLoader(root.subset, batch_size=1)

    for inputs, _ in classerr.utils.progress(loader):
        torch.testing.assert_close(tree(inputs), model(inputs))

    print("all correct")
