import dataclasses
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

LAYERS_NONLINEAR = (nn.ReLU, nn.MaxPool2d)
LAYERS_LINEAR = (
    # actual linear projections
    nn.Linear, nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm,
    # other kinds of layers not violating linearity
    nn.Flatten, nn.Dropout,
)


@dataclasses.dataclass
class Nonlinearity:
    module: torch.nn.Module
    num_branches: int
    idx_map: torch.Tensor


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

        self.input_tensors_linked: list[bool] = []
        self.input_weight: list[torch.Tensor] = []
        self.input_bias_init: list[torch.Tensor] = []  # initial bias (present in the original model)
        self.input_bias_addit: list[torch.Tensor] = []  # additional bias (collected from the ancestors)
        self.prev_mask: list[torch.Tensor] = []

        self.nonlinearities: list[Nonlinearity] = []
        self.num_active: list[torch.Tensor] = []

        self.decision_weight: torch.Tensor | None = None
        self.decision_bias: torch.Tensor | None = None
        self.children: list[BuildingNode] = []

    @staticmethod
    def raise_not_implemented(module: torch.nn.Module) -> typing.Never:
        raise NotImplementedError(
            f"this method in BuildingNode is not implemented for '{type(module).__name__}' module")

    def make_child(self, data_idx: torch.Tensor, name_suffix: str = "") -> 'BuildingNode':
        child = BuildingNode(self.dataset, self.model, self.batch_size, data_idx)
        child.name = self.name + name_suffix
        child.linear_seqs = self.linear_seqs
        child.output_shape = self.output_shape
        # we refer to the parent's tensors in the children (and perform copy-on-write)
        child.input_tensors_linked = [True] * len(self.input_tensors_linked)
        # so these are just shallow copies
        child.input_weight = self.input_weight[:]
        child.input_bias_init = self.input_bias_init[:]
        child.input_bias_addit = self.input_bias_addit[:]
        child.prev_mask = self.prev_mask[:]
        return child

    @property
    def num_hidden_layers(self):
        """number of hidden layers before reduction"""
        return len(self.nonlinearities)

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
            self.decision_weight, self.decision_bias = self.find_criterion(neurons_first_hidden)
            # create children + split data_idx and assign them to the children according to the decision criterion
            self.children = self.create_children()

        self.dump_useless_tensors()

    def dump_useless_tensors(self):
        """is called at the end of the build process"""

        if self.children:
            self.input_weight.clear()
            self.input_bias_init.clear()
            self.input_bias_addit.clear()
        else:
            # leaves need the last tensors for prediction (so we keep them)
            del self.input_weight[:-1]
            del self.input_bias_init[:-1]
            del self.input_bias_addit[:-1]

        self.prev_mask.clear()
        self.num_active.clear()

    def compute_activity(self):
        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.batch_size)

        for inputs, _ in classerr.utils.progress(loader):
            self.forward_count(inputs)

    def forward_count(self, inputs: torch.Tensor) -> torch.Tensor:
        xs = inputs
        last_nonlinearity = -1
        i = 0
        xs = self.disable_connections(i, xs)

        for j, module in enumerate(self.model):
            if isinstance(module, LAYERS_NONLINEAR):
                self.store_linear_seq(i, last_nonlinearity, j)
                xs = self.apply_linearity(i, xs, inputs)
                self.store_nonlinearity(i, xs, module)
                self.increment_activity_counters(i, xs)
                xs = module(xs)
                i += 1
                last_nonlinearity = j
                xs = self.disable_connections(i, xs)
            elif isinstance(module, LAYERS_LINEAR):
                xs = module(xs)
            else:
                BuildingNode.raise_not_implemented(module)

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

    def store_linear_seq(self, i: int, last_nonlinearity: int, j: int):
        if len(self.linear_seqs) == i:
            self.linear_seqs.append(self.model[last_nonlinearity+1:j])

    def store_nonlinearity(self, i: int, xs: torch.Tensor, module: torch.nn.Module):
        if len(self.nonlinearities) == i:
            num_branches: int | None = None
            idx_map: torch.Tensor | None = None

            match module:
                case nn.ReLU():
                    num_branches = 2
                    idx_map = torch.arange(xs[0].numel()).unsqueeze(1)
                case nn.MaxPool2d() as maxpool:
                    _, C, H, W = xs.shape
                    idx_grid = torch.arange(C * H * W).float().reshape(1, C, H, W)
                    idx_patches = F.unfold(
                        idx_grid,
                        maxpool.kernel_size,
                        maxpool.dilation,
                        maxpool.padding,
                        maxpool.stride,
                    ).long()
                    num_branches = idx_patches.shape[1] // C
                    idx_map = (
                        idx_patches.squeeze(0)  # [C * num_branches, L]
                        .unflatten(0, (C, num_branches))  # [C, num_branches, L]
                        .movedim(1, -1)  # [C, L, num_branches]
                        .flatten(0, 1)  # [C * L, num_branches]
                    )
                case _:
                    BuildingNode.raise_not_implemented(module)

            self.nonlinearities.append(Nonlinearity(module, num_branches, idx_map))

    def store_output_shape(self, shape: torch.Size):
        if self.output_shape is None:
            self.output_shape = torch.Size((1, *shape[1:]))

    def apply_linearity(self, i: int, xs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if i == len(self.input_weight):
            single_input_shape = torch.Size((1, *inputs.shape[1:]))
            single_output_shape = torch.Size((1, *xs.shape[1:]))
            w_shape = (single_output_shape.numel(), single_input_shape.numel())
            b_shape = (single_output_shape.numel(),)
            self.input_tensors_linked.append(False)
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

    def increment_activity_counters(self, i: int, xs: torch.Tensor):
        nl = self.nonlinearities[i]

        if i == len(self.num_active):
            self.num_active.append(torch.zeros((nl.idx_map.shape[0], nl.num_branches), dtype=torch.int))

        update: torch.Tensor | None = None

        match nl.module:
            case nn.ReLU():
                relu_linear = xs.flatten(1) >= 0
                relu_linear_num = relu_linear.sum(dim=0)
                relu_zero_num = len(xs)-relu_linear_num
                update = torch.stack((relu_linear_num, relu_zero_num), dim=1)
            case nn.MaxPool2d() as maxpool:
                patches = F.unfold(xs, maxpool.kernel_size, maxpool.dilation, maxpool.padding, maxpool.stride)
                windows = patches.unflatten(1, (xs.shape[1], nl.num_branches))  # [N, C, B, L]
                argmax_idx = windows.argmax(dim=2).flatten(1)  # [N, C*L]
                update = F.one_hot(argmax_idx, num_classes=nl.num_branches).sum(dim=0)
            case _:
                BuildingNode.raise_not_implemented(nl.module)

        self.num_active[i] += update

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
                # reducible neurons have all zeros except for one => their maximum is data_len
                maxima = num_active.max(dim=1)
                reducible = (maxima.values == data_len)

                if reducible.any():
                    self.reduce_layer(i, neurons_indices, reducible, maxima.indices)
                    to_be_rescanned.add(i)  # remove the reduced neurons from the list

                    if i+1 < self.num_hidden_layers:  # check if there's a next layer
                        # now, the neurons in the next layer may belong in the FHL
                        to_be_rescanned.add(i+1)

        return neurons_first_hidden

    def reduce_layer(self, i: int, neurons_indices: torch.Tensor, reducible: torch.Tensor, argmaxima: torch.Tensor):
        reducible_idx = neurons_indices[reducible]
        nl = self.nonlinearities[i]

        linear_out_idx: torch.Tensor | None = None
        linear_weight: torch.Tensor | None = None
        linear_bias: torch.Tensor | None = None

        match nl.module:
            case nn.ReLU():
                linear_out_idx = neurons_indices[reducible & (argmaxima == 0)]
                linear_in_idx = linear_out_idx
            case nn.MaxPool2d():
                linear_out_idx = reducible_idx
                linear_in_idx = nl.idx_map[reducible_idx, argmaxima[reducible]]
            case _ as m:
                BuildingNode.raise_not_implemented(m)

        linear_weight = self.input_weight[i][linear_in_idx]
        linear_bias = self.input_bias_init[i][linear_in_idx] + self.input_bias_addit[i][linear_in_idx]

        self.copy_tensors_before_write(i+1)  # copy-on-write

        if linear_out_idx.numel() > 0:
            for idx, w, b in zip(
                linear_out_idx.split(self.batch_size),
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

    def copy_tensors_before_write(self, i: int):
        if self.input_tensors_linked[i]:
            self.input_weight[i] = self.input_weight[i].clone()
            self.input_bias_addit[i] = self.input_bias_addit[i].clone()
            self.prev_mask[i] = self.prev_mask[i].clone()
            self.input_tensors_linked[i] = False

    def find_neurons_first_hidden(self, layers: typing.Iterable[int], neurons: Neurons):
        """
        A neuron is in the first hidden layer iff none of its ancestors is a hidden neuron.
        In our model, it suffices to verify that the neuron:
        1) is not disabled,
        2) is in the first layer or has no connections with the previous layer.
        """
        for i in layers:
            found = self.find_neurons_first_hidden_in_layer(i)

            if found is not None:
                neurons[i] = found
            else:
                neurons.pop(i, None)

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

            match self.nonlinearities[i].module:
                case nn.ReLU():
                    # 1 to 1 pairing
                    enabled_inputs = self.prev_mask[i+1]
                case nn.MaxPool2d() as maxpool:
                    ks = maxpool.kernel_size
                    k_size = (ks, ks) if isinstance(ks, int) else ks
                    weight = torch.ones((1, 1, *k_size))
                    # transpose(0, 1) allows us to allocate weight tensor with just a single channel
                    enabled_outputs = self.prev_mask[i+1].transpose(0, 1).float()  # [C, 1, H, W]
                    enabled_inputs = F.conv_transpose2d(
                        enabled_outputs,
                        weight,
                        stride=maxpool.stride,
                        padding=maxpool.padding,
                        dilation=maxpool.dilation,
                    ).transpose(0, 1)
                    height, width = likely_disconnected.shape[2:]
                    enabled_inputs = enabled_inputs[:, :, :height, :width] > 0  # crop if necessary and convert to bool
                case _ as m:
                    BuildingNode.raise_not_implemented(m)

            likely_disconnected &= enabled_inputs  # ignore inputs corresponding to disabled neurons

            if likely_disconnected.any():
                sus_indices_flat = likely_disconnected.flatten().nonzero(as_tuple=True)[0]
                numel = bias_init.numel()
                disconnected_masks = []

                for sus_indices_chunk in sus_indices_flat.split(self.batch_size):
                    N = len(sus_indices_chunk)
                    unit_vectors = torch.zeros((N, numel))
                    unit_vectors[torch.arange(N), sus_indices_chunk] = 1
                    unit_vectors = unit_vectors.reshape(N, *bias_init.shape)
                    grads = torch.func.vmap(vjp_fn)(unit_vectors)[0]
                    batch_disconnected = (grads == 0).flatten(1).any(dim=1)
                    disconnected_masks.append(batch_disconnected)

                disconnected_mask = torch.cat(disconnected_masks)
                disconnected_indices = sus_indices_flat[disconnected_mask]

                if isinstance(self.nonlinearities[i].module, nn.MaxPool2d):
                    maxpool = self.nonlinearities[i].module
                    reconstructed_mask = torch.zeros(bias_init.shape, dtype=torch.bool)
                    reconstructed_mask.view(-1)[disconnected_indices] = True
                    reconstructed_mask = (-maxpool(-reconstructed_mask.float())).bool()  # conjunction
                    disconnected_indices = reconstructed_mask.flatten().nonzero(as_tuple=True)[0]

                if disconnected_indices.numel() > 0:
                    return disconnected_indices

        return None

    def find_criterion(self, neurons_first_hidden: Neurons) -> tuple[torch.Tensor, torch.Tensor]:
        data_len = torch.tensor(len(self.data_idx))
        candidates = []

        # best candidate minimizes the distance (spread) between non-zero min num_active and max num_active
        # note that each candidate has at least two non-zero values and that the values sum up to data_len
        for i, neurons_indices in neurons_first_hidden.items():
            num_active = self.num_active[i][neurons_indices]
            masked = torch.where(num_active == 0, data_len, num_active)
            spread = num_active.max(dim=1).values - masked.min(dim=1).values
            minimum = spread.min(0)
            global_index = neurons_indices[minimum.indices]  # we used neurons_indices to select a subset of neurons
            candidates.append((minimum.values, i, global_index))

        _, layer, neuron = min(candidates)
        nl = self.nonlinearities[layer]

        # returned decision_weight and decision_bias should be copies, not views
        match nl.module:
            case nn.ReLU():
                weight = self.input_weight[layer][neuron]
                bias = self.input_bias_init[layer][neuron] + self.input_bias_addit[layer][neuron]
                decision_weight = torch.stack((weight, torch.zeros_like(weight)))
                decision_bias = torch.stack((bias, torch.zeros_like(bias)))
                return decision_weight, decision_bias
            case nn.MaxPool2d():
                in_index = nl.idx_map[neuron]
                decision_weight = self.input_weight[layer][in_index]
                decision_bias = self.input_bias_init[layer][in_index] + self.input_bias_addit[layer][in_index]
                active = self.num_active[layer][neuron] > 0
                return decision_weight[active], decision_bias[active]
            case _ as m:
                BuildingNode.raise_not_implemented(m)

    def create_children(self) -> list['BuildingNode']:
        loader = torch.utils.data.DataLoader(self.subset, batch_size=self.batch_size)
        branch_list = []
        assert self.decision_weight is not None
        num_branches = self.decision_weight.shape[0]

        for inputs, _ in classerr.utils.progress(loader):
            argmaxima = F.linear(inputs.flatten(1), self.decision_weight, self.decision_bias).argmax(dim=1)
            branch_list.append(argmaxima)

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
            self.input_weight = building_node.input_weight[-1].clone()
            self.input_bias = building_node.input_bias_init[-1] + building_node.input_bias_addit[-1]

    def __call__(self, xs: torch.Tensor):
        if xs.shape[0] != 1:
            raise classerr.utils.AppError("batch size has to be one")

        if self.is_leaf:
            assert self.input_weight is not None and self.input_bias is not None and self.output_shape is not None
            return F.linear(xs.flatten(1), self.input_weight, self.input_bias).reshape(self.output_shape)

        assert self.decision_weight is not None and self.decision_bias is not None
        branch = F.linear(xs.flatten(1), self.decision_weight, self.decision_bias).argmax(dim=1)
        branch_idx = int(branch)
        return self.children[branch_idx](xs)


def build_tree(model: classerr.trainable.TrainableModel, dataset: classerr.utils.Dataset, max_len: int = 50) -> Node:
    used_length = min(len_ds(dataset), max_len)
    building_root = BuildingNode(dataset, model.layers, model.batch_size, torch.arange(used_length))
    stack = [building_root]

    with tqdm.tqdm(total=used_length-1) as pbar:  # to split N items, we will make at most N-1 splits
        while stack:
            node = stack.pop()
            node.build()

            if node.children:
                stack.extend(reversed(node.children))
                pbar.update(1)

    tree = Node(building_root)
    loader = torch.utils.data.DataLoader(building_root.subset, batch_size=1)

    for inputs, _ in classerr.utils.progress(loader):
        torch.testing.assert_close(tree(inputs), model(inputs))

    return tree
