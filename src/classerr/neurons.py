from dataclasses import dataclass, field
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Message:
    point: torch.Tensor
    s_weight: torch.Tensor
    s_bias: torch.Tensor

    def __init__(self, point: torch.Tensor):
        """'point' needs to be a single point (not a batch)"""
        self.point = point.unsqueeze(0)  # batch-like
        # create a unit matrix with a shape corresponding to the input
        self.s_weight = torch.eye(point.numel()).reshape((-1, *point.shape))
        self.s_bias = torch.zeros_like(self.point)

    def apply(self, module: nn.Module):
        self.point = module(self.point)
        self.s_weight = module(self.s_weight)
        self.s_bias = module(self.s_bias)
        return self

    def cat(self, other: 'Message'):
        self.point = torch.cat((self.point, other.point), dim=-1)
        self.s_weight = torch.cat((self.s_weight, other.s_weight), dim=-1)
        self.s_bias = torch.cat((self.s_bias, other.s_bias), dim=-1)
        return self


@dataclass
class Constraints:
    U_weight: list[torch.Tensor] = field(default_factory=list)
    U_bias: list[torch.Tensor] = field(default_factory=list)
    S_weight: list[torch.Tensor] = field(default_factory=list)
    S_bias: list[torch.Tensor] = field(default_factory=list)
    neuron_states: list[torch.Tensor] = field(default_factory=list)


@torch.no_grad()
def collect(module: nn.Module, message: Message, constraints: Constraints) -> Message:
    """passes message through the module and collects necessary information;
    note that the message is modified **in place**"""
    match module:
        case nn.Sequential():
            for submodule in module:
                message = collect(submodule, message, constraints)
            return message
        case nn.ReLU():
            return collect_relu(module, message, constraints)
        case nn.Linear():
            return collect_linear(module, message, constraints)
        case nn.BatchNorm1d():
            return collect_batch_norm1d(module, message, constraints)
        case nn.Conv2d():
            return collect_conv2d(module, message, constraints)
        case nn.MaxPool2d():
            return collect_max_pool2d(module, message, constraints)
        case nn.Dropout():
            if module.training:
                raise RuntimeError('Dropout layer is in training mode')
            return message
        case nn.Flatten():
            return message.apply(module)

    raise NotImplementedError(f"neurons.collect is not implemented for '{type(module).__name__}' object")


def collect_relu(relu: nn.ReLU, message: Message, constraints: Constraints) -> Message:
    # find unsaturated neurons
    unsaturated = message.point >= 0
    # - non-strict inequality ensures that the unsaturated set stays the same for all the points in the polytope
    # - strict inequality would cause smaller unsaturated set for points from some of the edges of the polytope
    #   -> it would be possible to "move" from one polytope to another (but only in one direction)
    unsaturated_sq = unsaturated.squeeze(0)
    message.point = relu(message.point)

    # add constraints
    s_weight_T = message.s_weight.movedim(0, -1)
    constraints.U_weight.append(s_weight_T[unsaturated_sq])
    constraints.U_bias.append(message.s_bias[unsaturated])
    constraints.S_weight.append(s_weight_T[~unsaturated_sq])
    constraints.S_bias.append(message.s_bias[~unsaturated])

    # disable saturated neurons
    message.s_weight *= unsaturated
    message.s_bias *= unsaturated

    # log states of neurons
    constraints.neuron_states.append(unsaturated_sq)

    return message


def collect_linear(linear: nn.Linear, message: Message, constraints: Constraints) -> Message:
    message.point = linear(message.point)

    # s_weight = s_weight @ weight.t()
    message.s_weight = F.linear(message.s_weight, linear.weight, None)

    # s_bias = s_bias @ weight.t() + bias
    message.s_bias = linear(message.s_bias)

    return message


def collect_batch_norm1d(batch_norm1d: nn.BatchNorm1d, message: Message, constraints: Constraints) -> Message:
    if batch_norm1d.training:
        raise RuntimeError('BatchNorm1d layer is in training mode')
    elif not batch_norm1d.track_running_stats or batch_norm1d.running_mean is None:
        raise NotImplementedError('collect_batch_norm1d does not support track_running_stats=False')

    message.point = batch_norm1d(message.point)
    message.s_weight = F.batch_norm(
        message.s_weight,
        running_mean=torch.zeros_like(batch_norm1d.running_mean),
        running_var=batch_norm1d.running_var,
        weight=batch_norm1d.weight,
        bias=None,
        training=False,
        momentum=0.0,  # does not matter
        eps=batch_norm1d.eps,
    )
    message.s_bias = batch_norm1d(message.s_bias)
    return message


def collect_conv2d(conv2d: nn.Conv2d, message: Message, constraints: Constraints) -> Message:
    message.point = conv2d(message.point)
    message.s_weight = conv2d._conv_forward(message.s_weight, conv2d.weight, None)
    message.s_bias = conv2d(message.s_bias)
    return message


def batch_channels_take(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size = data.shape[0]
    channels = data.shape[1]
    batch_indices = indices.flatten(2).expand(batch_size, -1, -1)
    gathered = torch.gather(data.flatten(2), dim=2, index=batch_indices)
    return gathered.reshape(batch_size, channels, *indices.shape[2:])


def collect_max_pool2d(max_pool2d: nn.MaxPool2d, message: Message, constraints: Constraints) -> Message:
    if max_pool2d.ceil_mode:
        raise NotImplementedError('collect_max_pool2d does not support ceil_mode=True')

    _, C, M, N = message.point.shape
    message.point, indices_max = F.max_pool2d(
        message.point,
        max_pool2d.kernel_size,
        max_pool2d.stride,
        max_pool2d.padding,
        max_pool2d.dilation,
        return_indices=True,
    )

    # add constraints (we require that every non-maximum is less than the maximum)
    # s_weight_T[channel, pixel, input]
    s_weight_T = message.s_weight.movedim(0, -1).flatten(1, 2)
    # s_bias_sq[channel, pixel]
    s_bias_sq = message.s_bias.reshape(C, -1)
    # indices_all[(singleton), window, cell]
    indices_all = F.unfold(
        torch.arange(M*N).reshape(1, 1, M, N).float(),
        kernel_size=max_pool2d.kernel_size,
        stride=max_pool2d.stride,
        padding=max_pool2d.padding,
        dilation=max_pool2d.dilation,
    ).long().movedim(1, -1)
    window_cells = indices_all.shape[-1]
    # indices_all_sq[channel, pixel]
    indices_all_sq = indices_all.flatten().repeat(C)
    # indices_max_sq[channel, pixel]
    indices_max_sq = indices_max.flatten().repeat_interleave(window_cells, dim=-1)
    useful = indices_max_sq != indices_all_sq
    # flat tensors
    indices_max_sq = indices_max_sq[useful]
    indices_all_sq = indices_all_sq[useful]
    channels = torch.arange(C).repeat_interleave(indices_max_sq.shape[0] // C)
    # other + o_bias <= max + m_bias
    # other-max + ob-mb <= 0 (saturated constraint)
    constraints.S_weight.append(s_weight_T[channels, indices_all_sq] - s_weight_T[channels, indices_max_sq])
    constraints.S_bias.append(s_bias_sq[channels, indices_all_sq] - s_bias_sq[channels, indices_max_sq])

    # disable saturated neurons (take maxima)
    message.s_weight = batch_channels_take(message.s_weight, indices_max)
    message.s_bias = batch_channels_take(message.s_bias, indices_max)

    # log states of neurons
    constraints.neuron_states.append(indices_max)

    return message
