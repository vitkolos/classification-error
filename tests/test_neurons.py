import torch
import torch.nn as nn

import classerr.trainable
import classerr.neurons


def net_param(list):
    return nn.Parameter(torch.tensor(list), requires_grad=False)


class DummyNetShallow(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                fc1 := nn.Linear(2, 3),
                nn.ReLU(),
                fc2 := nn.Linear(3, 2),
            )
        )

        with torch.no_grad():
            fc1.weight = net_param([
                [1.0,  1.0],  # hidden neuron 1
                [-1.0, -1.0],  # hidden neuron 2
                [0.5, -0.5]   # hidden neuron 3
            ])
            fc1.bias = net_param([0.5, 0.0, 0.0])
            fc2.weight = net_param([
                [1.0, 1.0, 1.0],  # output neuron 1
                [1.0, 0.0, 0.0]  # output neuron 2
            ])
            fc2.bias = net_param([0.0, 0.0])


class DummyNetDeeper(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                fc1 := nn.Linear(2, 3),
                nn.ReLU(),
                fc2 := nn.Linear(3, 2),
                nn.ReLU(),
                fc3 := nn.Linear(2, 1, bias=False),
            )
        )

        with torch.no_grad():
            fc1.weight = net_param([
                [1.0,  1.0],  # H1.1
                [-1.0, -1.0],  # H1.2
                [0.5, -0.5]   # H1.3
            ])
            fc1.bias = net_param([0.5, 0, 0])
            fc2.weight = net_param([
                [1.0, 1.0, 1.0],  # H2.1
                [1.0, 0.0, 0.0]  # H2.2
            ])
            fc2.bias = net_param([0.25, 0.25])
            fc3.weight = net_param([
                [1.0, 2.0],  # O1.1
            ])


def test_shallow_neurons():
    net = DummyNetShallow()
    sample = torch.tensor([2.0, 1.0])
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(sample)
    message = classerr.neurons.collect(net.layers, message, constraints)
    output = sample @ message.s_weight + message.s_bias
    assert torch.equal(output, net(sample.unsqueeze(0)))
    assert torch.equal(output, message.point)
    assert torch.equal(constraints.S_weight[0], torch.tensor([[-1., -1.]])), \
        'weights corresponding to the (saturated) hidden neuron 2 should be (-1, -1)'


def test_deeper_constraints():
    net = DummyNetDeeper()
    sample = torch.tensor([2.0, 1.0])
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(sample)
    message = classerr.neurons.collect(net.layers, message, constraints)
    output = sample @ message.s_weight + message.s_bias
    assert torch.equal(output, net(sample.unsqueeze(0)))
    assert torch.equal(output, message.point)
    assert torch.equal(constraints.U_weight[1][0], torch.tensor([1.5, 0.5])), \
        'weights corresponding to the (unsaturated) neuron H2.1 should be (1.0 + 0.5, 1.0 - 0.5)'
    assert torch.equal(constraints.U_bias[1][0], torch.tensor(0.75)), \
        'bias corresponding to the (unsaturated) neuron H2.1 should be 0.5 + 0.25'


class DummyNetFC(classerr.trainable.TrainableModel):
    def __init__(self):
        """relies on random initialization of parameters"""
        super().__init__(
            nn.Sequential(
                nn.Linear(8, 24),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LazyLinear(48),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LazyLinear(61),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LazyLinear(3),
            )
        )


def test_fc_neurons_random():
    net = DummyNetFC().eval()
    sample = torch.testing.make_tensor(8, dtype=torch.float32, device='cpu', low=0.0, high=2.0)
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(sample)
    message = classerr.neurons.collect(net.layers, message, constraints)
    output = sample @ message.s_weight + message.s_bias
    torch.testing.assert_close(output, net(sample.unsqueeze(0)))
    torch.testing.assert_close(output, message.point)


class DummyNetBatchNorm1d(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                bn := nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        )

        with torch.no_grad():
            bn.weight.normal_(1, 0.5)
            bn.bias.normal_(0, 0.5)
            bn.running_mean.normal_(0, 1)
            bn.running_var.uniform_(0, 1)  # we need a positive variance


def test_batch_norm1d_neurons_random():
    net = DummyNetBatchNorm1d().eval()
    sample = torch.testing.make_tensor(16, dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(sample)
    message = classerr.neurons.collect(net.layers, message, constraints)
    output = sample @ message.s_weight + message.s_bias
    torch.testing.assert_close(output, net(sample.unsqueeze(0)))
    torch.testing.assert_close(output, message.point)


class DummyNetConvPure(classerr.trainable.TrainableModel):
    def __init__(self):
        """relies on random initialization of parameters"""
        super().__init__(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.LazyConv2d(2, kernel_size=3),
                nn.Flatten(),
            )
        )


def test_conv_pure_neurons_random():
    net = DummyNetConvPure().eval()
    sample = torch.testing.make_tensor((1, 16, 16), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(sample)
    message = classerr.neurons.collect(net.layers, message, constraints)
    output = sample.flatten(1) @ message.s_weight + message.s_bias
    torch.testing.assert_close(output, net(sample.unsqueeze(0)))
    torch.testing.assert_close(output, message.point)


class DummyNetMaxPool(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.MaxPool2d(3, 2),
                nn.MaxPool2d((3, 2), (2, 1)),
                nn.Flatten(),
            )
        )


def test_max_pool_neurons_random():
    net = DummyNetMaxPool().eval()
    sample = torch.testing.make_tensor((1, 32, 32), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(sample)
    message = classerr.neurons.collect(net.layers, message, constraints)
    output = sample.flatten(1) @ message.s_weight + message.s_bias
    torch.testing.assert_close(output, net(sample.unsqueeze(0)))
    torch.testing.assert_close(output, message.point)


def test_max_pool_batch_take():
    sample_old = torch.testing.make_tensor((5, 3, 32, 32), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    sample_new, indices = torch.nn.functional.max_pool2d(sample_old, 2, return_indices=True)
    torch.testing.assert_close(sample_new, classerr.neurons.batch_channels_take(sample_old, indices))
