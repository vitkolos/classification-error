import math
import torch
import torch.nn as nn

import classerr.neurons
import classerr.optimization
import classerr.solving
import classerr.trainable
from classerr.utils import Bounds
import tests.test_neurons


def optimize_testing_procedure(net: nn.Module, sample: torch.Tensor, bounds: Bounds, mixing: float = 0.5):
    net_collectable = net.layers if isinstance(net, classerr.trainable.TrainableModel) else net
    # constraints = classerr.neurons.Constraints()
    # message = classerr.neurons.Message(sample)
    # message = classerr.neurons.collect(net_collectable, message, constraints)
    # lp = classerr.optimization.lp_from_collected(message, constraints, bounds)
    lp = classerr.optimization.lp_from_net(net_collectable, bounds, sample)
    sample_found, err_found = classerr.solving.solve(lp)
    sample_found = sample_found.reshape_as(sample)
    sample_comb = (1-mixing)*sample_found + mixing*sample

    # check feasibility
    classerr.optimization.assert_inside_polytope(sample, lp)
    classerr.optimization.assert_inside_polytope(sample_found, lp)
    classerr.optimization.assert_inside_polytope(sample_comb, lp)

    # check found objective value
    err_computed = net(sample_found.unsqueeze(0)).item()
    assert math.isclose(err_found, err_computed, abs_tol=1e-6), 'Optimization returns different value than the network'

    # check if we get the same linear program for a point on the polytope
    # constraints_point = classerr.neurons.Constraints()
    # message_point = classerr.neurons.Message(sample_comb)
    # message_point = classerr.neurons.collect(net_collectable, message_point, constraints_point)
    # lp_point = classerr.optimization.lp_from_collected(message_point, constraints_point, bounds)
    lp_point = classerr.optimization.lp_from_net(net_collectable, bounds, sample_comb)

    for t1, t2 in zip(lp.neuron_states, lp_point.neuron_states):
        torch.testing.assert_close(t1, t2, msg='States of neurons differ')

    for attr in ['A_ub', 'b_ub', 'objective', 'bias']:
        torch.testing.assert_close(getattr(lp, attr), getattr(lp_point, attr), msg=f'Linear programs differ in {attr}')


def test_deeper_appmax():
    net = tests.test_neurons.DummyNetDeeper().eval()
    sample = torch.tensor([2.0, 1.0])
    optimize_testing_procedure(net, sample, bounds=Bounds([(-0.5, 3.0)]*2))


class DummyOptNetMaxPool(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
            )
        )


def test_max_pool_lp():
    net = DummyOptNetMaxPool().eval()
    # 1 channel, 4×4 input shape
    sample = torch.testing.make_tensor((1, 4, 4), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    optimize_testing_procedure(net, sample, bounds=Bounds([(-2.0, 3.0)]*16), mixing=0.01)


class DummyOptNetConv(classerr.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Conv2d(1, 2, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(8, 1)
            )
        )


def test_conv_lp():
    """relies on random initialization of parameters"""
    net = DummyOptNetConv().eval()
    # 1 channel, 4×4 input shape
    sample = torch.testing.make_tensor((1, 4, 4), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    optimize_testing_procedure(net, sample, bounds=Bounds([(-2.0, 3.0)]*16), mixing=0.01)


def test_batch_norm1d_lp():
    net = tests.test_neurons.DummyNetBatchNorm1d().eval()
    sample = torch.testing.make_tensor(16, dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    optimize_testing_procedure(net, sample, bounds=Bounds([(-2.0, 3.0)]*16), mixing=0.01)
