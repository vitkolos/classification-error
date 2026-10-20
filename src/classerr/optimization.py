from __future__ import annotations
from dataclasses import dataclass

import torch
import numpy as np

import classerr.neurons
import classerr.solving
from classerr.solving import Polytope, LinearProgram, PolytopeHashable, OptimizationResult
from classerr.utils import Bounds


def analyze_linear_region(
    bounds: Bounds,
    model_orig: torch.nn.Module,
    model_approx: torch.nn.Module,
    sample: torch.Tensor
):
    lp_from_net(model_orig, bounds, sample)


def lp_from_net(net: torch.nn.Module, bounds: Bounds, sample: torch.Tensor) -> LinearProgram:
    """'sample' needs to be a single sample (not a batch)"""
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(sample)
    message = classerr.neurons.collect(net, message, constraints)
    return lp_from_collected(message, constraints, bounds)


def lp_from_collected(message: classerr.neurons.Message, constraints: classerr.neurons.Constraints, bounds: Bounds) -> LinearProgram:
    TOL = 0  # 1e-8
    A_ub = []
    b_ub = []

    # (U)  Ax + b >= 0
    #         -Ax <= b
    if constraints.U_weight:
        A_ub.append(-torch.cat(constraints.U_weight))
        b_ub.append(torch.cat(constraints.U_bias) + TOL)

    # (S)  Ax + b <= 0
    #          Ax <= -b
    if constraints.S_weight:
        A_ub.append(torch.cat(constraints.S_weight))
        b_ub.append(-torch.cat(constraints.S_bias) + TOL)

    objective = message.s_weight.squeeze()
    bias = message.s_bias.item()

    return LinearProgram(bounds, torch.cat(A_ub), torch.cat(b_ub), objective, bias, maximize=True, neuron_states=constraints.neuron_states)
