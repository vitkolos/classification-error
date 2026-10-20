from __future__ import annotations
from dataclasses import dataclass

import torch
import numpy as np

import classerr.neurons
import classerr.solving
from classerr.solving import Polytope, LinearProgram, PolytopeHashable, OptimizationResult
from classerr.utils import Bounds

NUM_DIRECTIONS = 100


def analyze_linear_region(
    bounds: Bounds,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    point: torch.Tensor,
    target: int,
):
    lp_orig = lp_from_net(model_orig, bounds, point)
    lp_aprx = lp_from_net(model_aprx, bounds, point)
    A_cl, b_cl = constrain_class(target, lp_orig.objective, lp_orig.bias)
    breakpoint()


def lp_from_net(net: torch.nn.Module, bounds: Bounds, point: torch.Tensor) -> LinearProgram:
    """'point' needs to be a single input (not a batch)"""
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(point)
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

    objective = message.s_weight
    bias = message.s_bias

    return LinearProgram(bounds, torch.cat(A_ub), torch.cat(b_ub), objective, bias, maximize=True, neuron_states=constraints.neuron_states)


def constrain_class(dominant: int, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = weight.shape[1]
    A_ub, b_ub = [], []

    for k in range(num_classes):
        if k == dominant:
            continue

        # excitation_k <= excitation_dominant
        #   (w_k-w_d)x <= b_d-b_k
        A_ub.append(weight[:, k]-weight[:, dominant])
        b_ub.append(bias[:, dominant]-bias[:, k])

    return torch.stack(A_ub), torch.cat(b_ub)


def polytope_widths(polytope: Polytope, num_directions: int = NUM_DIRECTIONS, cummulative_avg: bool = False, num_jobs: int = 1) -> torch.Tensor:
    """returns widths of the polytope computed from many random directions (or the cummulative average in each step)"""
    # variables == dimensions
    num_variables = polytope.A_ub.shape[1]
    directions = torch.randn(num_directions, num_variables)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    lp = LinearProgram(polytope.bounds, polytope.A_ub, polytope.b_ub, objective=torch.empty(0))
    results = classerr.solving.solve_parallel(lp, directions, num_jobs)
    widths = torch.tensor([(res_max.fun - res_min.fun) for res_min, res_max in results])
    return widths if not cummulative_avg else widths.cumsum(dim=0) / torch.arange(1, num_directions+1)
