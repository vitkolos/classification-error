from __future__ import annotations
from dataclasses import dataclass
import time

import torch
import numpy as np

import classerr.neurons
import classerr.solving
from classerr.solving import Polytope, LinearProgram, PolytopeHashable, OptimizationResult
from classerr.utils import Bounds, Metadata

NUM_DIRECTIONS = 100


def analyze_linear_region(
    metadata: Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    point: torch.Tensor,
    target: int,
):
    start_time = time.time()
    point_max_err = find_error_maximum(metadata, model_orig, model_aprx, point, target)
    print(time.time() - start_time, "seconds")
    breakpoint()


def find_error_maximum(
    metadata: Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    point: torch.Tensor,
    target: int,
) -> torch.Tensor:
    lp_orig = lp_from_net(model_orig, metadata.bounds, point)
    lp_aprx = lp_from_net(model_aprx, metadata.bounds, point)
    A_cl, b_cl = constrain_class(target, lp_orig.objective, lp_orig.bias)
    A_ub = torch.cat([lp_orig.A_ub, lp_aprx.A_ub, A_cl])
    b_ub = torch.cat([lp_orig.b_ub, lp_aprx.b_ub, b_cl])
    results = []
    objectives = []

    for k in range(metadata.num_classes):
        if k != target:
            objective, bias = objective_class_error(k, target, lp_aprx.objective, lp_aprx.bias)
            objectives.append(objective)

    lp = LinearProgram(metadata.bounds, A_ub, b_ub, objective=torch.empty(0))
    results = classerr.solving.solve(lp, multiple_objectives=torch.stack(objectives))
    # results.append((fun, k, x))

    # fun_max, k_max, x_max = max(results)
    # return x_max
    return results


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


def constrain_class(class_dominant: int, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = weight.shape[1]
    A_ub, b_ub = [], []

    for class_k in range(num_classes):
        if class_k != class_dominant:
            # excitation_k <= excitation_dominant
            #   (w_k-w_d)x <= b_d-b_k
            A_ub.append(weight[:, class_k]-weight[:, class_dominant])
            b_ub.append(bias[:, class_dominant]-bias[:, class_k])

    return torch.stack(A_ub), torch.cat(b_ub)


def objective_class_error(class_k: int, class_dominant: int, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if class_k == class_dominant:
        raise ValueError

    # we want to maximize excitation_k-excitation_c
    # so we maximize (w_k-w_d)x + (b_k-b_d)
    weight = weight[:, class_k]-weight[:, class_dominant]
    bias = bias[:, class_k]-bias[:, class_dominant]
    return weight, bias


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
