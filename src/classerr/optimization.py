from __future__ import annotations
import time
import torch

import classerr.neurons
import classerr.solving
from classerr.solving import Polytope, LinearProgram
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
    lp_orig = lp_from_net(model_orig, metadata.bounds, point)
    lp_aprx = lp_from_net(model_aprx, metadata.bounds, point)
    point_max_err = find_error_maximum(metadata, lp_orig, lp_aprx, point, target)
    print(time.time() - start_time, "seconds")
    breakpoint()
    print('computing mean widths')
    mean_widths = compute_classes_mean_widths(metadata, lp_orig, lp_aprx, target)


def lp_from_net(net: torch.nn.Module, bounds: Bounds, point: torch.Tensor) -> LinearProgram:
    """'point' needs to be a single input (not a batch)"""
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(point)
    message = classerr.neurons.collect(net, message, constraints)

    A_ub = []
    b_ub = []

    # (U)  Ax + b >= 0
    #         -Ax <= b
    if constraints.U_weight:
        A_ub.append(-torch.cat(constraints.U_weight))
        b_ub.append(torch.cat(constraints.U_bias))

    # (S)  Ax + b <= 0
    #          Ax <= -b
    if constraints.S_weight:
        A_ub.append(torch.cat(constraints.S_weight))
        b_ub.append(-torch.cat(constraints.S_bias))

    objective = message.s_weight
    bias = message.s_bias

    return LinearProgram(
        bounds,
        torch.cat(A_ub),
        torch.cat(b_ub),
        objective,
        bias,
        maximize=True,
        neuron_states=constraints.neuron_states
    )


def find_error_maximum(
    metadata: Metadata, lp_orig: LinearProgram, lp_aprx: LinearProgram, point: torch.Tensor, target: int,
) -> torch.Tensor:
    A_ub, b_ub = correct_polytope_constraints(lp_orig, lp_aprx, target)
    results = []

    for k in range(metadata.num_classes):
        if k != target:
            objective, bias = objective_class_error(k, target, lp_aprx.objective, lp_aprx.bias)
            lp = LinearProgram(metadata.bounds, A_ub, b_ub, objective, bias, maximize=True)
            x, fun = classerr.solving.solve(lp)
            results.append((fun, k, x))

    fun_max, k_max, x_max = max(results)
    x_max = x_max.reshape_as(point)
    return x_max


def compute_classes_mean_widths(
    metadata: Metadata, lp_orig: LinearProgram, lp_aprx: LinearProgram, target: int
) -> list[float]:
    A_ub, b_ub = correct_polytope_constraints(lp_orig, lp_aprx, target)
    widths = []

    for k in range(metadata.num_classes):
        A_cl, b_cl = constrain_class(k, lp_aprx.objective, lp_aprx.bias)
        polytope = Polytope(metadata.bounds, torch.cat([A_ub, A_cl]), torch.cat([b_ub, b_cl]))
        # TODO: check if polytope is not empty
        widths.append(polytope_widths(polytope).mean().item())

    return widths


def correct_polytope_constraints(
    lp_orig: LinearProgram, lp_aprx: LinearProgram, target: int
) -> tuple[torch.Tensor, torch.Tensor]:
    A_cl, b_cl = constrain_class(target, lp_orig.objective, lp_orig.bias)
    A_ub = torch.cat([lp_orig.A_ub, lp_aprx.A_ub, A_cl])
    b_ub = torch.cat([lp_orig.b_ub, lp_aprx.b_ub, b_cl])
    return A_ub, b_ub


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


def objective_class_error(
    class_k: int, class_dominant: int, weight: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if class_k == class_dominant:
        raise ValueError

    # we want to maximize excitation_k-excitation_c
    # so we maximize (w_k-w_d)x + (b_k-b_d)
    weight = weight[:, class_k]-weight[:, class_dominant]
    bias = bias[:, class_k]-bias[:, class_dominant]
    return weight, bias


def polytope_widths(
    polytope: Polytope, num_directions: int = NUM_DIRECTIONS, cummulative_avg: bool = False, num_jobs: int = 1
) -> torch.Tensor:
    """returns widths of the polytope computed from many random directions (or the cummulative average in each step)"""
    # variables == dimensions
    num_variables = polytope.A_ub.shape[1]
    directions = torch.randn(num_directions, num_variables)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    lp = LinearProgram(polytope.bounds, polytope.A_ub, polytope.b_ub, objective=torch.empty(0))
    results = classerr.solving.solve_parallel(lp, directions, num_jobs)
    widths = torch.tensor([(res_max.fun - res_min.fun) for res_min, res_max in results])
    return widths if not cummulative_avg else widths.cumsum(dim=0) / torch.arange(1, num_directions+1)
