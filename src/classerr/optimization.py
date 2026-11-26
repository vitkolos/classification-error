import dataclasses
from typing import Any
import torch

import classerr.neurons
import classerr.hitandrun
import classerr.solving
from classerr.solving import Polytope, LinearProgram, OptimizationResult, PolytopeHashable
import classerr.utils
from classerr.utils import Bounds, Metadata

NUM_DIRECTIONS = 100
MCMC_NUM_POINTS = 150


@dataclasses.dataclass
class PolytopeResult:
    point_max_err: torch.Tensor | None = None
    pred_orig: torch.Tensor | None = None
    pred_aprx: torch.Tensor | None = None
    width: float | None = None
    width_pt: torch.Tensor | None = None

    # 1D or 2D lists
    widths_cls: list | None = None
    radii_cls: list | None = None

    # applicable for extended polytopes
    num_polytopes: int | None = None
    exploration: list[tuple[int, float]] | None = None

    def predict(self, model_orig: torch.nn.Module, model_aprx: torch.nn.Module):
        assert self.point_max_err is not None

        with torch.no_grad():
            input_b = self.point_max_err.unsqueeze(0)
            self.pred_orig = model_orig(input_b).squeeze(0)
            self.pred_aprx = model_aprx(input_b).squeeze(0)


class Subpolytopes:
    def __init__(self) -> None:
        self.polytopes: dict[Any, OptimizationResult] = {}
        self.exploration: list[tuple[int, OptimizationResult]] = []
        self.widths_cls: list = []
        self.radii_cls: list = []


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


def analyze_point(model_orig: torch.nn.Module, model_aprx: torch.nn.Module, point: torch.Tensor) -> PolytopeResult:
    result = PolytopeResult(point)
    result.predict(model_orig, model_aprx)
    return result


def analyze_polytope_sub(
    metadata: Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    lp_orig: LinearProgram,
    lp_aprx: LinearProgram,
    point: torch.Tensor,
    target: int,
    spt: Subpolytopes,
    generalized_acc_widths: bool,
) -> PolytopeResult:
    result = PolytopeResult()
    A_sub, b_sub = correct_polytope_constraints(target, lp_orig, lp_aprx)
    polytope_sub = Polytope(metadata.bounds, A_sub, b_sub)

    result.width_pt = polytope_widths(polytope_sub)
    result.width = result.width_pt.mean().item()

    # approach 1: linear multiclass margin loss
    max_err = find_error_maximum(metadata, A_sub, b_sub, lp_aprx, target)
    result.point_max_err = max_err.x.reshape_as(point)
    result.predict(model_orig, model_aprx)

    # approach 3: generalized accuracy
    if generalized_acc_widths:
        result.widths_cls, result.radii_cls = compute_classes_mean_widths(metadata, A_sub, b_sub, lp_aprx)

    # initialize the exploration of the extended polytope
    lp_sub_hashable = (lp_orig.to_polytope_hashable(), lp_aprx.to_polytope_hashable())
    spt.polytopes[lp_sub_hashable] = max_err
    spt.exploration.append((1, max_err))
    spt.widths_cls.append(result.widths_cls)
    spt.radii_cls.append(result.radii_cls)

    return result


def analyze_polytope_ext(
    metadata: Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    lp_orig: LinearProgram,
    point: torch.Tensor,
    target: int,
    spt: Subpolytopes,
    generalized_acc_widths: bool,
) -> PolytopeResult:
    result = PolytopeResult()
    A_ext, b_ext = correct_polytope_constraints(target, lp_orig, lp_aprx=None)
    polytope_ext = Polytope(metadata.bounds, A_ext, b_ext)

    result.width_pt = polytope_widths(polytope_ext)
    result.width = result.width_pt.mean().item()

    sampled_points = sample_points_in_polytope(polytope_ext, point)
    subpolytopes_extend(spt, metadata, model_orig, model_aprx, sampled_points, target, generalized_acc_widths)
    max_err = max(spt.polytopes.values(), key=lambda result: result.fun)
    result.point_max_err = max_err.x.reshape_as(point)
    result.predict(model_orig, model_aprx)

    result.widths_cls = spt.widths_cls
    result.radii_cls = spt.radii_cls
    result.num_polytopes = len(spt.polytopes)
    result.exploration = [(n, res.fun) for n, res in spt.exploration]
    return result


def find_error_maximum(
    metadata: Metadata, A_cor: torch.Tensor, b_cor: torch.Tensor, lp_aprx: LinearProgram, target: int
) -> OptimizationResult:
    results = []

    for k in range(metadata.num_classes):
        if k != target:
            objective, bias = objective_class_error(k, target, lp_aprx.objective, lp_aprx.bias)
            lp = LinearProgram(metadata.bounds, A_cor, b_cor, objective, bias, maximize=True)
            x, fun = classerr.solving.solve(lp)
            results.append((fun, k, x))

    fun_max, k_max, x_max = max(results)
    return OptimizationResult(x_max, fun_max)


def compute_classes_mean_widths(
    metadata: Metadata, A_cor: torch.Tensor, b_cor: torch.Tensor, lp_aprx: LinearProgram
) -> tuple[list[float], list[float]]:
    mean_widths = []
    radii = []

    for k in range(metadata.num_classes):
        A_cl, b_cl = constrain_class(k, lp_aprx.objective, lp_aprx.bias)
        polytope = Polytope(metadata.bounds, torch.cat([A_cor, A_cl]), torch.cat([b_cor, b_cl]))
        mean_width = 0.0
        radius = chebyshev_radius(polytope)

        # this is just a coarse filter to get rid of *some* flat polytopes
        if radius > 0:
            mean_width = polytope_widths(polytope).mean().item()

        mean_widths.append(mean_width)
        radii.append(radius)

    return mean_widths, radii


def correct_polytope_constraints(
    target: int, lp_orig: LinearProgram, lp_aprx: LinearProgram | None
) -> tuple[torch.Tensor, torch.Tensor]:
    A_cl, b_cl = constrain_class(target, lp_orig.objective, lp_orig.bias)
    A_ub = [A_cl, lp_orig.A_ub]
    b_ub = [b_cl, lp_orig.b_ub]

    if lp_aprx is not None:
        A_ub.append(lp_aprx.A_ub)
        b_ub.append(lp_aprx.b_ub)

    return torch.cat(A_ub), torch.cat(b_ub)


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
    polytope: Polytope,
    num_directions: int = NUM_DIRECTIONS,
    seed: int | torch.Generator | None = None,
    cumulative_avg: bool = False,
    num_jobs: int = 1,
) -> torch.Tensor:
    """returns widths of the polytope computed from many random directions (or the cumulative average in each step)"""
    generator = torch.Generator().manual_seed(seed) if isinstance(seed, int) else seed
    # variables == dimensions
    num_variables = polytope.A_ub.shape[1]
    directions = torch.randn(num_directions, num_variables, generator=generator)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    lp = LinearProgram(polytope.bounds, polytope.A_ub, polytope.b_ub, objective=torch.empty(0))
    results = classerr.solving.solve_parallel(lp, directions, num_jobs)
    widths = torch.tensor([(res_max.fun - res_min.fun) for res_min, res_max in results])
    return widths if not cumulative_avg else widths.cumsum(dim=0) / torch.arange(1, num_directions+1)


def subpolytopes_extend(
    spt: Subpolytopes,
    metadata: Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    points: torch.Tensor,
    target: int,
    compute_widths: bool,
    max_polytopes: int | None = None,
):
    for point in classerr.utils.progress(points):
        # we construct the polytope and check if it has already been analyzed
        lp_orig = lp_from_net(model_orig, metadata.bounds, point)
        lp_aprx = lp_from_net(model_aprx, metadata.bounds, point)
        A_sub, b_sub = correct_polytope_constraints(target, lp_orig, lp_aprx)
        h = (lp_orig.to_polytope_hashable(), lp_aprx.to_polytope_hashable())

        if h not in spt.polytopes:
            spt.polytopes[h] = find_error_maximum(metadata, A_sub, b_sub, lp_aprx, target)

            if compute_widths:
                widths_cls, radii_cls = compute_classes_mean_widths(metadata, A_sub, b_sub, lp_aprx)
                spt.widths_cls.append(widths_cls)
                spt.radii_cls.append(radii_cls)

        spt.exploration.append((len(spt.polytopes), spt.polytopes[h]))

        if max_polytopes is not None and len(spt.polytopes) >= max_polytopes:
            break


def sample_points_in_polytope(
    polytope: Polytope, point_initial: torch.Tensor, num_points: int = MCMC_NUM_POINTS
) -> torch.Tensor:
    A_full, b_full = polytope.get_full_constraints()
    # we stick to the "purest" form of MCMC Hit-and-Run
    init = point_initial
    thin = 1
    points = classerr.hitandrun.sample_points(A_full, b_full, init, num_iter=num_points * thin, thin=thin)
    return points.reshape(-1, *point_initial.shape)


def chebyshev_radius(polytope: Polytope) -> float:
    # this check is supposed to save time (it's faster to check feasibility than to find the Chebyshev center)
    if polytope_exists(polytope):
        A_full, b_full = polytope.get_full_constraints()
        _, radius = chebyshev_center(A_full, b_full)
        return radius
    else:
        return 0.0


def polytope_exists(polytope: Polytope) -> bool:
    try:
        lp = LinearProgram(polytope.bounds, polytope.A_ub, polytope.b_ub, objective=torch.zeros(len(polytope.bounds)))
        _ = classerr.solving.solve(lp)
        return True
    except classerr.solving.Infeasible:
        return False


def chebyshev_center(A_full: torch.Tensor, b_full: torch.Tensor) -> tuple[torch.Tensor, float]:
    """find the Chebyshev center: https://en.wikipedia.org/w/index.php?title=Chebyshev_center&oldid=1340455583#Linear_programming_problem"""
    num_vars = A_full.shape[1]
    bounds = Bounds([(None, None)] * num_vars + [(0.0, None)])  # we constrain r >= 0
    row_norms = torch.linalg.vector_norm(A_full, dim=1, keepdim=True)
    A_ex = torch.hstack([A_full, row_norms])
    objective = torch.tensor([0.0] * num_vars + [1.0])
    lp = LinearProgram(bounds, A_ex, b_full, objective)
    result = classerr.solving.solve(lp)
    center = result.x[:-1]
    radius = result.x[-1].item()
    return center, radius


def assert_inside_polytope(point: torch.Tensor, polytope: Polytope, abs_tol: float = 1e-6):
    problems = []
    point_flat = point.flatten()

    if len(polytope.bounds.seq) != point.numel():
        raise ValueError(
            f'{len(polytope.bounds.seq)} bounds were provided, but there are {point.numel()} input neurons')

    too_low = torch.nonzero(point_flat < torch.from_numpy(polytope.bounds.lb)).flatten().tolist()
    too_high = torch.nonzero(point_flat > torch.from_numpy(polytope.bounds.ub)).flatten().tolist()

    if too_low:
        problems.append(f'indices {too_low} < lower bounds')

    if too_high:
        problems.append(f'indices {too_high} > upper bounds')

    left_side = polytope.A_ub @ point_flat
    infeasible_rows = torch.nonzero(left_side > polytope.b_ub + abs_tol).flatten()

    for i in infeasible_rows:
        problems.append(f'violated constraint {i}: {left_side[i].item():.6f} <= {polytope.b_ub[i].item():.6f}')

    if problems:
        raise AssertionError(f"point is outside the polytope, reasons:\n{'\n'.join(problems)}\ninput tensor:\n{point}")
