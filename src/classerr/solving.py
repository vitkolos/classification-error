import dataclasses
from typing import NamedTuple, overload
from contextvars import ContextVar
from contextlib import contextmanager

import torch
import numpy as np
import scipy.optimize
import gurobipy
import joblib

from classerr.utils import Bounds, progress, AppError


@dataclasses.dataclass
class Polytope:
    bounds: Bounds
    A_ub: torch.Tensor
    b_ub: torch.Tensor

    def get_full_constraints(self) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(len(self.bounds))
        A_matrices = [self.A_ub]
        b_vectors = [self.b_ub]

        lb = torch.as_tensor(self.bounds.lb)
        valid_lb = torch.isfinite(lb)
        if torch.any(valid_lb):
            A_matrices.append(-eye[valid_lb])
            b_vectors.append(-lb[valid_lb])

        ub = torch.as_tensor(self.bounds.ub)
        valid_ub = torch.isfinite(ub)
        if torch.any(valid_ub):
            A_matrices.append(eye[valid_ub])
            b_vectors.append(ub[valid_ub])

        A_full = torch.cat(A_matrices)
        b_full = torch.cat(b_vectors)
        return A_full, b_full


@dataclasses.dataclass
class LinearProgram(Polytope):
    objective: torch.Tensor
    bias: torch.Tensor = dataclasses.field(default_factory=lambda: torch.tensor(0.0))
    maximize: bool = True
    neuron_states: list[torch.Tensor] | None = None  # used to characterize the polytope (not used by the solvers)

    def to_polytope_hashable(self) -> 'PolytopeHashable':
        if self.neuron_states is None:
            raise AppError('neuron_states not available')

        return PolytopeHashable(self.neuron_states)


class PolytopeHashable:
    def __init__(self, neuron_states: list[torch.Tensor]):
        self.neuron_states = neuron_states
        # hash_tensor reduces the whole tensor using XOR, there may be frequent collisions
        hashes = [layer.hash_tensor().item() for layer in neuron_states]
        self.hash = hash(tuple(hashes))

    def __eq__(self, other):
        return type(other) is PolytopeHashable and all(torch.equal(a, b) for a, b in zip(self.neuron_states, other.neuron_states))

    def __hash__(self):
        return self.hash


class OptimizationResult(NamedTuple):
    x: torch.Tensor
    fun: float


class SolverError(AppError):
    pass


class Infeasible(SolverError):
    pass


MultipleResults = list[tuple[OptimizationResult, OptimizationResult]]

SOLVER_DEFAULT = 'highs'
_active_solver = ContextVar('active_solver', default=SOLVER_DEFAULT)


@contextmanager
def solver_config(solver_name: str):
    solver_name = solver_name.lower()  # conver to lowercase
    token = _active_solver.set(solver_name if solver_name else SOLVER_DEFAULT)
    try:
        yield
    finally:
        _active_solver.reset(token)


def init_worker(solver_name: str):
    _active_solver.set(solver_name)


def get_min_max_lps(lp: LinearProgram, objective: torch.Tensor) -> tuple[LinearProgram, LinearProgram]:
    lp_min = dataclasses.replace(lp, objective=objective, maximize=False)
    lp_max = dataclasses.replace(lp, objective=objective, maximize=True)
    return lp_min, lp_max


@overload
def solve(lp: LinearProgram, solver: str = '', verbose: bool = False) -> OptimizationResult: ...


@overload
def solve(lp: LinearProgram, solver: str = '', verbose: bool = False,
          *, multiple_objectives: torch.Tensor) -> MultipleResults: ...


def solve(lp: LinearProgram, solver: str = '', verbose: bool = False, multiple_objectives: torch.Tensor | None = None) -> OptimizationResult | MultipleResults:
    """should raise Infeasible for infeasible instances"""
    solver = solver if solver else _active_solver.get()  # in lowercase

    if solver == 'gurobi' or solver == 'gurobi-barrier':
        return solve_gurobi(lp, verbose, force_barrier=(solver == 'gurobi-barrier'), multiple_objectives=multiple_objectives)

    def solve_single(lp: LinearProgram) -> OptimizationResult:
        match solver:
            case 'highs': return solve_scipy(lp, verbose)
            case _: raise NotImplementedError

    if multiple_objectives is not None:
        lps = (get_min_max_lps(lp, objective) for objective in progress(multiple_objectives))
        return [(solve_single(lp_min), solve_single(lp_max)) for lp_min, lp_max in lps]

    return solve_single(lp)


def solve_parallel(lp: LinearProgram, multiple_objectives: torch.Tensor, num_jobs: int) -> MultipleResults:
    if num_jobs <= 1:
        return solve(lp, multiple_objectives=multiple_objectives)

    solver = _active_solver.get()
    chunks = torch.chunk(multiple_objectives, num_jobs)
    results = []

    try:
        wrapped_solve = joblib.delayed(solve)
        with joblib.Parallel(return_as='generator_unordered', require='sharedmem') as para:
            results_gen = para(wrapped_solve(lp, solver, multiple_objectives=chunk) for chunk in chunks)
            for r in progress(results_gen, total=len(chunks), smoothing=0, main=True):
                results.extend(r)
    finally:
        # shutdown dangling loky workers
        joblib.externals.loky.get_reusable_executor().shutdown(wait=True, kill_workers=True)

    return results


def solve_scipy(lp: LinearProgram, verbose: bool) -> OptimizationResult:
    # we want to maximize the objective -> we minimize cx (for Ax <= b)
    sense = -1 if lp.maximize else 1
    c = sense * lp.objective
    result = scipy.optimize.linprog(c.numpy(), lp.A_ub.numpy(), lp.b_ub.numpy(),
                                    bounds=lp.bounds.seq, options={'disp': verbose})

    if not result.success:
        match result.status:
            case 2:
                raise Infeasible
            case 3:
                raise SolverError('optimization failed, problem unbounded (check if the bounds are set correctly)')
            case _:
                raise SolverError('optimization failed')
    elif result.x is None or result.fun is None:
        raise AppError('result is empty')

    found_x = torch.from_numpy(result.x).to(dtype=torch.get_default_dtype())
    found_maximum = sense * result.fun + lp.bias.item()
    return OptimizationResult(found_x, found_maximum)


def solve_gurobi(lp: LinearProgram, verbose: bool, force_barrier: bool = False, multiple_objectives: torch.Tensor | None = None) -> OptimizationResult | MultipleResults:
    num_variables = lp.A_ub.shape[1]

    with gurobipy.Env(empty=True) as env:
        env.setParam('LogToConsole', int(verbose))
        env.setParam('Threads', 1)

        if force_barrier:
            env.setParam('Method', 2)  # barrier solver

        env.start()

        with gurobipy.Model(env=env) as model:
            x = model.addMVar(shape=num_variables, lb=lp.bounds.lb, ub=lp.bounds.ub)
            model.addConstr(lp.A_ub.numpy() @ x <= lp.b_ub.numpy())

            def optimize(objective: torch.Tensor, maximize: bool) -> OptimizationResult:
                sense = gurobipy.GRB.MAXIMIZE if maximize else gurobipy.GRB.MINIMIZE
                model.setObjective(objective.numpy() @ x, sense)
                model.optimize()

                match model.Status:
                    case gurobipy.GRB.OPTIMAL:
                        found_x = torch.from_numpy(x.X).to(dtype=torch.get_default_dtype())
                        found_maximum = model.ObjVal + lp.bias.item()
                        return OptimizationResult(found_x, found_maximum)
                    case gurobipy.GRB.INFEASIBLE:
                        raise Infeasible

                raise SolverError(f'optimization ended with status {model.Status}')

            if multiple_objectives is not None:
                return [(optimize(o, maximize=False), optimize(o, maximize=True)) for o in progress(multiple_objectives)]
            else:
                return optimize(lp.objective, lp.maximize)
