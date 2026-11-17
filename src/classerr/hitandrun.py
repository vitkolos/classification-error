import numba as nb
import numpy as np
import torch


def sample_points(
    A: torch.Tensor,
    b: torch.Tensor,
    init: torch.Tensor,
    num_iter: int,
    burn_in: int = 0,
    thin: int = 1,
    seed: int | None = None,
    tol: float = 1e-13,
) -> torch.Tensor:
    A_np = np.ascontiguousarray(A.detach().cpu().numpy(), dtype=np.float64)
    b_np = np.ascontiguousarray(b.detach().cpu().numpy(), dtype=np.float64)
    init_np = np.ascontiguousarray(init.detach().cpu().numpy(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    raw_samples = _hitandrun_kernel(
        A_ub=A_np,
        b_ub=b_np,
        init=init_np,
        num_iter=num_iter,
        burn_in=burn_in,
        thin=thin,
        rng=rng,
        tol=tol,
    )
    return torch.from_numpy(raw_samples).to(device=init.device, dtype=init.dtype)


@nb.njit(inline='always')
def _find_chord_bounds(
    slack: np.ndarray,
    projected_dir: np.ndarray,
    tol: float,
) -> tuple[float, float]:
    lambda_min = -np.inf  # max M-
    lambda_max = np.inf  # min M+

    for i in range(slack.shape[0]):
        rate = projected_dir[i]

        if rate > tol:
            # rate positive -> mu in M+
            mu = slack[i] / rate
            if mu < lambda_max:
                lambda_max = mu
        elif rate < -tol:
            # rate negative -> mu in M-
            mu = slack[i] / rate
            if mu > lambda_min:
                lambda_min = mu
        else:
            # rate zero -> cannot divide
            pass

    # clamp against boundary drift
    if lambda_min > lambda_max:
        # it should hold that lambda_min <= 0 <= lambda_max
        # so zero is a safe option
        return 0.0, 0.0

    return lambda_min, lambda_max


@nb.njit(cache=True)
def _hitandrun_kernel(
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    init: np.ndarray,
    num_iter: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    tol: float,
) -> np.ndarray:
    dim = A_ub.shape[1]
    x = init.copy()
    slack = b_ub - (A_ub @ x)  # distance to each boundary hyperplane (non-negative)

    num_samples = (num_iter - burn_in) // thin
    samples = np.empty((num_samples, dim), dtype=np.float64)
    sample_idx = 0

    for step in range(1, num_iter + 1):
        # random direction drawn from the unit sphere
        u = rng.standard_normal(dim)
        norm = np.linalg.norm(u)
        u /= norm

        projected_dir = A_ub @ u

        lambda_min, lambda_max = _find_chord_bounds(slack, projected_dir, tol)
        lambda_ = lambda_min + rng.random() * (lambda_max - lambda_min)

        x += lambda_ * u
        # update slack (without matrix-vector multiplication)
        slack -= lambda_ * projected_dir

        if step > burn_in and (step - burn_in) % thin == 0:
            samples[sample_idx] = x
            sample_idx += 1

    return samples
