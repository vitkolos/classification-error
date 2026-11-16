import torch


@torch.inference_mode()
def generate(
    A_full: torch.Tensor,
    b_full: torch.Tensor,
    init: torch.Tensor,
    num_samples: int | None = None,
    num_iter: int | None = None,
    burn_in: int = 0,
    thin: int = 1,
    seed: int | None = None,
    epsilon: float = 1e-12
):
    """
    a) takes shape [dim,] and returns shape [num_samples, dim], or
    b) takes shape [chains, dim] and returns shape [num_samples, chains, dim]
    """
    single_point = init.ndim == 1
    x = init.unsqueeze(0) if single_point else init
    A_T = A_full.T

    if num_iter is None and num_samples is not None:
        num_iter = burn_in + num_samples * thin
    elif num_samples is None and num_iter is not None:
        num_samples = (num_iter - burn_in) // thin
    else:
        raise ValueError

    samples = torch.empty((num_samples, *x.shape), dtype=x.dtype, device=x.device)
    gen = torch.Generator(device=x.device)

    if seed is not None:
        gen.manual_seed(seed)

    idx = 0

    for i in range(1, num_iter + 1):
        x = single_step(x, A_T, b_full, gen, epsilon)
        if i > burn_in and (i - burn_in) % thin == 0:
            samples[idx] = x
            idx += 1

    return samples.squeeze(1) if single_point else samples


def single_step(
    x: torch.Tensor, A_T: torch.Tensor, b: torch.Tensor, gen: torch.Generator, epsilon: float
) -> torch.Tensor:
    # x: [chains, dim], A_T: [dim, constr], b: [constr,]

    # generate random directions on a unit sphere
    u = torch.randn_like(x, generator=gen)
    u /= torch.linalg.vector_norm(u, dim=1, keepdim=True)

    # find ray-hyperplane intersections
    numerator = b - (x @ A_T)
    denominator = u @ A_T

    # prevent division by zero
    den_pos = denominator > epsilon
    den_neg = denominator < -epsilon
    safe_den = torch.where(den_pos | den_neg, denominator, torch.ones_like(denominator))
    Ms = numerator / safe_den

    lambda_min = torch.where(den_neg, Ms, float('-inf')).amax(dim=-1, keepdim=True)
    lambda_max = torch.where(den_pos, Ms, float('inf')).amin(dim=-1, keepdim=True)
    # clamp against minor boundary drift (ensure that lambda_min <= lambda_max)
    lambda_min = torch.minimum(lambda_min, lambda_max)

    # uniformly sample from the interval
    alpha = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device, generator=gen)
    lambda_ = lambda_min + alpha * (lambda_max - lambda_min)

    return x + lambda_ * u
