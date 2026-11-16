import torch
import scipy.stats
import classerr.hitandrun


def test_hitandrun_hypercube():
    dim = 5

    # hypercube [-1, 1]^dim
    #  Ix <= 1
    # -Ix <= 1
    A = torch.vstack([torch.eye(dim), -torch.eye(dim)])
    b = torch.ones(2 * dim)
    init = torch.zeros(dim)

    chain = classerr.hitandrun.generate(A, b, init, num_samples=20000, thin=5, burn_in=2000, seed=42)

    assert torch.all(A @ chain.T <= b[:, None] + 1e-11), "samples escaped polytope"

    means = torch.mean(chain, axis=0)
    assert torch.all(torch.abs(means) < 0.05), f"non-zero mean: {means}"

    # true variance = 1/3
    vars_ = torch.var(chain, axis=0)
    assert torch.all(torch.abs(vars_ - 1/3) < 0.05), f"variance deviating: {vars_}"

    # KS test against Uniform(-1, 1)
    for dim in range(dim):
        res = scipy.stats.kstest(chain[:, dim], "uniform", args=(-1, 2))
        assert res.pvalue > 0.01, f"dimension {dim} failed uniform KS test (p-value {res.pvalue:.4e})"
