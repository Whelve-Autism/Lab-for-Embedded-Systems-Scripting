"""NICE w/o Local Search: disable CMA; return CMA FE budget to DE."""

from methods.NICE_DE.nice_de import optimize as _optimize


def optimize(evaluate, lower, upper, dim, max_fes, rng, checkpoints):
    return _optimize(
        evaluate, lower, upper, dim, max_fes, rng, checkpoints,
        ablation="no_local_search",
    )
