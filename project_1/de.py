"""Basic Differential Evolution: DE/rand/1/bin."""

from __future__ import annotations

import numpy as np


def de_rand1_bin(
    evaluate,
    lower: float,
    upper: float,
    dim: int,
    npop: int,
    F: float,
    CR: float,
    max_fes: int,
    rng: np.random.Generator,
    checkpoints: np.ndarray,
):
    """
    Run DE/rand/1/bin until exactly max_fes function evaluations.

    Returns
    -------
    best_fitness : float
    best_x : ndarray
    actual_fes : int
    curve_errors : ndarray
        Best-so-far fitness at each checkpoint (raw f, not error).
        Caller subtracts optimum if needed.
    """
    pop = rng.uniform(lower, upper, size=(npop, dim))
    fit = np.empty(npop, dtype=float)
    for i in range(npop):
        fit[i] = evaluate(pop[i])
    fes = npop

    best_idx = int(np.argmin(fit))
    best_x = pop[best_idx].copy()
    best_f = float(fit[best_idx])

    curve = np.full(len(checkpoints), np.nan, dtype=float)
    cp_i = 0
    while cp_i < len(checkpoints) and fes >= checkpoints[cp_i]:
        curve[cp_i] = best_f
        cp_i += 1

    while fes < max_fes:
        new_pop = pop.copy()
        new_fit = fit.copy()
        for i in range(npop):
            if fes >= max_fes:
                break
            # distinct random indices r1, r2, r3 != i
            idxs = list(range(npop))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lower, upper)

            j_rand = int(rng.integers(0, dim))
            trial = pop[i].copy()
            mask = rng.random(dim) <= CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]

            t_fit = float(evaluate(trial))
            fes += 1
            if t_fit <= fit[i]:
                new_pop[i] = trial
                new_fit[i] = t_fit
                if t_fit < best_f:
                    best_f = t_fit
                    best_x = trial.copy()

            while cp_i < len(checkpoints) and fes >= checkpoints[cp_i]:
                curve[cp_i] = best_f
                cp_i += 1

        pop, fit = new_pop, new_fit

    # fill any remaining checkpoints with final best
    while cp_i < len(checkpoints):
        curve[cp_i] = best_f
        cp_i += 1

    return best_f, best_x, fes, curve
