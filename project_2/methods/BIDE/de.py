"""BIDE: DE/rand/1/bin (Project 1 baseline)."""

from __future__ import annotations

import numpy as np

NPOP = 100
F = 0.5
CR = 0.9


def optimize(evaluate, lower, upper, dim, max_fes, rng, checkpoints):
    npop = NPOP
    pop = rng.uniform(lower, upper, size=(npop, dim))
    fit = np.array([evaluate(pop[i]) for i in range(npop)], dtype=float)
    fes = npop

    best_idx = int(np.argmin(fit))
    best_x = pop[best_idx].copy()
    best_f = float(fit[best_idx])

    curve = np.full(len(checkpoints), np.nan)
    cp_i = 0
    while cp_i < len(checkpoints) and fes >= checkpoints[cp_i]:
        curve[cp_i] = best_f
        cp_i += 1

    while fes < max_fes:
        new_pop, new_fit = pop.copy(), fit.copy()
        for i in range(npop):
            if fes >= max_fes:
                break
            idxs = list(range(npop))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            mutant = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), lower, upper)
            j_rand = int(rng.integers(0, dim))
            trial = pop[i].copy()
            mask = rng.random(dim) <= CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            t_fit = float(evaluate(trial))
            fes += 1
            if t_fit <= fit[i]:
                new_pop[i], new_fit[i] = trial, t_fit
                if t_fit < best_f:
                    best_f, best_x = t_fit, trial.copy()
            while cp_i < len(checkpoints) and fes >= checkpoints[cp_i]:
                curve[cp_i] = best_f
                cp_i += 1
        pop, fit = new_pop, new_fit

    while cp_i < len(checkpoints):
        curve[cp_i] = best_f
        cp_i += 1
    return best_f, best_x, fes, curve, {"population": npop}
