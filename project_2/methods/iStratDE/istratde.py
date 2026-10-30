"""
iStratDE — NumPy port of EMI-Group/istratde (PyTorch + EvoX operators).

Faithful to:
  src/istratde/algorithms/pytorch/istratde.py
  evox.operators.crossover.differential_evolution
  evox.operators.selection.find_pbest.select_rand_pbest

Paper: Feng et al., IEEE TEVC 2026 / arXiv:2602.01147.
Each individual keeps a fixed (F, CR, base_prim, base_sec, n_diff, crossover)
drawn at initialization; no population-level adaptation.
"""

from __future__ import annotations

import numpy as np


def _differential_sum(rng, pop, indices, num_diff, padding=9):
    """Vectorized port of EvoX DE_differential_sum."""
    npop, dim = pop.shape
    select_len = num_diff.astype(int) * 2 + 1  # (NP,)
    rand_indices = rng.integers(0, npop, size=(npop, padding))
    rand_indices = np.where(
        rand_indices == indices[:, None], npop - 1, rand_indices
    )
    pop_permute = pop[rand_indices]  # (NP, padding, dim)
    mask = np.arange(padding)[None, :] < select_len[:, None]
    pop_permute = np.where(mask[:, :, None], pop_permute, 0.0)
    diffs = pop_permute[:, 1:, :]
    pos = diffs[:, 0::2, :].sum(axis=1)
    neg = diffs[:, 1::2, :].sum(axis=1)
    return pos - neg, rand_indices[:, 0]


def _bin_cross(rng, mutant, current, cr):
    """EvoX DE_binary_crossover."""
    npop, dim = mutant.shape
    mask = rng.random((npop, dim)) < cr[:, None]
    rind = rng.integers(0, dim, size=npop)
    mask[np.arange(npop), rind] = True
    return np.where(mask, mutant, current)


def _exp_cross(rng, mutant, current, cr):
    """EvoX 1.1.1 DE_exponential_crossover (NumPy port).

    Official: ll = floor(log(U) / (-log1p(CR))), mask via tiled gather.
    Denominator is -log(1+CR) (<0 for CR>0), NOT -log(1-CR).
    """
    npop, dim = mutant.shape
    nn = rng.integers(0, dim, size=npop)
    float_tiny = 1.1754943508222875e-38
    ll = np.maximum(rng.random(npop), float_tiny)
    cr = np.asarray(cr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # EvoX: ll.log() / (-CR.log1p())
        denom = -np.log1p(cr)
        L = np.floor(np.log(ll) / denom)
    L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
    length = np.minimum(L, dim) - 1  # may be -1 / 0 → empty mask
    prefix = np.arange(dim)[None, :] < length[:, None]
    # EvoX gather on tile(mask, (1,2)) at nn+arange(dim) ≡ roll by -nn
    trial = current.copy()
    for i in range(npop):
        mask = np.roll(prefix[i], -int(nn[i]))
        trial[i] = np.where(mask, mutant[i], current[i])
    return trial


def _arith_cross(mutant, current, k):
    """EvoX DE_arithmetic_recombination."""
    return current + k[:, None] * (mutant - current)


def _select_rand_pbest(rng, percent, pop, fit):
    """EvoX select_rand_pbest."""
    npop = len(fit)
    top_p = max(int(npop * percent), 1)
    pool = np.argsort(fit)[:top_p]
    pick = rng.integers(0, top_p, size=npop)
    return pop[pool[pick]]


def optimize(evaluate, lower, upper, dim, max_fes, rng, checkpoints):
    # Paper stresses large NP for individual-level diversity. Under course
    # MaxFEs=10000D we use NP=max(200, 50*D) (D=10 -> 500) so many fixed
    # strategies coexist while still allowing ≥200 generations.
    NP = max(200, 50 * dim)
    diff_padding = 9

    pop = rng.uniform(lower, upper, size=(NP, dim))
    fit = np.empty(NP, dtype=float)
    fes = 0
    best_x = pop[0].copy()
    best_f = np.inf

    curve = np.full(len(checkpoints), np.nan)
    cp_i = 0

    def record():
        nonlocal cp_i
        while cp_i < len(checkpoints) and fes >= checkpoints[cp_i]:
            curve[cp_i] = best_f
            cp_i += 1

    # Evaluate init one-by-one so early checkpoints never see future info
    for i in range(NP):
        if fes >= max_fes:
            fit[i:] = np.inf
            break
        fit[i] = float(evaluate(pop[i]))
        fes += 1
        if fit[i] < best_f:
            best_f = float(fit[i])
            best_x = pop[i].copy()
        record()

    grid = np.linspace(0.0, 1.0, 1001)
    Fs = grid[rng.integers(0, 1001, size=NP)]
    CRs = grid[rng.integers(0, 1001, size=NP)]
    base_prim = rng.integers(0, 4, size=NP)  # rand/best/pbest/current
    base_sec = rng.integers(0, 4, size=NP)
    num_diff = rng.integers(0, 4, size=NP)
    cross_id = rng.integers(0, 3, size=NP)  # bin/exp/arith

    best_idx = int(np.argmin(fit))
    if fit[best_idx] < best_f:
        best_f = float(fit[best_idx])
        best_x = pop[best_idx].copy()

    while fes < max_fes:
        indices = np.arange(NP)
        diff_sum, rand_idx = _differential_sum(
            rng, pop, indices, num_diff, diff_padding
        )
        rand_vecs = pop[rand_idx]
        best_vecs = np.tile(pop[best_idx][None, :], (NP, 1))
        pbest_vecs = _select_rand_pbest(rng, 0.05, pop, fit)
        current_vecs = pop

        merge = np.stack([rand_vecs, best_vecs, pbest_vecs, current_vecs], axis=0)
        prim = np.zeros((NP, dim))
        sec = np.zeros((NP, dim))
        for t in range(4):
            m = base_prim == t
            prim[m] = merge[t, m]
            m = base_sec == t
            sec[m] = merge[t, m]

        base = prim + Fs[:, None] * (sec - prim)
        mutants = base + diff_sum * Fs[:, None]

        # Match official: compute all three crossover forms, then select by strategy id
        bin_t = _bin_cross(rng, mutants, current_vecs, CRs)
        exp_t = _exp_cross(rng, mutants, current_vecs, CRs)
        ari_t = _arith_cross(mutants, current_vecs, CRs)
        trials = np.zeros((NP, dim))
        trials = np.where((cross_id == 0)[:, None], bin_t, trials)
        trials = np.where((cross_id == 1)[:, None], exp_t, trials)
        trials = np.where((cross_id == 2)[:, None], ari_t, trials)
        trials = np.clip(trials, lower, upper)

        n_eval = min(NP, max_fes - fes)
        for i in range(n_eval):
            t_fit = float(evaluate(trials[i]))
            fes += 1
            if t_fit <= fit[i]:
                pop[i] = trials[i]
                fit[i] = t_fit
                if t_fit < best_f:
                    best_f = t_fit
                    best_x = trials[i].copy()
            record()
            if fes >= max_fes:
                break

        best_idx = int(np.argmin(fit))
        if fit[best_idx] < best_f:
            best_f = float(fit[best_idx])
            best_x = pop[best_idx].copy()

    while cp_i < len(checkpoints):
        curve[cp_i] = best_f
        cp_i += 1
    return best_f, best_x, fes, curve, {"population": NP}
