"""
DLS-DE — main-text reconstruction (Chen & Meng, Swarm Evol. Comput. 100, 2026).

Aligned to Algorithms 4–6 and Eqs. (8)–(17). No official code released.
"""

from __future__ import annotations

import math

import numpy as np


def _init_ps(dim: int) -> int:
    # Table 2 / DE-EXP: PS = 25 * ln(D) * sqrt(D)
    return max(4, int(round(25.0 * math.log(dim) * math.sqrt(dim))))


def _cauchy(rng, mu, sigma):
    return mu + sigma * math.tan(math.pi * (rng.random() - 0.5))


def _exp_crossover(target, mutant, cr, rng):
    d = len(target)
    trial = target.copy()
    mask = np.zeros(d, dtype=int)
    j = int(rng.integers(0, d))
    L = 0
    while True:
        trial[j] = mutant[j]
        mask[j] = 1
        L += 1
        j = (j + 1) % d
        if L >= d or rng.random() >= cr:
            break
    return trial, mask


def _bound_mid(x, parent, lower, upper):
    x = x.copy()
    lo = np.full_like(x, lower, dtype=float) if np.isscalar(lower) else lower
    hi = np.full_like(x, upper, dtype=float) if np.isscalar(upper) else upper
    low = x < lo
    high = x > hi
    x[low] = (parent[low] + lo[low]) * 0.5
    x[high] = (parent[high] + hi[high]) * 0.5
    return x


def optimize(evaluate, lower, upper, dim, max_fes, rng, checkpoints):
    ps_init = _init_ps(dim)
    ps_min = 4
    ps = ps_init
    H = 6
    p = 0.11
    r_arc = 1.6
    ST = 30
    tau = 0.3
    lam_sigma = 0.1
    phase_thr = 0.3 * max_fes

    muF_mem = np.full(H, 0.5)
    sigmaF = 0.05
    Prob = np.full((H, dim), 1.0 / dim)
    mem_pos = 0
    SF_prev_mean = 0.0

    pop = rng.uniform(lower, upper, size=(ps, dim))
    fit = np.empty(ps, dtype=float)
    fes = 0
    best_x = pop[0].copy()
    best_f = np.inf
    last_SR = 0.5

    curve = np.full(len(checkpoints), np.nan)
    cp_i = 0

    def record():
        nonlocal cp_i
        while cp_i < len(checkpoints) and fes >= checkpoints[cp_i]:
            curve[cp_i] = best_f
            cp_i += 1

    for i in range(ps):
        if fes >= max_fes:
            fit[i:] = np.inf
            break
        fit[i] = float(evaluate(pop[i]))
        fes += 1
        if fit[i] < best_f:
            best_f = float(fit[i])
            best_x = pop[i].copy()
        record()

    Y = pop.copy()
    fit_Y = fit.copy()  # Alg.4 needs f(Y)
    LV = np.zeros((ps, dim))
    LF = np.zeros((ps, dim))
    ct = np.zeros(ps, dtype=int)
    archive = np.empty((0, dim))

    while fes < max_fes:
        SF = []
        success_k = []
        success_vix = []
        success_omega = []  # VIX weights for Prob update
        n_success = 0

        order = np.argsort(fit)
        pbest_pool = max(2, int(math.ceil(p * ps)))

        F_list = np.zeros(ps)
        CR_list = np.zeros(ps)
        mu_mean = float(np.mean(muF_mem))
        mu_min = float(np.min(muF_mem))
        mem_choice = np.zeros(ps, dtype=int)

        for i in range(ps):
            ri = int(rng.integers(0, H))
            mem_choice[i] = ri
            # Eq. (13) overrides when stagnation count is multiple of 10
            if ct[i] != 0 and (ct[i] % 10) == 0:
                Fi = _cauchy(rng, muF_mem[ri], 0.05)
                while Fi <= 0.0:
                    Fi = _cauchy(rng, muF_mem[ri], 0.05)
                Fi = min(Fi, 1.0)
            elif fes < phase_thr:
                # Eq. (8)+(9): designed around ~[0.35, 0.75] then soft-clip
                SR = float(np.clip(last_SR, 0.0, 1.0))
                eps = math.sin(2 * math.pi * rng.random()) * (mu_mean - mu_min)
                Fi = 0.35 + 0.4 * (SR / (0.1 + SR)) + eps
                Fi = float(np.clip(Fi, 0.05, 1.0))
            else:
                Fi = _cauchy(rng, muF_mem[ri], sigmaF)
                while Fi <= 0.0:
                    Fi = _cauchy(rng, muF_mem[ri], sigmaF)
                Fi = min(Fi, 1.0)
            F_list[i] = Fi

            # Eq. (14): k = Pi · [1..D]^T (expectation), not a random draw
            Pi = Prob[ri]
            Pi = Pi / (Pi.sum() + 1e-12)
            k_mean = float(Pi @ np.arange(1, dim + 1, dtype=float))
            Em = 10.0 * k_mean
            CR_list[i] = Em / (Em + 1.0)

        U = np.zeros_like(pop)
        masks = np.zeros((ps, dim), dtype=int)
        deltas = np.zeros((ps, dim))

        for i in range(ps):
            xp = pop[order[int(rng.integers(0, pbest_pool))]].copy()
            # Index-identity exclusion (not numerical allclose)
            cand_r1 = [j for j in range(ps) if j != i]
            r1 = int(rng.choice(cand_r1))
            n_arc = len(archive)
            union_n = ps + n_arc
            # r2 from P∪A, index ≠ i and ≠ r1 (archive indices offset by ps)
            cand_r2 = [j for j in range(union_n) if j != i and j != r1]
            r2 = int(rng.choice(cand_r2)) if cand_r2 else r1
            x_r2 = pop[r2] if r2 < ps else archive[r2 - ps]
            Fi = F_list[i]
            # Eq. (3): Δ unscaled; v = x + F*Δ. Learning stores Δ, not (v-x).
            direction = (xp - pop[i]) + (pop[r1] - x_r2)
            mutant = _bound_mid(
                pop[i] + Fi * direction, pop[i], lower, upper
            )
            trial, mask = _exp_crossover(pop[i], mutant, CR_list[i], rng)
            U[i], masks[i], deltas[i] = trial, mask, direction

        # Algorithm 5: if tau < rand => enter (prob 1-tau); matches paper pseudocode
        # Ambiguity note: body text vs pseudocode on trigger; we follow Alg.5.
        if tau < rng.random():
            for i in range(ps):
                if ct[i] > ST and np.count_nonzero(LF[i] == 0) == 0:
                    xp = pop[order[int(rng.integers(0, pbest_pool))]].copy()
                    Fi = F_list[i]
                    lv = LV[i] / np.maximum(LF[i], 1.0)
                    # Eq. (17)
                    direction = (xp - pop[i]) + lv
                    U[i] = _bound_mid(
                        pop[i] + Fi * direction, pop[i], lower, upper
                    )
                    masks[i] = np.ones(dim, dtype=int)
                    deltas[i] = direction

        n_eval = min(ps, max_fes - fes)
        for i in range(n_eval):
            u_fit = float(evaluate(U[i]))
            fes += 1
            xu = (U[i] - pop[i]) * masks[i]
            if dim > 1:
                mu_xu = float(np.mean(xu))
                vix = math.sqrt(np.sum((xu - mu_xu) ** 2) / (dim - 1) + 1e-18)
            else:
                vix = abs(float(xu[0]))

            if u_fit <= fit[i]:
                n_success += 1
                SF.append(F_list[i])
                success_k.append(int(np.sum(masks[i])))
                success_vix.append(max(vix, 1e-12))
                success_omega.append(max(vix, 1e-12))
                if len(archive) < int(r_arc * ps):
                    archive = (
                        np.vstack([archive, pop[i]])
                        if len(archive)
                        else pop[i][None, :]
                    )
                else:
                    archive[int(rng.integers(0, len(archive)))] = pop[i]

                Y[i] = pop[i].copy()
                fit_Y[i] = fit[i]
                pop[i] = U[i]
                fit[i] = u_fit
                ct[i] = 0
                LV[i] = 0.0
                LF[i] = 0.0
                if u_fit < best_f:
                    best_f, best_x = u_fit, U[i].copy()
            else:
                # Algorithm 4: f(X) < f(U) < f(Y)
                if fit[i] < u_fit < fit_Y[i]:
                    for j in range(dim):
                        if masks[i, j] == 1:
                            LV[i, j] += deltas[i, j]
                            LF[i, j] += 1.0
                ct[i] += 1
            record()
            if fes >= max_fes:
                break

        last_SR = n_success / max(ps, 1)

        if SF:
            w = np.asarray(success_vix, dtype=float)
            w = w / (w.sum() + 1e-12)
            sf = np.asarray(SF, dtype=float)
            muF_mem[mem_pos] = float(np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-12))
            mF = 0.5 * (float(np.mean(sf)) + SF_prev_mean)
            SF_prev_mean = float(np.mean(sf))
            sigmaF = lam_sigma * math.sqrt(
                float(np.mean((sf - mF) ** 2)) + 1e-12
            )
            sigmaF = float(np.clip(sigmaF, 1e-3, 0.5))

            # Eqs. (15)–(16) preference update with VIX weights
            omega = np.asarray(success_omega, dtype=float)
            counts = np.zeros(dim)
            v_sum = np.zeros(dim)
            v_cnt = np.zeros(dim)
            for k, om in zip(success_k, omega):
                kk = min(max(k, 1), dim) - 1
                counts[kk] += 1.0
                v_sum[kk] += om
                v_cnt[kk] += 1.0
            ns = counts.sum()
            if ns > 0:
                rk = counts / ns
                vk = np.where(v_cnt > 0, v_sum / np.maximum(v_cnt, 1.0), 0.0)
                xi = min(1.0 / max(ps, 1), 1.0 / dim)
                if counts[0] == ns:
                    # all success in one group edge case -> paper uses zero then xi
                    pk = np.full(dim, xi)
                else:
                    denom = float(np.sum(vk) + 1e-12)
                    pk = np.where(
                        counts > 0, rk * vk / denom + xi, xi
                    )
                Prob[mem_pos] = pk / (pk.sum() + 1e-12)
            mem_pos = (mem_pos + 1) % H

        ps_new = int(round((ps_min - ps_init) * fes / max_fes + ps_init))
        ps_new = max(ps_min, min(ps, ps_new))
        if ps_new < ps:
            keep = np.argsort(fit)[:ps_new]
            pop, fit = pop[keep], fit[keep]
            Y, fit_Y = Y[keep], fit_Y[keep]
            LV, LF, ct = LV[keep], LF[keep], ct[keep]
            ps = ps_new
            max_arc = max(1, int(r_arc * ps))
            if len(archive) > max_arc:
                archive = archive[rng.choice(len(archive), max_arc, replace=False)]

    while cp_i < len(checkpoints):
        curve[cp_i] = best_f
        cp_i += 1
    return best_f, best_x, fes, curve, {"population": ps_init}
