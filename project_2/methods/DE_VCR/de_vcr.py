"""
DE-VCR — Zhao et al., Swarm Evol. Comput. 104 (2026) 102359.

  - Archive A = gamma-transformed inferiors (p=0.25); B = raw SHADE archive
  - λ from Eq.(16) Gamma *density at FT* (not a Gamma random sample)
  - Alg.2 transform applied when building A after successful replacement
  - Mutation Eq.(15): r1 ~ P∪A, r2 ~ P∪B
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln


def _init_ps(dim: int) -> int:
    return max(4, int(round(25.0 * math.log(dim) * math.sqrt(dim))))


def _cauchy(rng, mu, sigma=0.1):
    return mu + sigma * math.tan(math.pi * (rng.random() - 0.5))


def _lehmer(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    d = float(np.dot(weights, values))
    if d <= 1e-18:
        return float(np.mean(values))
    return float(np.dot(weights, values * values) / d)


def _bound_mid(x, parent, lower, upper):
    x = np.asarray(x, dtype=float).copy()
    lo = float(lower)
    hi = float(upper)
    low = x < lo
    high = x > hi
    x[low] = (parent[low] + lo) * 0.5
    x[high] = (parent[high] + hi) * 0.5
    return x


def optimize(evaluate, lower, upper, dim, max_fes, rng, checkpoints):
    ps_max = _init_ps(dim)
    ps_min = 4
    ps = ps_max
    H = 4
    p = 0.11
    k_w = 0.2
    C_stag = 7 * dim
    theta = 0.9
    phase_thr = 0.2 * max_fes
    a_min, a_max = 0.1, 2.0
    b_min, b_max = 2.0, 30.0

    muF = np.full(H, 0.6)
    muCR = np.full(H, 0.6)
    mem_i = 0

    pop = rng.uniform(lower, upper, size=(ps, dim))
    fit = np.empty(ps, dtype=float)
    fes = 0
    # A: gamma-transformed inferiors; B: raw eliminated parents (SHADE-style)
    archive_A = np.empty((0, dim))
    archive_B = np.empty((0, dim))
    counter = np.zeros(ps, dtype=int)

    best_x, best_f = pop[0].copy(), np.inf

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

    def gen_F_CR(ri):
        # Eqs. (9)–(11)
        eps = 0.1 * math.sin((rng.random() - 0.8) * math.pi)
        mu = float(muF[ri])
        if fes < phase_thr:
            Fi = math.exp(-math.pi / 10.0) * math.cos(mu * math.pi / 2.0) * (
                1.0 - abs(mu) / 2.0
            ) + eps
            Fi = float(np.clip(Fi, 1e-8, 1.0))
        else:
            Fi = _cauchy(rng, mu, 0.1)
            while Fi <= 0:
                Fi = _cauchy(rng, mu, 0.1)
            Fi = min(Fi, 1.0)
        if muCR[ri] < 0:
            CRi = 0.0
        else:
            CRi = float(rng.normal(muCR[ri], 0.1))
        if fes < phase_thr:
            CRi = max(CRi, 0.6)
        else:
            CRi = float(np.clip(CRi, 0.0, 1.0))
        return float(Fi), float(CRi)

    def gamma_density_lambda(f_x, f_u, f_best):
        # Eq. (16): λ = Gamma-PDF(FT; α, rate=β), using raw f values
        prog = fes / max_fes
        alpha = (a_max - a_min) * prog + a_min
        beta = (b_max - b_min) * prog + b_min
        alpha = max(float(alpha), 1e-12)
        beta = max(float(beta), 1e-12)
        denom = f_u + f_best
        if not np.isfinite(denom) or denom == 0.0:
            raise ValueError(f"DE-VCR Eq.16 FT denom invalid: f_u={f_u}, f_best={f_best}")
        FT = f_x / denom
        if not np.isfinite(FT) or FT <= 0.0:
            raise ValueError(f"DE-VCR Eq.16 FT invalid: FT={FT}, f_x={f_x}")
        # log λ = α log β − log Γ(α) + (α−1) log FT − β FT
        log_lam = (
            alpha * math.log(beta)
            - float(gammaln(alpha))
            + (alpha - 1.0) * math.log(FT)
            - beta * FT
        )
        lam = float(math.exp(log_lam))
        if not np.isfinite(lam):
            raise ValueError(f"DE-VCR Eq.16 λ non-finite: α={alpha}, β={beta}, FT={FT}")
        return lam

    def push_archive(arc, x, limit):
        if len(arc) < limit:
            return np.vstack([arc, x]) if len(arc) else x[None, :]
        arc = arc.copy()
        arc[int(rng.integers(0, len(arc)))] = x
        return arc

    while fes < max_fes:
        order = np.argsort(fit)
        xbest = pop[order[0]].copy()
        pbest_n = max(2, int(math.ceil(p * ps)))
        SF, SCR, weights = [], [], []

        U = np.zeros_like(pop)
        F_used = np.zeros(ps)
        CR_used = np.zeros(ps)

        for i in range(ps):
            ri = int(rng.integers(0, H))
            Fi, CRi = gen_F_CR(ri)
            F_used[i], CR_used[i] = Fi, CRi

            xp = pop[order[int(rng.integers(0, pbest_n))]].copy()
            union_A = np.vstack([pop, archive_A]) if len(archive_A) else pop
            union_B = np.vstack([pop, archive_B]) if len(archive_B) else pop

            # sample distinct-ish donors
            ia = int(rng.integers(0, len(union_A)))
            ib = int(rng.integers(0, len(union_B)))
            for _ in range(5):
                if ib != ia:
                    break
                ib = int(rng.integers(0, len(union_B)))
            r1 = union_A[ia].copy()
            r2 = union_B[ib].copy()

            # Eq. (15)
            mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (r1 - r2)
            mutant = _bound_mid(mutant, pop[i], lower, upper)
            j_rand = int(rng.integers(0, dim))
            trial = pop[i].copy()
            mask = rng.random(dim) <= CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            U[i] = trial

        n_eval = min(ps, max_fes - fes)
        u_fit = np.full(ps, np.inf)
        for i in range(n_eval):
            u_fit[i] = float(evaluate(U[i]))
            fes += 1
            # Update observed best BEFORE recording the checkpoint
            if u_fit[i] < best_f:
                best_f, best_x = float(u_fit[i]), U[i].copy()
            record()
            if fes >= max_fes:
                n_eval = i + 1
                break

        for i in range(n_eval):
            if u_fit[i] <= fit[i]:
                if u_fit[i] < fit[i]:
                    # Algorithm 1 weights
                    df1 = abs(fit[i] - u_fit[i])
                    df2 = float(np.max(np.abs(pop[i] - xbest)))
                    fk = k_w * df2 + (1.0 - k_w) * df1
                    SF.append(F_used[i])
                    SCR.append(CR_used[i])
                    weights.append(max(fk, 1e-12))

                    parent = pop[i].copy()
                    parent_fit = float(fit[i])
                    # B: raw inferior (SHADE external archive)
                    archive_B = push_archive(archive_B, parent, ps)

                    # A: with p=0.25, gamma-transform inferior (Alg.2 / Eq.16)
                    if rng.random() < 0.25:
                        lam = gamma_density_lambda(parent_fit, float(u_fit[i]), best_f)
                        xp = pop[order[int(rng.integers(0, pbest_n))]].copy()
                        transformed = _bound_mid(
                            parent + lam * (xp - parent), parent, lower, upper
                        )
                        archive_A = push_archive(archive_A, transformed, ps)

                pop[i] = U[i]
                fit[i] = u_fit[i]
                counter[i] = 0
                if u_fit[i] < best_f:
                    best_f, best_x = float(u_fit[i]), U[i].copy()
                    xbest = best_x.copy()
            else:
                # stagnation: trial not better
                counter[i] += 1
        # Keep xbest synced after selection
        order_now = np.argsort(fit)
        xbest = pop[order_now[0]].copy()
        if fit[order_now[0]] < best_f:
            best_f, best_x = float(fit[order_now[0]]), xbest.copy()

        if SF:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
            muF[mem_i] = _lehmer(SF, w)
            if np.max(SCR) == 0:
                muCR[mem_i] = -1.0
            else:
                muCR[mem_i] = _lehmer(SCR, w)
            mem_i = (mem_i + 1) % H

        # Algorithm 3 — diversity enhancement when CVCR_half > θ
        if ps >= 2 and fes < max_fes:
            Xc = pop - pop.mean(axis=0, keepdims=True)
            try:
                cov = np.cov(Xc, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)[::-1]
                eigvals = np.maximum(eigvals[idx], 0.0)
                eigvecs = eigvecs[:, idx]
                total = float(eigvals.sum()) + 1e-12
                half = max(1, dim // 2)
                cvcr_half = float(eigvals[:half].sum() / total)
                Xpca = (Xc @ eigvecs[:, :half]) @ eigvecs[:, :half].T + pop.mean(
                    axis=0
                )
            except np.linalg.LinAlgError:
                cvcr_half = 0.0
                Xpca = pop.copy()

            if cvcr_half > theta:
                center = pop.mean(axis=0)
                for i in range(ps):
                    if counter[i] > C_stag and fes < max_fes:
                        # Eq. (19)
                        pop[i] = _bound_mid(
                            pop[i]
                            + cvcr_half * (center - pop[i])
                            + float(rng.random()) * (Xpca[i] - center),
                            pop[i],
                            lower,
                            upper,
                        )
                        fit[i] = float(evaluate(pop[i]))
                        fes += 1
                        counter[i] = 0
                        if fit[i] < best_f:
                            best_f, best_x = fit[i], pop[i].copy()
                            xbest = best_x.copy()
                        record()
                        if fes >= max_fes:
                            break

        if len(archive_A) > ps:
            archive_A = archive_A[rng.choice(len(archive_A), ps, replace=False)]
        if len(archive_B) > ps:
            archive_B = archive_B[rng.choice(len(archive_B), ps, replace=False)]

        # Eq. (14) linear PSR
        ps_new = int(round((ps_min - ps_max) * fes / max_fes + ps_max))
        ps_new = max(ps_min, min(ps, ps_new))
        if ps_new < ps:
            keep = np.argsort(fit)[:ps_new]
            pop, fit, counter = pop[keep], fit[keep], counter[keep]
            ps = ps_new

    while cp_i < len(checkpoints):
        curve[cp_i] = best_f
        cp_i += 1
    return best_f, best_x, fes, curve, {"population": ps_max}
