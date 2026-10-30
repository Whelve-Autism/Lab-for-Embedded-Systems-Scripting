"""
ECL-SPACMA — Mao et al., Swarm Evol. Comput. 100 (2026) 102257.

Base: LSHADE-SPACMA / mLSHADE-SPACMA + ECL:
  Eqs. (23)/(36) Fwi + fitness-directed mutation
  Eqs. (37)–(41) nonlinear exponential population reduction (c=0.9)
  Eqs. (42)–(44) elastic geometric weighted archive pruning
  Adaptive-probability SQP local search after 0.75 MaxFEs (scipy SLSQP)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def _bound(vi, pop, lower, upper):
    vi = vi.copy()
    for i in range(len(vi)):
        for j in range(len(vi[i])):
            if vi[i, j] < lower:
                vi[i, j] = (pop[i, j] + lower) / 2.0
            elif vi[i, j] > upper:
                vi[i, j] = (pop[i, j] + upper) / 2.0
    return vi


def _gn_r1_r2(rng, np1, np2, r0):
    r1 = np.empty(np1, dtype=int)
    r2 = np.empty(np1, dtype=int)
    for i in range(np1):
        a = int(rng.integers(0, np1))
        while a == r0[i]:
            a = int(rng.integers(0, np1))
        b = int(rng.integers(0, np2))
        while b == r0[i] or b == a:
            b = int(rng.integers(0, np2))
        r1[i], r2[i] = a, b
    return r1, r2


def _lehmer(good, weights):
    good = np.asarray(good, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = float(np.dot(weights, good))
    if denom <= 1e-18:
        return float(np.mean(good))
    return float(np.dot(weights, good * good) / denom)


def _elastic_prune(arc_pop, arc_fit, limit, rng, g, basek=1.1, alpha=0.05):
    """Eqs. (42)–(44): remove with probability ∝ 1/Wi (elites have large Wi)."""
    if len(arc_pop) <= limit:
        return arc_pop, arc_fit
    # ascending fitness (best first), 1-based index i
    order = np.argsort(arc_fit)
    pop = arc_pop[order].copy()
    fit = arc_fit[order].copy()
    while len(pop) > limit:
        A = len(pop)
        idx = np.arange(1, A + 1)  # i = 1..A
        W = basek * (1.0 + 2.0 * g) * np.exp(alpha * (A - idx)) + 1.0
        # removal weight ∝ 1/W (worse / small-W more likely removed)
        rem = 1.0 / W
        rem = rem / rem.sum()
        kill = int(rng.choice(A, p=rem))
        pop = np.delete(pop, kill, axis=0)
        fit = np.delete(fit, kill, axis=0)
    return pop, fit


def _plan_pop(nfes, max_fes, max_pop, min_pop, c):
    """Eqs. (37)–(41): NP(t)=round(NP0 * r^(t^k)), r=NPmin/NP0.

    Choose k so the curve passes through NPtarget at t=c:
      rc=(1-c)+c*r,  k=log(log(rc)/log(r))/log(c).
    """
    t = min(1.0, max(0.0, nfes / max_fes))
    if max_pop <= min_pop:
        return int(min_pop)
    r = min_pop / max_pop
    if r <= 0.0 or r >= 1.0:
        return int(min_pop if t >= 1.0 else max_pop)
    rc = (1.0 - c) + c * r
    # numerical guards
    rc = float(np.clip(rc, r + 1e-15, 1.0 - 1e-15))
    k = math.log(math.log(rc) / math.log(r)) / math.log(c)
    np_t = max_pop * (r ** (t ** k))
    return max(int(min_pop), int(round(np_t)))


def optimize(evaluate, lower, upper, dim, max_fes, rng, checkpoints):
    p_best_rate = 0.11
    arc_rate = 1.4
    memory_size = 5
    pop_size = 18 * dim
    max_pop_size = pop_size
    min_pop_size = 4
    L_Rate = 0.80
    c_exp = 0.9
    basek = 1.1
    alpha = 0.05
    NFE_LS = 0.75
    P_LSmin, P_LSmax = 0.01, 0.4
    First_class_percentage = 0.5

    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.empty(pop_size, dtype=float)
    nfes = 0
    bsf_fit = np.inf
    bsf_sol = pop[0].copy()

    curve = np.full(len(checkpoints), np.nan)
    cp_i = 0
    ls_success = 0
    ls_total = 0

    def record():
        nonlocal cp_i
        while cp_i < len(checkpoints) and nfes >= checkpoints[cp_i]:
            curve[cp_i] = bsf_fit
            cp_i += 1

    for i in range(pop_size):
        if nfes >= max_fes:
            fitness[i:] = np.inf
            break
        fitness[i] = float(evaluate(pop[i]))
        nfes += 1
        if fitness[i] < bsf_fit:
            bsf_fit = float(fitness[i])
            bsf_sol = pop[i].copy()
        record()

    popold = pop.copy()

    memory_sf = np.full(memory_size, 0.5)
    memory_cr = np.full(memory_size, 0.5)
    memory_pos = 0
    memory_1st = np.full(memory_size, First_class_percentage)

    archive_pop = np.empty((0, dim))
    archive_fit = np.empty(0)
    archive_NP = int(arc_rate * pop_size)

    sigma = 0.5
    xmean = rng.uniform(lower, upper, size=dim)
    Hybridization_flag = 1
    mu = max(1, pop_size // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / weights.sum()
    mueff = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    pc = np.zeros(dim)
    ps_path = np.zeros(dim)
    B = np.eye(dim)
    D = np.ones(dim)
    C = np.eye(dim)
    invsqrtC = np.eye(dim)
    eigeneval = 0
    chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    def update_cma_sizes():
        nonlocal mu, weights, mueff
        mu = max(1, pop_size // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = float(np.sum(weights) ** 2 / np.sum(weights ** 2))

    while nfes < max_fes:
        pop = popold.copy()
        sorted_index = np.argsort(fitness)
        g_prog = nfes / max_fes

        mem_rand_index = rng.integers(0, memory_size, size=pop_size)
        mu_sf = memory_sf[mem_rand_index]
        mu_cr = memory_cr[mem_rand_index]
        mem_rand_ratio = rng.random(pop_size)

        cr = rng.normal(mu_cr, 0.1)
        cr = np.where(mu_cr < 0, 0.0, cr)
        cr = np.clip(cr, 0.0, 1.0)

        # Alg.1 lines 12–15: Fi = 0.5 + 0.05*rand in first half
        if nfes < 0.5 * max_fes:
            sf = 0.5 + 0.05 * rng.random(pop_size)
        else:
            sf = mu_sf + 0.1 * np.tan(np.pi * (rng.random(pop_size) - 0.5))
            while np.any(sf <= 0):
                pos = sf <= 0
                sf[pos] = mu_sf[pos] + 0.1 * np.tan(
                    np.pi * (rng.random(np.sum(pos)) - 0.5)
                )
        sf = np.minimum(sf, 1.0)

        # Eq. (23) Fwi — three segments: 0.7 / 0.8 / 1.2
        t = nfes / max_fes
        if t < 0.2:
            Fwi = 0.7 * sf
        elif t < 0.4:
            Fwi = 0.8 * sf
        else:
            Fwi = 1.2 * sf

        Class_Select = memory_1st[mem_rand_index] >= mem_rand_ratio
        if Hybridization_flag == 0:
            Class_Select[:] = True

        r0 = np.arange(pop_size)
        if len(archive_pop):
            popAll = np.vstack([pop, archive_pop])
        else:
            popAll = pop
        r1, r2 = _gn_r1_r2(rng, pop_size, len(popAll), r0)

        pNP = max(int(round(p_best_rate * pop_size)), 2)
        randindex = rng.integers(0, pNP, size=pop_size)
        pbest = pop[sorted_index[randindex]]

        # Eqs. (21)–(22): linear Rank_i = 3*(NP-i)+1, p_i ∝ Rank
        # i = 1..NP is rank position after ascending fitness sort
        rank_pos = np.empty(pop_size)
        rank_pos[sorted_index] = np.arange(1, pop_size + 1)
        Rank = 3.0 * (pop_size - rank_pos) + 1.0
        Pr = Rank / Rank.sum()
        press_idx = rng.choice(pop_size, size=pop_size, replace=True, p=Pr)
        pop_press = pop[press_idx]

        vi = np.zeros((pop_size, dim))
        idx1 = np.where(Class_Select)[0]
        for i in idx1:
            xpr1 = pop_press[i]
            xr2 = popAll[r2[i]]
            f_pr1 = float(fitness[press_idx[i]])
            if r2[i] < pop_size:
                f_r2 = float(fitness[r2[i]])
            else:
                f_r2 = (
                    float(archive_fit[r2[i] - pop_size])
                    if len(archive_fit)
                    else f_pr1
                )
            # Eq. (36)
            if f_pr1 <= f_r2:
                diff2 = xpr1 - xr2
            else:
                diff2 = xr2 - xpr1
            vi[i] = pop[i] + Fwi[i] * (pbest[i] - pop[i]) + sf[i] * diff2

        idx2 = np.where(~Class_Select)[0]
        if len(idx2) and Hybridization_flag:
            try:
                for i in idx2:
                    z = rng.normal(0, 1, dim)
                    vi[i] = xmean + sigma * (B @ (D * z))
            except Exception:
                # fall back: treat CMA class as DE class this generation
                Hybridization_flag = 0
                for i in idx2:
                    xpr1 = pop_press[i]
                    xr2 = popAll[r2[i]]
                    f_pr1 = float(fitness[press_idx[i]])
                    if r2[i] < pop_size:
                        f_r2 = float(fitness[r2[i]])
                    else:
                        f_r2 = (
                            float(archive_fit[r2[i] - pop_size])
                            if len(archive_fit)
                            else f_pr1
                        )
                    diff2 = xpr1 - xr2 if f_pr1 <= f_r2 else xr2 - xpr1
                    vi[i] = pop[i] + Fwi[i] * (pbest[i] - pop[i]) + sf[i] * diff2

        if not np.isrealobj(vi) or np.any(~np.isfinite(vi)):
            Hybridization_flag = 0
            # repair non-finite rows with parent (do NOT skip generation)
            bad = ~np.isfinite(vi).all(axis=1)
            vi[bad] = pop[bad]

        vi = _bound(vi, pop, lower, upper)

        mask = rng.random((pop_size, dim)) > cr[:, None]
        jrand = rng.integers(0, dim, size=pop_size)
        mask[np.arange(pop_size), jrand] = False
        ui = vi.copy()
        ui[mask] = pop[mask]

        children = np.empty(pop_size)
        for i in range(pop_size):
            if nfes >= max_fes:
                children[i] = np.inf
                continue
            children[i] = float(evaluate(ui[i]))
            nfes += 1
            if children[i] < bsf_fit:
                bsf_fit = children[i]
                bsf_sol = ui[i].copy()
            record()

        dif = np.abs(fitness - children)
        child_better = fitness > children

        if np.any(child_better):
            add_p = popold[child_better]
            add_f = fitness[child_better]
            if len(archive_pop) == 0:
                archive_pop, archive_fit = add_p.copy(), add_f.copy()
            else:
                archive_pop = np.vstack([archive_pop, add_p])
                archive_fit = np.concatenate([archive_fit, add_f])
            archive_pop, archive_fit = _elastic_prune(
                archive_pop,
                archive_fit,
                max(1, archive_NP),
                rng,
                g_prog,
                basek,
                alpha,
            )

        goodCR = cr[child_better]
        goodF = sf[child_better]
        dif_val = dif[child_better]
        dif1 = dif[child_better & Class_Select]
        dif2 = dif[child_better & ~Class_Select]

        improve = children <= fitness
        popold = pop.copy()
        popold[improve] = ui[improve]
        fitness = np.where(improve, children, fitness)

        if len(goodCR) > 0 and dif_val.sum() > 0:
            w = dif_val / dif_val.sum()
            if nfes > 0.5 * max_fes:
                memory_sf[memory_pos] = _lehmer(goodF, w)
            if np.max(goodCR) == 0 or memory_cr[memory_pos] < 0:
                memory_cr[memory_pos] = -1
            else:
                memory_cr[memory_pos] = _lehmer(goodCR, w)
            if Hybridization_flag and (dif1.sum() + dif2.sum()) > 0:
                memory_1st[memory_pos] = (
                    memory_1st[memory_pos] * L_Rate
                    + (1 - L_Rate)
                    * (dif1.sum() / (dif1.sum() + dif2.sum() + 1e-12))
                )
                memory_1st[memory_pos] = float(
                    np.clip(memory_1st[memory_pos], 0.2, 0.8)
                )
            memory_pos = (memory_pos + 1) % memory_size

        plan_pop_size = _plan_pop(nfes, max_fes, max_pop_size, min_pop_size, c_exp)
        if pop_size > plan_pop_size:
            reduction = pop_size - plan_pop_size
            for _ in range(reduction):
                worst = int(np.argmax(fitness))
                popold = np.delete(popold, worst, axis=0)
                fitness = np.delete(fitness, worst, axis=0)
                pop_size -= 1
            archive_NP = max(1, int(round(arc_rate * pop_size)))
            archive_pop, archive_fit = _elastic_prune(
                archive_pop, archive_fit, archive_NP, rng, g_prog, basek, alpha
            )
            update_cma_sizes()

        if Hybridization_flag and pop_size >= 2 * mu:
            try:
                popindex = np.argsort(fitness)
                xold = xmean.copy()
                xmean = popold[popindex[:mu]].T @ weights
                ps_path = (1 - cs) * ps_path + math.sqrt(cs * (2 - cs) * mueff) * (
                    invsqrtC @ ((xmean - xold) / sigma)
                )
                hsig = float(
                    np.sum(ps_path ** 2)
                    / (1 - (1 - cs) ** (2 * nfes / max(pop_size, 1)))
                    / dim
                    < 2 + 4 / (dim + 1)
                )
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (
                    (xmean - xold) / sigma
                )
                artmp = (1 / sigma) * (popold[popindex[:mu]].T - xold[:, None])
                C = (
                    (1 - c1 - cmu) * C
                    + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                    + cmu * artmp @ np.diag(weights) @ artmp.T
                )
                sigma = sigma * math.exp(
                    (cs / damps) * (np.linalg.norm(ps_path) / chiN - 1)
                )
                if nfes - eigeneval > pop_size / (c1 + cmu) / dim / 10:
                    eigeneval = nfes
                    C = np.triu(C) + np.triu(C, 1).T
                    if np.any(~np.isfinite(C)):
                        Hybridization_flag = 0
                    else:
                        Ddiag, Bmat = np.linalg.eigh(C)
                        Ddiag = np.sqrt(np.maximum(Ddiag, 1e-20))
                        B, D = Bmat, Ddiag
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
            except Exception:
                Hybridization_flag = 0

        # Adaptive SQP local search (Alg.1 lines 40–47)
        # Faithful mode: no silent Gaussian fallback; every real eval counts immediately.
        if nfes > NFE_LS * max_fes and nfes < max_fes:
            rate = ls_success / max(ls_total, 1)
            PLS = P_LSmin + rate * (P_LSmax - P_LSmin)
            if rng.random() < PLS:
                ls_total += 1
                budget = min(int(0.02 * max_fes), max_fes - nfes)
                if budget > 0:
                    n_calls = [0]
                    x0 = bsf_sol.copy()
                    bsf_before = float(bsf_fit)
                    ls_best_x = x0.copy()
                    ls_best_f = bsf_before

                    class _LSBudgetDone(Exception):
                        pass

                    def obj(z):
                        nonlocal nfes, bsf_fit, bsf_sol, ls_best_x, ls_best_f
                        if n_calls[0] >= budget or nfes >= max_fes:
                            raise _LSBudgetDone()
                        val = float(evaluate(z))
                        n_calls[0] += 1
                        nfes += 1
                        zc = np.asarray(z, dtype=float).copy()
                        if val < ls_best_f:
                            ls_best_f = val
                            ls_best_x = zc
                        if val < bsf_fit:
                            bsf_fit = val
                            bsf_sol = zc
                        record()
                        return val

                    bounds = [(float(lower), float(upper))] * dim
                    try:
                        minimize(
                            obj,
                            x0,
                            method="SLSQP",
                            bounds=bounds,
                            options={
                                "maxiter": max(1, budget),
                                "ftol": 1e-12,
                                "disp": False,
                            },
                        )
                    except _LSBudgetDone:
                        pass
                    except Exception:
                        # No silent Gaussian fallback; keep any LS-evaluated best.
                        pass

                    if n_calls[0] > 0 and ls_best_f < bsf_before:
                        ls_success += 1
                        wi = int(np.argmax(fitness))
                        popold[wi] = np.clip(ls_best_x, lower, upper)
                        fitness[wi] = ls_best_f
                        bsf_fit, bsf_sol = ls_best_f, popold[wi].copy()

    while cp_i < len(checkpoints):
        curve[cp_i] = bsf_fit
        cp_i += 1
    return bsf_fit, bsf_sol, nfes, curve, {"population": max_pop_size}
