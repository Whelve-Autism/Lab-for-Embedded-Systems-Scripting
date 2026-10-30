"""
NICE-DE: Niche-Informed Covariance-Enhanced Differential Evolution.

Fixed-NP SHADE (current-to-pbest/1) for global search, plus up to two
persistent niche-seeded CMA-ES islands (pycma ask/tell) for local
covariance search on separated high-quality regions.

Collaboration cycle:
  1) up to 1000 DE trial evaluations
  2) up to 100 CMA FE across active islands (global CMA cap 12000)
"""

from __future__ import annotations

import math

import cma
import numpy as np


NPOP = 100
H = 6
ARCHIVE_CAP = 100
EPS = 1e-12

MAX_ISLANDS = 2
DE_CYCLE_FE = 1000
CMA_CYCLE_FE = 100
CMA_GLOBAL_CAP = 12000
ISLAND_STALE_GENS = 40
SEPARATION_FRAC = 0.15
NOVEL_FRAC = 0.12
MAX_INJECT_PER_GEN = 2
MAX_INJECT_PER_CYCLE = 4


def _repair(x, parent, lo, hi):
    x = np.asarray(x, dtype=float)
    y = np.where(x < lo, (lo + parent) * 0.5, x)
    y = np.where(y > hi, (hi + parent) * 0.5, y)
    return np.clip(y, lo, hi)


def _fcr(rng, mf, mcr):
    k = int(rng.integers(H))
    F = -1.0
    while F <= 0:
        F = float(mf[k] + 0.1 * rng.standard_cauchy())
    F = min(F, 1.0)
    CR = float(np.clip(rng.normal(mcr[k], 0.1), 0.0, 1.0))
    return F, CR


def _median_neighbor_sigma(pop, center, dim, lo, hi):
    """sigma0 from nearest-10 median distance around a seed."""
    dist = np.linalg.norm(pop - center, axis=1)
    # exclude near-duplicates of the seed itself
    order = np.argsort(dist)
    take = []
    for j in order:
        if dist[j] > 1e-12:
            take.append(dist[j])
        if len(take) >= 10:
            break
    if not take:
        med = 0.05 * (hi - lo) * math.sqrt(dim)
    else:
        med = float(np.median(take))
    span = float(hi - lo)
    return float(np.clip(
        0.5 * med / math.sqrt(dim),
        0.001 * span,
        0.08 * span,
    ))


class CMAIsland:
    """One persistent pycma CMA-ES niche."""

    def __init__(self, x0, sigma0, dim, lo, hi, seed):
        self.d = int(dim)
        self.lo = float(lo)
        self.hi = float(hi)
        self.anchor = np.asarray(x0, dtype=float).copy()
        self.region_best = float("inf")
        self.region_best_x = self.anchor.copy()
        self.stale = 0
        self.gens = 0
        self.total_improvement = 0.0

        self.es = cma.CMAEvolutionStrategy(
            self.anchor.tolist(),
            float(sigma0),
            {
                "bounds": [self.lo, self.hi],
                "verbose": -9,
                "verb_disp": 0,
                "verb_log": 0,
                "verb_plot": 0,
                "seed": int(seed),
            },
        )
        self.lam = int(self.es.popsize)

    @property
    def mean(self):
        return np.asarray(self.es.mean, dtype=float).copy()

    @property
    def sigma(self):
        return float(self.es.sigma)

    @property
    def condition_number(self):
        try:
            return float(self.es.condition_number)
        except Exception:
            C = np.asarray(self.es.C, dtype=float)
            eig = np.linalg.eigvalsh(0.5 * (C + C.T))
            return float(max(eig[-1], EPS) / max(eig[0], EPS))

    def should_terminate(self):
        if self.stale >= ISLAND_STALE_GENS:
            return True
        if self.sigma <= 1e-12:
            return True
        if self.condition_number > 1e14:
            return True
        stop = self.es.stop()
        return bool(stop)

    def run_generation(self, evaluate, remaining_fes):
        """
        One complete ask/tell generation.
        Returns (consumed_fes, candidates[(x,f), ...]) or None if skipped.
        """
        if remaining_fes < self.lam:
            return None

        asked = self.es.ask()
        xs = []
        fs = []
        for raw in asked:
            parent = self.mean
            cand = _repair(raw, parent, self.lo, self.hi)
            val = float(evaluate(cand))
            if not np.isfinite(val):
                raise ValueError("Nonfinite CMA objective")
            xs.append(cand)
            fs.append(val)

        # tell with the same points that were evaluated
        self.es.tell(xs, fs)
        self.gens += 1

        best_idx = int(np.argmin(fs))
        best_f = float(fs[best_idx])
        best_x = xs[best_idx]
        if not np.isfinite(self.region_best):
            self.region_best = best_f
            self.region_best_x = np.asarray(best_x, dtype=float).copy()
            self.stale = 0
        else:
            tol = 1e-12 * max(1.0, abs(self.region_best))
            if best_f < self.region_best - tol:
                self.total_improvement += float(self.region_best - best_f)
                self.region_best = best_f
                self.region_best_x = np.asarray(best_x, dtype=float).copy()
                self.stale = 0
            else:
                self.stale += 1

        order = np.argsort(fs)
        candidates = [
            (xs[i], float(fs[i]))
            for i in order[:MAX_INJECT_PER_GEN]
        ]
        return self.lam, candidates


class NicheManager:
    """Maintain up to two spatially separated CMA niches."""

    def __init__(self, dim, lo, hi, rng):
        self.d = dim
        self.lo = float(lo)
        self.hi = float(hi)
        self.rng = rng
        self.islands: list[CMAIsland] = []
        self.creations = 0
        self.terminations = 0
        self.space = (self.hi - self.lo) * math.sqrt(dim)

    def _min_sep(self):
        if not self.islands:
            return 0.0
        sig = max(isl.sigma for isl in self.islands)
        return max(
            SEPARATION_FRAC * self.space,
            3.0 * sig * math.sqrt(self.d),
        )

    def _seed_candidates(self, pop, fit):
        order = np.argsort(fit)
        top = order[: max(1, NPOP // 2)]
        return top

    def maybe_create(self, pop, fit):
        while len(self.islands) < MAX_ISLANDS:
            top = self._seed_candidates(pop, fit)
            if len(self.islands) == 0:
                idx = int(top[0])
                x0 = pop[idx].copy()
            else:
                centers = np.stack([isl.mean for isl in self.islands])
                pts = pop[top]
                dist = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
                nearest = np.min(dist, axis=1)
                sep = self._min_sep()
                mask = nearest >= sep
                if not np.any(mask):
                    break
                # among separated points, prefer better fitness (earlier in top)
                rel = np.where(mask)[0]
                # quality: rank within top + novelty
                novelty = nearest[rel] / max(float(nearest[rel].max()), EPS)
                quality = 1.0 - rel / max(1, len(top) - 1)
                scores = 0.6 * quality + 0.4 * novelty
                pick = int(rel[int(np.argmax(scores))])
                x0 = pts[pick].copy()

            sigma0 = _median_neighbor_sigma(pop, x0, self.d, self.lo, self.hi)
            seed = int(self.rng.integers(1, 2**31 - 1))
            self.islands.append(
                CMAIsland(x0, sigma0, self.d, self.lo, self.hi, seed)
            )
            self.creations += 1

    def prune(self):
        kept = []
        for isl in self.islands:
            if isl.should_terminate():
                self.terminations += 1
            else:
                kept.append(isl)
        self.islands = kept

    def snapshot_sigma_cond(self):
        sigmas = [isl.sigma for isl in self.islands]
        conds = [isl.condition_number for isl in self.islands]
        return sigmas, conds


def _try_inject(x, f, pop, fit, archive, lo, hi, dim, allow_novel_worst, rng):
    """Inject one CMA elite into DE pop if strictly improving a member."""
    worse = np.where(fit > f + 1e-15)[0]
    if worse.size == 0:
        return False

    dists_worse = np.linalg.norm(pop[worse] - x, axis=1)
    target = int(worse[int(np.argmin(dists_worse))])

    space = (hi - lo) * math.sqrt(dim)
    min_all = float(np.min(np.linalg.norm(pop - x, axis=1)))
    if allow_novel_worst and min_all > NOVEL_FRAC * space:
        worst = int(np.argmax(fit))
        if fit[worst] > f + 1e-15:
            target = worst

    if not (fit[target] > f + 1e-15):
        return False

    old = pop[target].copy()
    pop[target] = np.asarray(x, dtype=float).copy()
    fit[target] = float(f)
    if len(archive) < ARCHIVE_CAP:
        archive.append(old)
    else:
        archive[int(rng.integers(ARCHIVE_CAP))] = old
    return True


def optimize(
    evaluate,
    lower,
    upper,
    dim,
    max_fes,
    rng,
    checkpoints,
):
    if max_fes < NPOP or dim < 2:
        raise ValueError("Need dim>=2 and max_fes>=NPOP")

    lower = float(lower)
    upper = float(upper)
    checkpoints = np.asarray(checkpoints, dtype=int)
    history = np.empty(len(checkpoints), dtype=float)

    pop = rng.uniform(lower, upper, (NPOP, dim))
    fit = np.asarray([float(evaluate(x)) for x in pop])
    if not np.isfinite(fit).all():
        raise ValueError("Nonfinite initial objective")
    fes = NPOP

    bi = int(np.argmin(fit))
    best_f = float(fit[bi])
    best_x = pop[bi].copy()

    cp = 0
    while cp < len(checkpoints) and fes >= checkpoints[cp]:
        history[cp] = best_f
        cp += 1

    mf = np.full(H, 0.5)
    mcr = np.full(H, 0.8)
    mempos = 0
    archive: list[np.ndarray] = []

    niches = NicheManager(dim, lower, upper, rng)

    de_fes = 0
    cma_fes = 0
    cma_insertions = 0
    cma_global_best_improvements = 0
    de_global_best_improvements = 0
    cma_total_improvement = 0.0

    def bump_best(x, v, source):
        nonlocal best_f, best_x, cma_global_best_improvements
        nonlocal de_global_best_improvements
        if v < best_f:
            best_f = float(v)
            best_x = np.asarray(x, dtype=float).copy()
            if source == "cma":
                cma_global_best_improvements += 1
            else:
                de_global_best_improvements += 1

    def mark_cp():
        nonlocal cp
        while cp < len(checkpoints) and fes >= checkpoints[cp]:
            history[cp] = best_f
            cp += 1

    def eval_de(trial):
        nonlocal fes, de_fes
        v = float(evaluate(trial))
        if not np.isfinite(v):
            raise ValueError(f"Nonfinite DE objective at FE={fes + 1}")
        fes += 1
        de_fes += 1
        bump_best(trial, v, "de")
        mark_cp()
        return v

    def eval_cma(cand):
        nonlocal fes, cma_fes
        v = float(evaluate(cand))
        if not np.isfinite(v):
            raise ValueError(f"Nonfinite CMA objective at FE={fes + 1}")
        fes += 1
        cma_fes += 1
        bump_best(cand, v, "cma")
        mark_cp()
        return v

    def de_generation(ntrials):
        nonlocal pop, fit, mempos
        if ntrials <= 0:
            return 0
        srcpop, srcfit = pop.copy(), fit.copy()
        nxtpop, nxtfit = pop.copy(), fit.copy()
        order = rng.permutation(NPOP)
        successful_f: list[float] = []
        successful_cr: list[float] = []
        gains: list[float] = []
        used = 0

        for i in order[:ntrials]:
            F, CR = _fcr(rng, mf, mcr)
            top = max(2, math.ceil(float(rng.uniform(0.05, 0.20)) * NPOP))
            pbest = int(rng.choice(np.argpartition(srcfit, top - 1)[:top]))

            r1 = int(rng.integers(NPOP - 1))
            if r1 >= i:
                r1 += 1
            total = NPOP + len(archive)
            while True:
                r2 = int(rng.integers(total))
                if r2 >= NPOP or (r2 != i and r2 != r1):
                    break
            x2 = srcpop[r2] if r2 < NPOP else archive[r2 - NPOP]

            mutant = (
                srcpop[i]
                + F * (srcpop[pbest] - srcpop[i])
                + F * (srcpop[r1] - x2)
            )
            mutant = _repair(mutant, srcpop[i], lower, upper)
            mask = rng.random(dim) < CR
            mask[int(rng.integers(dim))] = True
            trial = np.where(mask, mutant, srcpop[i])

            value = eval_de(trial)
            used += 1

            if value <= srcfit[i]:
                nxtpop[i] = trial
                nxtfit[i] = value
                if value < srcfit[i]:
                    successful_f.append(F)
                    successful_cr.append(CR)
                    gains.append(float(srcfit[i] - value))
                    if len(archive) < ARCHIVE_CAP:
                        archive.append(srcpop[i].copy())
                    else:
                        archive[int(rng.integers(ARCHIVE_CAP))] = srcpop[i].copy()

        pop, fit = nxtpop, nxtfit
        if gains:
            w = np.asarray(gains) / sum(gains)
            fs = np.asarray(successful_f)
            crs = np.asarray(successful_cr)
            mf[mempos] = np.clip(
                np.dot(w, fs ** 2) / max(np.dot(w, fs), EPS),
                0.05,
                1.0,
            )
            mcr[mempos] = np.clip(np.dot(w, crs), 0.0, 1.0)
            mempos = (mempos + 1) % H
        return used

    def cma_cycle(budget):
        nonlocal cma_insertions, cma_total_improvement
        if budget <= 0:
            return 0
        niches.maybe_create(pop, fit)
        if not niches.islands:
            return 0

        spent = 0
        cycle_inserts = 0
        # round-robin complete generations while budget remains
        while spent < budget and niches.islands and fes < max_fes:
            progressed = False
            for isl in list(niches.islands):
                remain = min(budget - spent, max_fes - fes, CMA_GLOBAL_CAP - cma_fes)
                if remain < isl.lam:
                    continue
                before_imp = isl.total_improvement
                out = isl.run_generation(eval_cma, remain)
                if out is None:
                    continue
                consumed, candidates = out
                spent += consumed
                progressed = True
                cma_total_improvement += max(
                    0.0, isl.total_improvement - before_imp
                )

                for x, fval in candidates:
                    if cycle_inserts >= MAX_INJECT_PER_CYCLE:
                        break
                    novel_ok = True
                    if _try_inject(
                        x, fval, pop, fit, archive,
                        lower, upper, dim, novel_ok, rng,
                    ):
                        cma_insertions += 1
                        cycle_inserts += 1

                if isl.should_terminate():
                    niches.islands.remove(isl)
                    niches.terminations += 1
                    niches.maybe_create(pop, fit)

            if not progressed:
                break

        niches.prune()
        return spent

    # ============================================================
    # Main collaboration loop
    # ============================================================
    while fes < max_fes:
        # --- DE phase (up to 1000 trial evaluations) ---
        de_left = min(DE_CYCLE_FE, max_fes - fes)
        de_done = 0
        while de_done < de_left and fes < max_fes:
            n = min(NPOP, de_left - de_done, max_fes - fes)
            if n <= 0:
                break
            de_done += de_generation(n)

        # --- CMA phase (optional, capped) ---
        cma_room = CMA_GLOBAL_CAP - cma_fes
        cma_budget = min(CMA_CYCLE_FE, cma_room, max_fes - fes)
        if cma_budget > 0:
            niches.maybe_create(pop, fit)
            if niches.islands:
                cma_cycle(cma_budget)
            # if no islands could be formed, skip and continue DE

    while cp < len(checkpoints):
        history[cp] = best_f
        cp += 1

    assert fes == max_fes
    assert de_fes + cma_fes == max_fes - NPOP
    assert cma_fes <= CMA_GLOBAL_CAP
    assert np.all(np.isfinite(history))
    assert np.all(np.diff(history) <= 1e-8)

    sigmas, conds = niches.snapshot_sigma_cond()
    meta = {
        "population": int(NPOP),
        "nice_diagnostics": {
            "de_fes": int(de_fes),
            "cma_fes": int(cma_fes),
            "n_active_islands": int(len(niches.islands)),
            "island_creations": int(niches.creations),
            "island_terminations": int(niches.terminations),
            "island_sigma": [float(s) for s in sigmas],
            "island_condition_number": [float(c) for c in conds],
            "cma_global_best_improvements": int(cma_global_best_improvements),
            "cma_total_improvement": float(cma_total_improvement),
            "cma_insertions": int(cma_insertions),
            "de_global_best_improvements": int(de_global_best_improvements),
        },
    }
    return best_f, best_x, fes, history, meta
