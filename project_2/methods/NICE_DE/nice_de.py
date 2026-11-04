"""
NICE-DE: Niche-Informed Covariance-Enhanced Differential Evolution.

Fixed-NP SHADE (current-to-pbest/1) for global search, plus up to two
persistent niche-seeded CMA-ES islands (pycma ask/tell) for local
covariance search on separated high-quality regions.

Collaboration cycle:
  1) up to 1000 DE trial evaluations
  2) up to 100 CMA FE across active islands (global CMA cap 12000)

Formal hard ablations (FE total fixed at max_fes):
  full               : complete NICE-DE
  no_niche           : single randomly seeded local island (no niche guidance)
  no_persistence     : destroy & re-seed local search after every CMA generation
  no_collaboration   : no injection; CMA cannot update reported global best
  no_local_search    : disable CMA entirely; former CMA FE budget returned to DE

Legacy (kept in code, not in formal four):
  no_covariance      : fixed-center isotropic Monte Carlo (no C/mean/sigma adapt)
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

VALID_ABLATIONS = {
    "full",
    "no_niche",
    "no_covariance",
    "no_persistence",
    "no_collaboration",
    "no_local_search",
}


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


def _cma_weights(dim):
    lam = 4 + int(3 * math.log(dim))
    if lam % 2:
        lam += 1
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = weights / weights.sum()
    mueff = 1.0 / float(np.sum(w ** 2))
    cs = (mueff + 2) / (dim + mueff + 5)
    damps = (
        1
        + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1)
        + cs
    )
    chi = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))
    return lam, mu, w, mueff, cs, damps, chi


class CMAIsland:
    """One persistent local-search niche (full CMA or isotropic)."""

    def __init__(
        self, x0, sigma0, dim, lo, hi, seed,
        isotropic=False, freeze_adaptation=False,
    ):
        self.d = int(dim)
        self.lo = float(lo)
        self.hi = float(hi)
        self.isotropic = bool(isotropic)
        self.freeze_adaptation = bool(freeze_adaptation)
        self.anchor = np.asarray(x0, dtype=float).copy()
        self.region_best = float("inf")
        self.region_best_x = self.anchor.copy()
        self.stale = 0
        self.gens = 0
        self.total_improvement = 0.0
        self.rng = np.random.default_rng(int(seed))
        self.sigma0 = float(sigma0)

        if self.isotropic:
            (
                self.lam, self.mu, self.w, self.mueff,
                self.cs, self.damps, self.chi,
            ) = _cma_weights(self.d)
            self.m = self.anchor.copy()
            self._sigma = float(sigma0)
            self.ps = np.zeros(self.d)
            self.es = None
        else:
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
        if self.isotropic:
            return self.m.copy()
        return np.asarray(self.es.mean, dtype=float).copy()

    @property
    def sigma(self):
        if self.isotropic:
            return float(self._sigma)
        return float(self.es.sigma)

    @sigma.setter
    def sigma(self, value):
        self._sigma = float(value)

    @property
    def condition_number(self):
        if self.isotropic:
            return 1.0
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
        if not self.isotropic and self.condition_number > 1e14:
            return True
        if not self.isotropic:
            return bool(self.es.stop())
        return False

    def _update_region_best(self, best_x, best_f):
        if not np.isfinite(self.region_best):
            self.region_best = best_f
            self.region_best_x = np.asarray(best_x, dtype=float).copy()
            self.stale = 0
            return
        tol = 1e-12 * max(1.0, abs(self.region_best))
        if best_f < self.region_best - tol:
            self.total_improvement += float(self.region_best - best_f)
            self.region_best = best_f
            self.region_best_x = np.asarray(best_x, dtype=float).copy()
            self.stale = 0
        else:
            self.stale += 1

    def run_generation(self, evaluate, remaining_fes):
        """
        One complete ask/tell generation.
        Returns (consumed_fes, candidates[(x,f), ...]) or None if skipped.
        """
        if remaining_fes < self.lam:
            return None

        if self.isotropic:
            return self._run_isotropic(evaluate)
        return self._run_cma(evaluate)

    def _run_cma(self, evaluate):
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

        self.es.tell(xs, fs)
        self.gens += 1

        best_idx = int(np.argmin(fs))
        self._update_region_best(xs[best_idx], float(fs[best_idx]))

        order = np.argsort(fs)
        candidates = [(xs[i], float(fs[i])) for i in order[:MAX_INJECT_PER_GEN]]
        return self.lam, candidates

    def _run_isotropic(self, evaluate):
        """
        Isotropic local sampling (C = I).

        Hard no_covariance: freeze center at the seed and sigma at sigma0
        (local Monte Carlo only; no mean/sigma adaptation).
        """
        center = self.anchor if self.freeze_adaptation else self.m
        sig = self.sigma0 if self.freeze_adaptation else self._sigma

        half = self.lam // 2
        Z = self.rng.standard_normal((half, self.d))
        Z = np.vstack((Z, -Z))
        xs = []
        steps = []
        fs = []
        for z in Z:
            cand = center + sig * z
            cand = _repair(cand, center, self.lo, self.hi)
            step = (cand - center) / max(sig, EPS)
            val = float(evaluate(cand))
            if not np.isfinite(val):
                raise ValueError("Nonfinite isotropic objective")
            xs.append(cand)
            steps.append(step)
            fs.append(val)

        order = np.argsort(fs)
        if not self.freeze_adaptation:
            steps_arr = np.asarray(steps, dtype=float)
            y_w = np.sum(self.w[:, None] * steps_arr[order[: self.mu]], axis=0)
            old_m = self.m.copy()
            self.m = np.clip(old_m + self._sigma * y_w, self.lo, self.hi)
            self.ps = (
                (1 - self.cs) * self.ps
                + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * y_w
            )
            path_norm = float(np.linalg.norm(self.ps))
            self._sigma *= math.exp(
                min(
                    0.5,
                    max(
                        -0.5,
                        (self.cs / self.damps) * (path_norm / self.chi - 1.0),
                    ),
                )
            )
            self._sigma = float(np.clip(self._sigma, 1e-10, self.hi - self.lo))

        self.gens += 1
        best_idx = int(order[0])
        self._update_region_best(xs[best_idx], float(fs[best_idx]))
        candidates = [
            (xs[i], float(fs[i])) for i in order[:MAX_INJECT_PER_GEN]
        ]
        return self.lam, candidates


class NicheManager:
    """Maintain up to two spatially separated local-search niches."""

    def __init__(self, dim, lo, hi, rng, ablation="full"):
        self.d = dim
        self.lo = float(lo)
        self.hi = float(hi)
        self.rng = rng
        self.ablation = ablation
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
        return order[: max(1, NPOP // 2)]

    def _pick_seed(self, pop, fit):
        # A1 hard: one random population member; no fitness/spatial niche guidance.
        if self.ablation == "no_niche":
            return pop[int(self.rng.integers(len(pop)))].copy()

        top = self._seed_candidates(pop, fit)
        if len(self.islands) == 0:
            return pop[int(top[0])].copy()

        centers = np.stack([isl.mean for isl in self.islands])
        pts = pop[top]
        dist = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        nearest = np.min(dist, axis=1)
        sep = self._min_sep()
        mask = nearest >= sep
        if not np.any(mask):
            return None
        rel = np.where(mask)[0]
        novelty = nearest[rel] / max(float(nearest[rel].max()), EPS)
        quality = 1.0 - rel / max(1, len(top) - 1)
        scores = 0.6 * quality + 0.4 * novelty
        pick = int(rel[int(np.argmax(scores))])
        return pts[pick].copy()

    def maybe_create(self, pop, fit):
        # A1 hard keeps a single unguided island only.
        cap = 1 if self.ablation == "no_niche" else MAX_ISLANDS
        while len(self.islands) < cap:
            x0 = self._pick_seed(pop, fit)
            if x0 is None:
                break
            if self.islands:
                dmin = min(
                    float(np.linalg.norm(x0 - isl.mean))
                    for isl in self.islands
                )
                if dmin < 1e-12:
                    break
            sigma0 = _median_neighbor_sigma(pop, x0, self.d, self.lo, self.hi)
            seed = int(self.rng.integers(1, 2**31 - 1))
            self.islands.append(
                CMAIsland(
                    x0,
                    sigma0,
                    self.d,
                    self.lo,
                    self.hi,
                    seed,
                    isotropic=(self.ablation == "no_covariance"),
                    freeze_adaptation=(self.ablation == "no_covariance"),
                )
            )
            self.creations += 1

    def reinit_persistent_state(self, pop, fit):
        """A3: drop adapted CMA state; re-seed niches for the new cycle."""
        for isl in self.islands:
            self.terminations += 1
        self.islands = []
        self.maybe_create(pop, fit)

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
    """Inject one local elite into DE pop if strictly improving a member."""
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
    ablation: str = "full",
):
    if ablation not in VALID_ABLATIONS:
        raise ValueError(f"Unknown ablation={ablation!r}")
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

    niches = NicheManager(dim, lower, upper, rng, ablation=ablation)

    de_fes = 0
    cma_fes = 0
    cma_insertions = 0
    cma_global_best_improvements = 0
    de_global_best_improvements = 0
    cma_total_improvement = 0.0
    allow_inject = ablation not in {"no_collaboration", "no_local_search"}
    # Collaboration hard: CMA may not update the reported global best / curve.
    cma_updates_global_best = ablation not in {"no_collaboration", "no_local_search"}
    use_local_search = ablation != "no_local_search"

    def bump_best(x, v, source):
        nonlocal best_f, best_x, cma_global_best_improvements
        nonlocal de_global_best_improvements
        if source == "cma" and not cma_updates_global_best:
            return
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

                if allow_inject:
                    for x, fval in candidates:
                        if cycle_inserts >= MAX_INJECT_PER_CYCLE:
                            break
                        if _try_inject(
                            x, fval, pop, fit, archive,
                            lower, upper, dim, True, rng,
                        ):
                            cma_insertions += 1
                            cycle_inserts += 1

                # A3 hard: drop adapted state after every generation.
                if ablation == "no_persistence":
                    niches.islands.remove(isl)
                    niches.terminations += 1
                    niches.maybe_create(pop, fit)
                elif isl.should_terminate():
                    niches.islands.remove(isl)
                    niches.terminations += 1
                    niches.maybe_create(pop, fit)

            if not progressed:
                break

        niches.prune()
        return spent

    while fes < max_fes:
        if not use_local_search:
            # A4: all remaining FE goes to global DE (no CMA branch).
            n = min(NPOP, max_fes - fes)
            if n <= 0:
                break
            de_generation(n)
            continue

        de_left = min(DE_CYCLE_FE, max_fes - fes)
        de_done = 0
        while de_done < de_left and fes < max_fes:
            n = min(NPOP, de_left - de_done, max_fes - fes)
            if n <= 0:
                break
            de_done += de_generation(n)

        cma_room = CMA_GLOBAL_CAP - cma_fes
        cma_budget = min(CMA_CYCLE_FE, cma_room, max_fes - fes)
        if cma_budget > 0:
            cma_cycle(cma_budget)

    while cp < len(checkpoints):
        history[cp] = best_f
        cp += 1

    assert fes == max_fes
    assert de_fes + cma_fes == max_fes - NPOP
    assert cma_fes <= CMA_GLOBAL_CAP
    assert np.all(np.isfinite(history))
    assert np.all(np.diff(history) <= 1e-8)
    if ablation == "no_local_search":
        assert cma_fes == 0
        assert cma_insertions == 0
        assert niches.creations == 0
    if ablation == "no_collaboration":
        assert cma_insertions == 0
        assert cma_global_best_improvements == 0

    sigmas, conds = niches.snapshot_sigma_cond()
    meta = {
        "population": int(NPOP),
        "nice_diagnostics": {
            "ablation": ablation,
            "ablation_hardness": "hard",
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
