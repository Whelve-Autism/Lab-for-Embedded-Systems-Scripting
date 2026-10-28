"""
Patch opfunu 1.0.1 modified_schwefel_func to the corrected formula
from opfunu >= 1.0.2 (official fix).

opfunu 1.0.4 cannot install on Python 3.12+ (Requires-Python <3.12),
so we apply the upstream fix in-process.
"""

from __future__ import annotations

import numpy as np


def modified_schwefel_func_fixed(x):
    """Corrected CEC modified Schwefel (opfunu 1.0.2+ / upstream master)."""
    z = np.array(x).ravel() + 4.209687462275036e+002
    nx = len(z)

    mask1 = z > 500
    mask2 = z < -500
    mask3 = ~mask1 & ~mask2
    fx = np.zeros(nx)
    fx[mask1] -= (
        (500.0 - np.fmod(z[mask1], 500))
        * np.sin(np.sqrt(500.0 - np.fmod(z[mask1], 500)))
        - ((z[mask1] - 500.0) / 100.0) ** 2 / nx
    )
    fx[mask2] -= (
        (-500.0 + np.fmod(np.abs(z[mask2]), 500))
        * np.sin(np.sqrt(500.0 - np.fmod(np.abs(z[mask2]), 500)))
        - ((z[mask2] + 500.0) / 100.0) ** 2 / nx
    )
    fx[mask3] -= z[mask3] * np.sin(np.sqrt(np.abs(z[mask3])))

    return np.sum(fx) + 4.189828872724338e+002 * nx


def apply_schwefel_patch() -> str:
    """Monkey-patch opfunu.utils.operator. Returns a short status string."""
    from opfunu.utils import operator

    probe_old = float(
        operator.modified_schwefel_func(np.array([500.0], dtype=float))
    )
    operator.modified_schwefel_func = modified_schwefel_func_fixed
    probe_new = float(
        operator.modified_schwefel_func(np.array([500.0], dtype=float))
    )
    return (
        f"schwefel_patch applied: probe@500 "
        f"before={probe_old:.6f} after={probe_new:.6f}"
    )
