"""Patch opfunu 1.0.1 Schwefel to the corrected upstream formula."""

from __future__ import annotations

import numpy as np


def modified_schwefel_func_fixed(x):
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
    from opfunu.utils import operator

    before = float(operator.modified_schwefel_func(np.array([500.0])))
    operator.modified_schwefel_func = modified_schwefel_func_fixed
    after = float(operator.modified_schwefel_func(np.array([500.0])))
    return f"schwefel_patch: @500 before={before:.6f} after={after:.6f}"
