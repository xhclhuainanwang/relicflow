import math

import numpy as np
from scipy import special

from .constants import X_BEXP


def bessel_k1_approx(x: float | np.ndarray) -> float | np.ndarray:
    xx = np.asarray(x, dtype=float)
    out = np.empty_like(xx)
    mask = xx < X_BEXP
    out[mask] = special.kv(1, xx[mask])
    xm = xx[~mask]
    out[~mask] = np.exp(-xm) * np.sqrt(np.pi / (2.0 * xm)) * (
        1.0
        + 72765.0 / (262144.0 * xm**5)
        - 4725.0 / (32768.0 * xm**4)
        + 105.0 / (1024.0 * xm**3)
        - 15.0 / (128.0 * xm**2)
        + 3.0 / (8.0 * xm)
    )
    if np.isscalar(x):
        return float(out)
    return out


def bessel_k2_approx(x: float | np.ndarray) -> float | np.ndarray:
    xx = np.asarray(x, dtype=float)
    out = np.empty_like(xx)
    mask = xx < X_BEXP
    out[mask] = special.kv(2, xx[mask])
    xm = xx[~mask]
    out[~mask] = np.exp(-xm) * np.sqrt(np.pi / (2.0 * xm)) * (
        1.0
        - 135135.0 / (262144.0 * xm**5)
        + 10395.0 / (32768.0 * xm**4)
        - 315.0 / (1024.0 * xm**3)
        + 105.0 / (128.0 * xm**2)
        + 15.0 / (8.0 * xm)
    )
    if np.isscalar(x):
        return float(out)
    return out


def neq(mass: float, x: float, g_dm: float) -> float:
    return mass**3 * g_dm * bessel_k2_approx(x) / (2.0 * math.pi**2 * x)

