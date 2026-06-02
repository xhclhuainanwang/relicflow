from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import M_PL_REDUCED


@dataclass
class DofTable:
    t: np.ndarray
    heff: np.ndarray
    sqrtgeff: np.ndarray
    sqrtgstar: np.ndarray
    dheff_dt: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path) -> "DofTable":
        arr = np.loadtxt(path, skiprows=1)
        arr = arr[np.argsort(arr[:, 0])]
        t = arr[:, 0]
        heff = arr[:, 1]
        sqrtgeff = arr[:, 2]
        sqrtgstar = arr[:, 3]
        dheff_dt = np.gradient(heff, t)
        return cls(t=t, heff=heff, sqrtgeff=sqrtgeff, sqrtgstar=sqrtgstar, dheff_dt=dheff_dt)

    def _interp(self, x: float | np.ndarray, xp: np.ndarray, fp: np.ndarray) -> float | np.ndarray:
        xx = np.asarray(x, dtype=float)
        yy = np.interp(xx, xp, fp, left=fp[0], right=fp[-1])
        if np.isscalar(x):
            return float(yy)
        return yy

    def iheff(self, temp: float | np.ndarray) -> float | np.ndarray:
        return self._interp(temp, self.t, self.heff)

    def isqrtgeff(self, temp: float | np.ndarray) -> float | np.ndarray:
        return self._interp(temp, self.t, self.sqrtgeff)

    def isqrtgstar(self, temp: float | np.ndarray) -> float | np.ndarray:
        return self._interp(temp, self.t, self.sqrtgstar)

    def gtilde(self, temp: float | np.ndarray) -> float | np.ndarray:
        return np.asarray(temp) / 3.0 * self._interp(temp, self.t, self.dheff_dt) / self.iheff(temp)

    def H(self, temp: float | np.ndarray) -> float | np.ndarray:
        return np.pi / (3.0 * np.sqrt(10.0)) * self.isqrtgeff(temp) * np.asarray(temp) ** 2 / M_PL_REDUCED

    def s(self, temp: float | np.ndarray) -> float | np.ndarray:
        return self.iheff(temp) * (2.0 * np.pi**2) / 45.0 * np.asarray(temp) ** 3

