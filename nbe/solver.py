import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constants import MIN_NUMBER, RHO_CRITICAL, T_TODAY
from .dof import DofTable
from .mathutils import bessel_k2_approx


def neq(mass: float, x: float, g_dm: float) -> float:
    return mass**3 * g_dm * bessel_k2_approx(x) / (2.0 * math.pi**2 * x)


def yeq(mass: float, x: float, g_dm: float, dof: DofTable) -> float:
    temp = mass / x
    return neq(mass, x, g_dm) / dof.s(temp)


@dataclass
class NBEResult:
    oh2_nbe: float
    xy: np.ndarray


def solve_nbe(params: dict, settings: dict, svx: Callable[[float], float], dof: DofTable) -> NBEResult:
    m_dm = float(params["mDM"])
    g_dm = float(params["gDM"])
    xmin = float(settings["xmin"])
    xmax = float(settings["xmax"])
    h = float(settings["nBEhinit"])
    eps = float(settings["nBEerr"])
    safety = float(settings["nBEsafety"])

    xi = xmin
    yi = yeq(m_dm, xi, g_dm, dof)
    yeqi = yi
    table = [(xi, yi)]

    def lam(x: float) -> float:
        temp = m_dm / x
        return 3.2262726e18 * dof.isqrtgstar(temp) * m_dm / x**2 * svx(x)

    lambdai = lam(xi)

    while xi < xmax:
        hh = h
        if xi + hh > xmax:
            hh = xmax - xi

        while True:
            xip1 = xi + hh
            if xip1 == xi:
                raise RuntimeError("nBE solver step-size overflow")

            yeqip1 = yeq(m_dm, xip1, g_dm, dof)
            lambdaip1 = lam(xip1)
            if lambdaip1 <= 0.0:
                lambdaip1 = MIN_NUMBER

            u = hh * lambdaip1
            rho = lambdai / lambdaip1
            c = 2.0 * yi + u * ((yeqip1**2 + rho * yeqi**2) - rho * yi**2)

            disc_t = 1.0 + u * c
            if disc_t < 0.0:
                hh *= 0.5
                continue
            trap_y = c / (1.0 + math.sqrt(disc_t))

            cc = 4.0 * (yi + u * yeqip1**2)
            disc_e = 1.0 + u * cc
            if disc_e < 0.0:
                hh *= 0.5
                continue
            eul_y = cc / 2.0 / (1.0 + math.sqrt(disc_e))

            den = max(abs(trap_y), MIN_NUMBER) * eps
            err = abs((trap_y - eul_y) / den)
            if err > 1.0:
                hh = max(safety / math.sqrt(err), 0.1) * hh
                continue

            xi = xip1
            yi = trap_y
            yeqi = yeqip1
            lambdai = lambdaip1
            table.append((xi, yi))

            if err == 0.0:
                h = 1.5 * hh
            else:
                h = min(safety / math.sqrt(err), 5.0) * hh
            break

    entropy_today = dof.iheff(T_TODAY) * 2.0 * math.pi**2 / 45.0 * T_TODAY**3
    oh2 = m_dm * entropy_today / RHO_CRITICAL * yi
    return NBEResult(oh2_nbe=float(oh2), xy=np.array(table, dtype=float))

