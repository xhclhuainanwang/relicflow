import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate
from scipy import interpolate

from .constants import MIN_NUMBER, X_BEXP
from .mathutils import bessel_k1_approx, bessel_k2_approx
from .models.base import ThermalChannel


@dataclass
class LogLogInterp:
    x: np.ndarray
    y: np.ndarray
    x_min: float
    x_max: float
    y_log_min: float
    y_log_max: float
    interp_obj: object

    @classmethod
    def from_table(cls, table_xy: np.ndarray) -> "LogLogInterp":
        x = table_xy[:, 0]
        y = np.maximum(table_xy[:, 1], MIN_NUMBER)
        lx = np.log(x)
        ly = np.log(y)
        if len(x) >= 4:
            interp_obj = interpolate.CubicSpline(lx, ly, extrapolate=True)
        else:
            interp_obj = interpolate.interp1d(lx, ly, kind="linear", fill_value="extrapolate")
        return cls(
            x=x,
            y=y,
            x_min=float(x[0]),
            x_max=float(x[-1]),
            y_log_min=float(ly[0]),
            y_log_max=float(ly[-1]),
            interp_obj=interp_obj,
        )

    def __call__(self, x: float) -> float:
        xx = float(x)
        if xx <= self.x_min:
            return float(math.exp(self.y_log_min))
        if xx >= self.x_max:
            return float(math.exp(self.y_log_max))
        ly = float(self.interp_obj(math.log(xx)))
        return float(math.exp(ly))


def thermal_kernel_identical_dm(x: float, s: float, svfun: Callable[[float], float], m_dm: float) -> float:
    if x < X_BEXP:
        z = math.sqrt(s) / (m_dm / x)
        num = (s - 2.0 * m_dm**2) * math.sqrt(max(s - 4.0 * m_dm**2, 0.0)) * bessel_k1_approx(z)
        den = 8.0 * (m_dm / x) * m_dm**4 * bessel_k2_approx(x) ** 2
        return num / den * svfun(s)

    st = s / (4.0 * m_dm**2)
    if st <= 1.0:
        return 0.0
    pref = (
        math.exp(-2.0 * (-1.0 + math.sqrt(st)) * x)
        * math.sqrt(st - 1.0)
        * (-1.0 + 2.0 * st)
        * math.sqrt(1.0 / x)
        * (-15.0 + 24.0 * math.sqrt(st) * (-15.0 + 4.0 * x) + 16.0 * st * (285.0 - 120.0 * x + 32.0 * x**2))
        / (256.0 * math.sqrt(math.pi) * st ** 1.25)
    )
    return 1.0 / (4.0 * m_dm**2) * pref * svfun(s)


def thermal_average_channel(
    x: float,
    channel: ThermalChannel,
    m_dm: float,
    precision_goal: float,
) -> float:
    s_lo = channel.s_min
    s_hi = s_lo * (1.0 + 20.0 / x)
    epsrel = 10.0 ** (-float(precision_goal))
    val, _ = integrate.quad(
        lambda ss: thermal_kernel_identical_dm(x, ss, channel.sigma_s, m_dm),
        s_lo,
        s_hi,
        epsrel=epsrel,
        epsabs=0.0,
        limit=200,
    )
    if not np.isfinite(val) or val <= 0.0:
        return MIN_NUMBER
    return float(val)


def tabulate_1d_loglog(func: Callable[[float], float], xmin: float, xmax: float, nx: int, iacc: float, imax: int) -> np.ndarray:
    tx1 = [xmin / 5.0 * ((1000.0 / (xmin / 5.0)) ** ((i - 1) / (nx - 1))) for i in range(1, nx + 1)]
    tx2 = [1000.0 * ((xmax / 1000.0) ** (i / 10.0)) for i in range(2, 11)]
    tx = np.array(sorted(set(tx1 + tx2)), dtype=float)
    fvals = np.array([max(float(func(x)), MIN_NUMBER) for x in tx], dtype=float)
    table = np.column_stack((np.log(tx), np.log(fvals)))

    step = 0
    add = [1.0]
    while step == 0 or (len(add) != 0 and step < imax):
        step += 1
        add = []
        xlog = table[:, 0]
        ylog = table[:, 1]
        if len(table) >= 4:
            ilinear = interpolate.interp1d(xlog, ylog, kind="linear")
            icubic = interpolate.CubicSpline(xlog, ylog)
        else:
            ilinear = interpolate.interp1d(xlog, ylog, kind="linear")
            icubic = ilinear
        for i in range(1, len(table)):
            lxm = 0.5 * (table[i, 0] + table[i - 1, 0])
            if abs(float(icubic(lxm)) - float(ilinear(lxm))) > iacc * abs(table[i, 0] - table[i - 1, 0]):
                add.append(lxm)
        if add:
            add = sorted(set(add))
            x_add = np.exp(np.array(add))
            y_add = np.array([max(float(func(x)), MIN_NUMBER) for x in x_add], dtype=float)
            add_table = np.column_stack((np.log(x_add), np.log(y_add)))
            table = np.vstack((table, add_table))
            table = table[np.argsort(table[:, 0])]

    while len(table) >= 2 and abs(table[-1, 1] - table[-2, 1]) > 0.01 * abs(table[-1, 1]) and table[-1, 1] > math.log10(MIN_NUMBER):
        new_lx = table[-1, 0] + 1.0
        new_x = math.exp(new_lx)
        new_y = max(float(func(new_x)), MIN_NUMBER)
        table = np.vstack((table, np.array([[new_lx, math.log(new_y)]], dtype=float)))

    return np.column_stack((np.exp(table[:, 0]), np.exp(table[:, 1])))


def build_total_svx(
    channels: list[ThermalChannel],
    settings: dict,
    m_dm: float,
) -> tuple[Callable[[float], float], dict]:
    t_build0 = time.perf_counter()
    xmin = float(settings["xmin"])
    xmax = float(settings["xmax"])
    nx = int(settings.get("Nx", 50))
    iacc = float(settings.get("iacc", 0.1))
    imax = int(settings.get("imax", 5))
    pg = float(settings.get("pg", 5))

    channel_tables: dict[str, np.ndarray] = {}
    channel_interps: dict[str, LogLogInterp] = {}
    channel_funcs: dict[str, Callable[[float], float]] = {}
    channel_timings: dict[str, float] = {}
    for ch in channels:
        t_ch0 = time.perf_counter()
        tab = tabulate_1d_loglog(
            lambda x, ch=ch: thermal_average_channel(x, ch, m_dm, pg),
            xmin,
            xmax,
            nx,
            iacc,
            imax,
        )
        channel_timings[ch.name] = time.perf_counter() - t_ch0
        channel_tables[ch.name] = tab
        channel_interps[ch.name] = LogLogInterp.from_table(tab)
        channel_funcs[ch.name] = channel_interps[ch.name]

    def svx(x: float) -> float:
        return float(sum(ch_interp(x) for ch_interp in channel_interps.values()))

    return svx, {
        "tables": channel_tables,
        "channel_svx_funcs": channel_funcs,
        "timing_s": {
            "channels": channel_timings,
            "build_total_svx": time.perf_counter() - t_build0,
        },
    }
