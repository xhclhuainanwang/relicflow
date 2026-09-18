import math
import os
import time
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from scipy import integrate
from scipy import interpolate

from .constants import MIN_NUMBER, X_BEXP
from .mathutils import bessel_k1_approx, bessel_k2_approx
from .models.base import ThermalChannel


def _eval_many(func: Callable[[float], float], xs: np.ndarray, workers: int) -> list[float]:
    if workers <= 1 or len(xs) <= 1:
        return [float(func(x)) for x in xs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [float(v) for v in pool.map(func, xs)]

#python relicflow/relicflow.py A4 parameters2
@dataclass
class LogLogInterp:
    x: np.ndarray
    y: np.ndarray
    x_min: float
    x_max: float
    y_log_min: float
    y_log_max: float
    interp_obj: object
    log_x: np.ndarray
    coeffs: np.ndarray | None

    @classmethod
    def from_table(cls, table_xy: np.ndarray) -> "LogLogInterp":
        x = table_xy[:, 0]
        y = np.maximum(table_xy[:, 1], MIN_NUMBER)
        lx = np.log(x)
        ly = np.log(y)
        coeffs = None
        if len(x) >= 4:
            # A cubic spline can overshoot badly when a thresholded rate
            # reaches MIN_NUMBER and then remains on that floor.  In log-log
            # space this creates artificial rate spikes between floor points.
            # PCHIP preserves the tabulated monotonicity and keeps a flat
            # numerical floor flat.
            interp_obj = interpolate.PchipInterpolator(lx, ly, extrapolate=True)
            # Reuse SciPy's exact PCHIP interval coefficients in the scalar
            # hot path instead of calling the interpolator object each time.
            coeffs = np.asarray(interp_obj.c, dtype=float)
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
            log_x=lx,
            coeffs=coeffs,
        )

    def __call__(self, x: float) -> float:
        xx = float(x)
        if xx <= self.x_min:
            return float(math.exp(self.y_log_min))
        if xx >= self.x_max:
            return float(math.exp(self.y_log_max))
        lx = math.log(xx)
        if self.coeffs is not None:
            idx = min(max(int(np.searchsorted(self.log_x, lx, side="right")) - 1, 0), self.coeffs.shape[1] - 1)
            dx = lx - float(self.log_x[idx])
            c = self.coeffs[:, idx]
            ly = float(((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3])
        else:
            ly = float(self.interp_obj(lx))
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


def thermal_kernel_pair_dm(
    x: float,
    s: float,
    svfun: Callable[[float], float],
    m_dm: float,
    m1: float,
    m2: float,
) -> float:
    temp = m_dm / x
    if temp <= 0.0 or m1 <= 0.0 or m2 <= 0.0:
        return MIN_NUMBER
    s_min = (m1 + m2) ** 2
    if s <= s_min:
        return MIN_NUMBER
    lam12 = (s**2 + m1**4 + m2**4 - 2.0 * s * m1**2 - 2.0 * s * m2**2 - 2.0 * m1**2 * m2**2)
    if lam12 <= 0.0:
        return MIN_NUMBER
    x1 = m1 / temp
    x2 = m2 / temp
    den = 8.0 * temp * m1**2 * m2**2 * math.sqrt(s) * bessel_k2_approx(x1) * bessel_k2_approx(x2)
    if den <= 0.0 or not np.isfinite(den):
        return MIN_NUMBER
    val = ((s - m1**2 - m2**2) * math.sqrt(lam12) * bessel_k1_approx(math.sqrt(s) / temp)) / den * svfun(s)
    if not np.isfinite(val) or val <= 0.0:
        return MIN_NUMBER
    return float(val)


def thermal_average_channel(
    x: float,
    channel: ThermalChannel,
    m_dm: float,
    precision_goal: float,
) -> float:
    if channel.thermal_average_x is not None:
        val = float(channel.thermal_average_x(x))
        if np.isfinite(val) and val > 0.0:
            return val
        return MIN_NUMBER

    s_lo = channel.s_min
    s_hi = s_lo * (1.0 + 20.0 / x)
    epsrel = 10.0 ** (-float(precision_goal))
    m1 = channel.initial_m1 if channel.initial_m1 is not None else m_dm
    m2 = channel.initial_m2 if channel.initial_m2 is not None else m_dm
    use_identical = abs(float(m1) - float(m_dm)) <= 1.0e-15 and abs(float(m2) - float(m_dm)) <= 1.0e-15
    kernel = thermal_kernel_identical_dm if use_identical else thermal_kernel_pair_dm
    points = [s_lo]
    for bp in getattr(channel, "breakpoints", ()):
        try:
            bpf = float(bp)
        except (TypeError, ValueError):
            continue
        if s_lo < bpf < s_hi:
            points.append(bpf)
    points.append(s_hi)
    points = sorted(set(points))

    val = 0.0
    for a, b in zip(points[:-1], points[1:]):
        if b <= a:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            sub_val, _ = integrate.quad(
                lambda ss: kernel(x, ss, channel.sigma_s, m_dm)
                if kernel is thermal_kernel_identical_dm
                else kernel(x, ss, channel.sigma_s, m_dm, float(m1), float(m2)),
                a,
                b,
                epsrel=epsrel,
                epsabs=0.0,
                limit=200,
            )
        val += float(sub_val)
    if not np.isfinite(val) or val <= 0.0:
        return MIN_NUMBER
    return float(val)


def tabulate_1d_loglog(
    func: Callable[[float], float],
    xmin: float,
    xmax: float,
    nx: int,
    iacc: float,
    imax: int,
    workers: int = 1,
) -> np.ndarray:
    tx1 = [xmin / 5.0 * ((1000.0 / (xmin / 5.0)) ** ((i - 1) / (nx - 1))) for i in range(1, nx + 1)]
    tx2 = [1000.0 * ((xmax / 1000.0) ** (i / 10.0)) for i in range(2, 11)]
    tx = np.array(sorted(set(tx1 + tx2)), dtype=float)
    fvals = np.array([max(v, MIN_NUMBER) for v in _eval_many(func, tx, workers)], dtype=float)
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
        midpoints = 0.5 * (table[1:, 0] + table[:-1, 0])
        widths = np.abs(table[1:, 0] - table[:-1, 0])
        refine = np.abs(icubic(midpoints) - ilinear(midpoints)) > iacc * widths
        add = midpoints[refine].tolist()
        if add:
            add = sorted(set(add))
            x_add = np.exp(np.array(add))
            y_add = np.array([max(v, MIN_NUMBER) for v in _eval_many(func, x_add, workers)], dtype=float)
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
    workers_default = 1
    workers = max(1, int(settings.get("thermalWorkers", settings.get("workers", workers_default))))
    channel_workers_default = 1
    channel_workers = max(1, int(settings.get("channelWorkers", channel_workers_default)))
    per_channel_workers = max(1, int(settings.get("thermalWorkersPerChannel", workers // channel_workers)))

    channel_tables: dict[str, np.ndarray] = {}
    channel_interps: dict[str, LogLogInterp] = {}
    channel_funcs: dict[str, Callable[[float], float]] = {}
    channel_timings: dict[str, float] = {}

    def build_one_channel(ch: ThermalChannel) -> tuple[str, np.ndarray, float]:
        t_ch0 = time.perf_counter()
        tab = tabulate_1d_loglog(
            lambda x, ch=ch: thermal_average_channel(x, ch, m_dm, pg),
            xmin,
            xmax,
            nx,
            iacc,
            imax,
            per_channel_workers,
        )
        return ch.name, tab, time.perf_counter() - t_ch0

    if channel_workers > 1 and len(channels) > 1:
        with ThreadPoolExecutor(max_workers=channel_workers) as pool:
            built_channels = list(pool.map(build_one_channel, channels))
    else:
        built_channels = [build_one_channel(ch) for ch in channels]

    for name, tab, elapsed in built_channels:
        channel_timings[name] = elapsed
        channel_tables[name] = tab
        channel_interps[name] = LogLogInterp.from_table(tab)
        channel_funcs[name] = channel_interps[name]

    def svx(x: float) -> float:
        return float(sum(ch_interp(x) for ch_interp in channel_interps.values()))

    return svx, {
        "tables": channel_tables,
        "channel_svx_funcs": channel_funcs,
        "timing_s": {
            "channels": channel_timings,
            "build_total_svx": time.perf_counter() - t_build0,
            "thermal_workers": float(workers),
            "channel_workers": float(channel_workers),
            "thermal_workers_per_channel": float(per_channel_workers),
        },
    }
