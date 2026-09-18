"""Native RelicFlow adapter for the Z4/VLL model.

The implementation is a direct Python translation of the active files in
``E:\\work content\\Z3\\7.30\\DRAKE_v1.0 3``.  In particular, the source of
the model is ``models\\VLL\\VLL.wl`` and the coupled solvers are wired by the
model-local runner to ``src\\nBEl.wl`` and ``src\\cBEA.wl``.

This module contains model physics and thermal-rate tabulation.  The
``models/Z4/z4_runner.py`` module contains the two implicit Boltzmann
integrators, keeping the shared RelicFlow pipeline unchanged for other
models.
"""

from __future__ import annotations

import math
import hashlib
import json
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import integrate, interpolate, special

from ..constants import MIN_NUMBER, M_PL_REDUCED, RHO_CRITICAL, T_TODAY, X_BEXP
from ..mathutils import bessel_k1_approx, bessel_k2_approx
from ..models.base import RelicFunctions, ThermalChannel
from ..thermal import tabulate_1d_loglog


_PI = math.pi
_SAFE_FLOOR = 1.0e-120
_Z4_RATE_CACHE_SCHEMA = "z4_vll_rate_cache_v1"
_Z4_RATE_NAMES = ("svx", "svxl", "svxl1", "svxl2", "sv2x", "gs")


@lru_cache(maxsize=16)
def _gauss_legendre_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache fixed nodes for the narrow-resonance Python fallback."""

    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)


def _finite_positive(value: float, floor: float = MIN_NUMBER) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        return float(floor)
    return value


def _safe_exp(value: float) -> float:
    return math.exp(min(700.0, max(-745.0, float(value))))


def _bessel_k2_scaled(x: float) -> float:
    """Return exp(-x)/K2(x), using the same x=100 branch as DRAKE."""

    xx = max(float(x), 1.0e-12)
    if xx < X_BEXP:
        k2 = float(special.kv(2, xx))
        return _finite_positive(math.exp(-xx) / k2)
    asym = 1.0 - 135135.0 / (262144.0 * xx**5)
    asym += 10395.0 / (32768.0 * xx**4) - 315.0 / (1024.0 * xx**3)
    asym += 105.0 / (128.0 * xx**2) + 15.0 / (8.0 * xx)
    return _finite_positive(1.0 / (math.sqrt(_PI / (2.0 * xx)) * asym))


def _der_bessel_k2_over_k2(x: float) -> float:
    xx = max(float(x), 1.0e-12)
    if xx < X_BEXP:
        k2 = float(special.kv(2, xx))
        if k2 <= 0.0 or not math.isfinite(k2):
            return -1.0
        return float(0.5 * (-special.kv(1, xx) - special.kv(3, xx)) / k2)
    return float(
        -1.0
        - 0.5 / xx
        - 15.0 / (8.0 * xx**2)
        + 15.0 / (8.0 * xx**3)
        - 135.0 / (128.0 * xx**4)
        - 45.0 / (32.0 * xx**5)
        + 7425.0 / (1024.0 * xx**6)
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _file_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:32]
    except OSError:
        return "unreadable"


def _dof_digest(dof: Any) -> str:
    digest = hashlib.sha256()
    for name in ("t", "heff", "sqrtgeff", "sqrtgstar", "dheff_dt"):
        value = np.ascontiguousarray(np.asarray(getattr(dof, name), dtype=np.float64))
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()[:32]


def _z4_rate_cache_dir(settings: dict[str, Any]) -> Path | None:
    if not _as_bool(settings.get("z4RateCache"), True):
        return None
    configured = settings.get("z4RateCacheDir")
    if configured is None or str(configured).strip() == "":
        return Path(__file__).resolve().parents[2] / "cache" / "z4_rates"
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def _z4_rate_cache_fingerprint(
    physics: "Z4Physics",
    settings: dict[str, Any],
    backend: str,
    native: Any,
) -> str:
    native_path = getattr(native, "path", None)
    native_path = Path(native_path) if native_path else None
    source_path = Path(__file__).resolve()
    cpp_path = source_path.parents[1] / "native" / "z4_native.cpp"
    payload = {
        "schema": _Z4_RATE_CACHE_SCHEMA,
        "backend": backend,
        "params": physics.params,
        "settings": settings,
        "dof_sha256": _dof_digest(physics.dof),
        "python_model_sha256": _file_digest(source_path),
        "native_cpp_sha256": _file_digest(cpp_path),
        "native_dll_sha256": _file_digest(native_path),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _load_z4_rate_cache(
    cache_path: Path,
    fingerprint: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    metadata_path = cache_path.with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != _Z4_RATE_CACHE_SCHEMA:
            return None
        if metadata.get("fingerprint") != fingerprint:
            return None
        with np.load(cache_path, allow_pickle=False) as data:
            arrays = {
                name: np.asarray(data[name], dtype=np.float64)
                for name in _Z4_RATE_NAMES
            }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    for name, table in arrays.items():
        if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2:
            return None
        if not np.all(np.isfinite(table)) or np.any(table[:, 0] <= 0.0) or np.any(table[:, 1] <= 0.0):
            return None
    return arrays, metadata


def _write_z4_rate_cache(
    cache_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
    temporary_json = cache_path.with_suffix(".tmp.json")
    np.savez_compressed(
        temporary_npz,
        **{name: np.asarray(arrays[name], dtype=np.float64) for name in _Z4_RATE_NAMES},
    )
    temporary_npz.replace(cache_path)
    temporary_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_json.replace(cache_path.with_suffix(".json"))


@dataclass
class PowerTailLogLogInterp:
    """DRAKE's log-log interpolation with power-law endpoint tails."""

    table: np.ndarray
    core: Any
    left_slope: float
    right_slope: float
    log_x: np.ndarray
    log_y: np.ndarray

    @classmethod
    def from_table(cls, table: np.ndarray) -> "PowerTailLogLogInterp":
        data = np.asarray(table, dtype=float)
        data = data[np.argsort(data[:, 0])]
        data[:, 1] = np.maximum(data[:, 1], MIN_NUMBER)
        lx = np.log(data[:, 0])
        ly = np.log(data[:, 1])
        # The active MMA ratesl.wl card uses cubic interpolation in log-log
        # space for all VLL thermal rates.  Keep that rule here so the two
        # large Y_{B-L} source terms retain the same cancellation structure.
        core = interpolate.CubicSpline(lx, ly, extrapolate=True) if len(data) >= 4 else interpolate.interp1d(lx, ly, kind="linear", fill_value="extrapolate")
        left = float((ly[1] - ly[0]) / max(lx[1] - lx[0], 1.0e-300))
        right = float((ly[-1] - ly[-2]) / max(lx[-1] - lx[-2], 1.0e-300))
        return cls(data, core, left, right, lx, ly)

    def __call__(self, x: float) -> float:
        xx = max(float(x), 1.0e-300)
        lx = math.log(xx)
        lxt = self.log_x
        lyt = self.log_y
        if lx < lxt[0]:
            ly = lyt[0] + self.left_slope * (lx - lxt[0])
        elif lx > lxt[-1]:
            ly = lyt[-1] + self.right_slope * (lx - lxt[-1])
        else:
            ly = float(self.core(lx))
        return _finite_positive(_safe_exp(ly))

    def derivative(self, x: float) -> float:
        """Derivative with respect to the physical ``x`` variable."""

        xx = max(float(x), 1.0e-300)
        lx = math.log(xx)
        lxt = self.log_x
        if lx < lxt[0]:
            slope = self.left_slope
        elif lx > lxt[-1]:
            slope = self.right_slope
        else:
            derivative = getattr(self.core, "derivative", None)
            slope = float(derivative()(lx)) if callable(derivative) else self.right_slope
        return float(self(xx) * slope / xx)


def _tabulate_1d_loglog_batch(
    batch_func: Callable[[np.ndarray], np.ndarray],
    xmin: float,
    xmax: float,
    nx: int,
    iacc: float,
    imax: int,
) -> np.ndarray:
    """The active ``tabulate_1d_loglog`` grid with vector-valued evaluations.

    The grid/refinement rule is kept in Python so interpolation and endpoint
    behavior stay unchanged.  Only the expensive function evaluations are
    handed to the native backend in batches.
    """

    nxx = max(4, int(nx))
    tx1 = [xmin / 5.0 * ((1000.0 / (xmin / 5.0)) ** ((i - 1) / (nxx - 1))) for i in range(1, nxx + 1)]
    tx2 = [1000.0 * ((xmax / 1000.0) ** (i / 10.0)) for i in range(2, 11)]
    tx = np.array(sorted(set(tx1 + tx2)), dtype=float)

    def evaluate(xvals: np.ndarray) -> np.ndarray:
        values = np.asarray(batch_func(np.asarray(xvals, dtype=float)), dtype=float).reshape(-1)
        if values.size != len(xvals):
            raise ValueError("native Z4 batch evaluator returned the wrong number of values")
        return np.maximum(values, MIN_NUMBER)

    fvals = evaluate(tx)
    table = np.column_stack((np.log(tx), np.log(fvals)))
    step = 0
    add: list[float] = [1.0]
    while step == 0 or (len(add) != 0 and step < max(0, int(imax))):
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
        refine = np.abs(icubic(midpoints) - ilinear(midpoints)) > float(iacc) * widths
        add = sorted(set(midpoints[refine].tolist()))
        if add:
            x_add = np.exp(np.asarray(add, dtype=float))
            y_add = evaluate(x_add)
            add_table = np.column_stack((np.log(x_add), np.log(y_add)))
            table = np.vstack((table, add_table))
            table = table[np.argsort(table[:, 0])]

    while len(table) >= 2 and abs(table[-1, 1] - table[-2, 1]) > 0.01 * abs(table[-1, 1]) and table[-1, 1] > math.log(MIN_NUMBER):
        new_lx = table[-1, 0] + 1.0
        new_x = math.exp(new_lx)
        new_y = float(evaluate(np.array([new_x], dtype=float))[0])
        table = np.vstack((table, np.array([[new_lx, math.log(new_y)]], dtype=float)))
    return np.column_stack((np.exp(table[:, 0]), np.exp(table[:, 1])))


@dataclass
class Z4RateSet:
    """Thermal tables created by the active ``ratesl.wl`` workflow."""

    physics: "Z4Physics"
    settings: dict[str, Any]
    svx: Callable[[float], float] | None = None
    sv2x: Callable[[float], float] | None = None
    svxl: Callable[[float], float] | None = None
    svxl1: Callable[[float], float] | None = None
    svxl2: Callable[[float], float] | None = None
    gs: Callable[[float], float] | None = None
    tables: dict[str, np.ndarray] | None = None
    timings: dict[str, Any] | None = None

    def _make_table(self, func: Callable[[float], float], nx: int, iacc: float, imax: int) -> np.ndarray:
        p = self.physics
        return tabulate_1d_loglog(
            lambda x: _finite_positive(func(float(x))),
            float(self.settings["xmin"]),
            float(self.settings["xmax"]),
            max(4, int(nx)),
            float(iacc),
            max(0, int(imax)),
            max(1, int(self.settings.get("thermalWorkers", 1))),
        )

    def build(self) -> "Z4RateSet":
        import time

        p = self.physics
        nx = int(self.settings.get("Nx", 50))
        nx1 = int(self.settings.get("Nx1", 20))
        iacc = float(self.settings.get("iacc", 0.1))
        iacc1 = float(self.settings.get("iacc1", 0.6))
        imax = int(self.settings.get("imax", 5))
        imax1 = int(self.settings.get("imax1", 5))
        t0 = time.perf_counter()
        tables: dict[str, np.ndarray] = {}
        funcs: dict[str, Callable[[float], float]] = {}

        native = None
        native_cache: dict[float, np.ndarray] = {}
        native_batch = None
        try:
            from ..native.z4_native import get_backend

            native = get_backend()
            if native is not None:
                native_params = np.array([p.r, p.m_dm, p.delta, p.y, p.lam, p.m_l], dtype=float)

                def cached_native_batch(xvals: np.ndarray) -> np.ndarray:
                    values = np.asarray(xvals, dtype=float).reshape(-1)
                    missing = [float(x) for x in values if float(x) not in native_cache]
                    if missing:
                        rows = native.thermal_rates(native_params, np.asarray(missing, dtype=float), int(self.settings.get("pg", 3)))
                        for x, row in zip(missing, rows):
                            native_cache[x] = np.asarray(row, dtype=float)
                    return np.vstack([native_cache[float(x)] for x in values])

                native_batch = cached_native_batch
        except Exception as exc:  # pragma: no cover - optional backend
            warnings.warn(f"Z4 native backend unavailable; using Python rates: {exc}", RuntimeWarning)
            native = None

        definitions: list[tuple[str, Callable[[float], float], int, float, int, bool]] = [
            ("svx", p.thermal_svx, nx, iacc, imax, False),
            ("svxl", p.thermal_svxl, nx, iacc, imax, False),
            ("svxl1", p.thermal_svxl1, nx1, iacc1, imax1, True),
            ("svxl2", p.thermal_svxl2, nx1, iacc1, imax1, True),
            ("sv2x", p.thermal_sv2x, nx, iacc, imax, False),
            ("gs", p.thermal_gs, nx, iacc, imax, False),
        ]
        native_index = {"svx": 0, "sv2x": 1, "svxl": 2, "svxl1": 3, "svxl2": 4, "gs": 5}
        backend_name = "cpp" if native is not None else "python"
        cache_dir = _z4_rate_cache_dir(self.settings)
        cache_path: Path | None = None
        cache_fingerprint: str | None = None
        cache_hit = False
        cached_metadata: dict[str, Any] = {}
        if cache_dir is not None:
            cache_fingerprint = _z4_rate_cache_fingerprint(p, self.settings, backend_name, native)
            cache_path = cache_dir / f"z4_{cache_fingerprint}.npz"
            cached = _load_z4_rate_cache(cache_path, cache_fingerprint)
            if cached is not None:
                tables, cached_metadata = cached
                cache_hit = True

        if not cache_hit:
            for name, func, nxi, acc, imi, _power_tail in definitions:
                if native_batch is not None:
                    idx = native_index[name]
                    tab = _tabulate_1d_loglog_batch(
                        lambda xvals, idx=idx: native_batch(xvals)[:, idx],
                        self.settings["xmin"],
                        self.settings["xmax"],
                        nxi,
                        acc,
                        imi,
                    )
                else:
                    tab = self._make_table(func, nxi, acc, imi)
                tables[name] = tab
            if cache_path is not None and cache_fingerprint is not None:
                _write_z4_rate_cache(
                    cache_path,
                    tables,
                    {
                        "schema": _Z4_RATE_CACHE_SCHEMA,
                        "fingerprint": cache_fingerprint,
                        "backend": backend_name,
                        "native_unique_x": int(len(native_cache)),
                        "table_sizes": {name: int(len(table)) for name, table in tables.items()},
                    },
                )
        for name, _func, _nxi, _acc, _imi, power_tail in definitions:
            tab = tables[name]
            funcs[name] = PowerTailLogLogInterp.from_table(tab) if power_tail else _ConstantTailInterp.from_table(tab)

        self.svx = funcs["svx"]
        self.svxl = funcs["svxl"]
        self.svxl1 = funcs["svxl1"]
        self.svxl2 = funcs["svxl2"]
        self.sv2x = funcs["sv2x"]
        self.gs = funcs["gs"]
        self.tables = tables
        self.timings = {
            "build_rates": time.perf_counter() - t0,
            "backend": backend_name,
            "native_unique_x": float(
                len(native_cache)
                if native is not None and not cache_hit
                else cached_metadata.get("native_unique_x", 0.0)
            ),
            "cache_enabled": cache_dir is not None,
            "cache_hit": cache_hit,
            "cache_path": str(cache_path) if cache_path is not None else None,
            "cache_fingerprint": cache_fingerprint,
        }
        return self


@dataclass
class _ConstantTailInterp:
    table: np.ndarray
    core: Any
    log_x: np.ndarray

    @classmethod
    def from_table(cls, table: np.ndarray) -> "_ConstantTailInterp":
        data = np.asarray(table, dtype=float)
        data = data[np.argsort(data[:, 0])]
        data[:, 1] = np.maximum(data[:, 1], MIN_NUMBER)
        lx = np.log(data[:, 0])
        ly = np.log(data[:, 1])
        # Match the VLL/MMA ratesl.wl interpolation order for this model.
        core = interpolate.CubicSpline(lx, ly, extrapolate=True) if len(data) >= 4 else interpolate.interp1d(lx, ly, kind="linear", fill_value="extrapolate")
        return cls(data, core, lx)

    def __call__(self, x: float) -> float:
        xx = float(x)
        lx = math.log(max(xx, 1.0e-300))
        lxt = self.log_x
        if lx <= lxt[0]:
            return _finite_positive(self.table[0, 1])
        if lx >= lxt[-1]:
            return _finite_positive(self.table[-1, 1])
        return _finite_positive(_safe_exp(float(self.core(lx))))

    def derivative(self, x: float) -> float:
        """Derivative with respect to the physical ``x`` variable."""

        xx = max(float(x), 1.0e-300)
        lx = math.log(xx)
        lxt = self.log_x
        if lx <= lxt[0] or lx >= lxt[-1]:
            return 0.0
        derivative = getattr(self.core, "derivative", None)
        slope = float(derivative()(lx)) if callable(derivative) else 0.0
        return float(self(xx) * slope / xx)


class Z4Physics:
    """Active VLL/Z4 formulae from the current MMA model card."""

    name = "Z4"

    def __init__(self, params: dict[str, Any], settings: dict[str, Any], dof: Any):
        self.params = {key: float(value) for key, value in params.items() if _is_number(value)}
        self.settings = dict(settings)
        self.dof = dof
        self._dof_t = tuple(float(value) for value in np.asarray(dof.t, dtype=float))
        self._dof_heff = tuple(float(value) for value in np.asarray(dof.heff, dtype=float))
        self._dof_sqrtgeff = tuple(float(value) for value in np.asarray(dof.sqrtgeff, dtype=float))
        self._dof_dheff_dt = tuple(float(value) for value in np.asarray(dof.dheff_dt, dtype=float))
        # ``neq`` is the equilibrium density of one internal state.  The
        # active Dirac-DM convention tracks the physical total X + Xbar
        # density through ``nchi_eq = gDM * neq``.
        self.g_dm = float(self.params.get("gDM", 2.0))
        self.g_l = 1.0
        self.g_psi = 2.0
        self.r = self._required("r")
        self.m_dm = self._required("mDM")
        self.delta = self._required("delta")
        self.y = self._required("y")
        self.lam = self._required("lam")
        self.m_l = self._required("mL")
        self.eps1 = self._required("eps1")
        self.m_s2 = float(self.params.get("mS2", 10.0 * 4.0 * self.m_dm**2 / (1.0 + self.delta)))
        self.gammas = self._width_visible()
        self.gammap = self._width_parent()
        # The active source multiplies both branching fractions by exactly 0.
        self.brc = 0.0
        self.br_l = 0.0
        self.delta_indicator = 1.0 if self.delta > 0.51 else 0.0

    def _required(self, key: str) -> float:
        if key not in self.params:
            raise ValueError(f"missing Z4/VLL parameter: {key}")
        return float(self.params[key])

    def temperature(self, x: float) -> float:
        return self.m_dm / max(float(x), 1.0e-300)

    def _dof_interp_scalar(self, temp: float, values: tuple[float, ...]) -> float:
        """Scalar equivalent of DofTable._interp without NumPy call overhead."""

        tt = float(temp)
        grid = self._dof_t
        if tt <= grid[0]:
            return values[0]
        if tt >= grid[-1]:
            return values[-1]
        index = bisect_right(grid, tt) - 1
        fraction = (tt - grid[index]) / (grid[index + 1] - grid[index])
        return values[index] + fraction * (values[index + 1] - values[index])

    @lru_cache(maxsize=4096)
    def entropy(self, x: float) -> float:
        temp = self.temperature(x)
        return self._dof_interp_scalar(temp, self._dof_heff) * (2.0 * _PI**2) / 45.0 * temp**3

    @lru_cache(maxsize=4096)
    def hubble(self, x: float) -> float:
        temp = self.temperature(x)
        sqrt_geff = self._dof_interp_scalar(temp, self._dof_sqrtgeff)
        return _PI / (3.0 * math.sqrt(10.0)) * sqrt_geff * temp**2 / M_PL_REDUCED

    @lru_cache(maxsize=4096)
    def gtilde(self, x: float) -> float:
        temp = self.temperature(x)
        heff = self._dof_interp_scalar(temp, self._dof_heff)
        dheff_dt = self._dof_interp_scalar(temp, self._dof_dheff_dt)
        return temp / 3.0 * dheff_dt / heff

    @lru_cache(maxsize=4096)
    def neq(self, mass: float, z: float) -> float:
        zz = max(float(z), 1.0e-12)
        return _finite_positive(float(mass) ** 3 * float(bessel_k2_approx(zz)) / (2.0 * _PI**2 * zz))

    @lru_cache(maxsize=4096)
    def nchi_eq(self, x: float) -> float:
        """Total equilibrium DM density, including X and Xbar."""

        return _finite_positive(self.g_dm * self.neq(self.m_dm, x))

    @lru_cache(maxsize=4096)
    def yeq(self, x: float) -> float:
        return _finite_positive(self.m_dm**2 / max(float(x), 1.0e-300) * self.entropy(x) ** (-2.0 / 3.0))

    @lru_cache(maxsize=4096)
    def Yeq(self, x: float) -> float:
        # Yeq is obtained from the already total density nchi_eq; do not
        # multiply Yeq by gDM again at the caller.
        return _finite_positive(self.nchi_eq(x) / self.entropy(x))

    @lru_cache(maxsize=4096)
    def YLeq(self, x: float) -> float:
        return _finite_positive(self.g_l * self.neq(self.m_l, x * self.m_l / self.m_dm) / self.entropy(x))

    @lru_cache(maxsize=4096)
    def Ypeq(self, x: float) -> float:
        return _finite_positive(self.g_psi * self.neq(self.r * self.m_dm, x * self.r) / self.entropy(x))

    @lru_cache(maxsize=4096)
    def Yseq(self, x: float) -> float:
        mass = 2.0 * self.m_dm / math.sqrt(1.0 + self.delta)
        z = x * 2.0 / math.sqrt(1.0 + self.delta)
        return _finite_positive((1.0 / self.g_dm) * self.neq(mass, z) / self.entropy(x))

    @lru_cache(maxsize=4096)
    def Yg(self, x: float) -> float:
        geff = max(self._dof_interp_scalar(self.temperature(x), self._dof_sqrtgeff), 1.0e-12)
        return _finite_positive(45.0 * 1.202 / (4.0 * _PI**4 * geff**2))

    @lru_cache(maxsize=4096)
    def spectator_ratio(self, x: float) -> float:
        """Return the dynamic equilibrium ratio used by spectator relations."""

        return float((self.g_psi * self.neq(self.r * self.m_dm, x * self.r)) / max(
            self.g_l * self.neq(self.m_l, x * self.m_l / self.m_dm), _SAFE_FLOOR
        ))

    @lru_cache(maxsize=4096)
    def ksi(self, x: float) -> float:
        ratio = self.spectator_ratio(x)
        return (16.0 + 12.0 * ratio) / max(3.0 + 12.0 * ratio, _SAFE_FLOOR)

    @lru_cache(maxsize=4096)
    def eta(self, x: float) -> float:
        """Return the unabsorbed spectator coefficient from the manuscript."""

        ratio = self.spectator_ratio(x)
        return (2.0 * (7.0 + 28.0 * ratio)) / max(79.0 + 355.0 * ratio, _SAFE_FLOOR)

    @lru_cache(maxsize=4096)
    def ybl_sm(self, x: float, y_delta_l: float) -> float:
        """Reconstruct the visible-sector B-L asymmetry dynamically."""

        ratio = self.spectator_ratio(x)
        conversion = -(79.0 / 102.0) * (51.0 + 243.0 * ratio) / max(7.0 + 28.0 * ratio, _SAFE_FLOOR)
        return float(conversion * float(y_delta_l))

    def _width_visible(self) -> float:
        phase = 4.0 * self.m_dm**2 - (self.m_l - self.m_dm * self.r) ** 2 * (1.0 + self.delta)
        rad = (1.0 - (self.m_l - self.m_dm * self.r) ** 2 * (1.0 + self.delta) / (4.0 * self.m_dm**2))
        rad *= 1.0 - (self.m_l + self.m_dm * self.r) ** 2 * (1.0 + self.delta) / (4.0 * self.m_dm**2)
        g_brc = 0.0
        if -1.0 < self.delta < 0.0:
            g_brc = self.m_dm * 16.0 * self.y**2 * math.sqrt(max(-self.delta, 0.0)) / (32.0 * _PI * math.sqrt(1.0 + self.delta))
        g_br_l = 0.0
        if phase > 0.0 and rad > 0.0:
            g_br_l = 4.0 * self.lam**2 * phase * math.sqrt(rad) / (32.0 * self.m_dm * _PI * math.sqrt(1.0 + self.delta))
        return _finite_positive(g_brc + g_br_l)

    def _width_parent(self) -> float:
        phase = 4.0 * self.m_dm**2 - (self.m_l - self.m_dm * self.r) ** 2 * (1.0 + self.delta)
        rad = (1.0 - (self.m_l - self.m_dm * self.r) ** 2 * (1.0 + self.delta) / (4.0 * self.m_dm**2))
        rad *= 1.0 - (self.m_l + self.m_dm * self.r) ** 2 * (1.0 + self.delta) / (4.0 * self.m_dm**2)
        if phase <= 0.0 or rad <= 0.0:
            return MIN_NUMBER
        return _finite_positive(2.0 * self.y**2 * phase * math.sqrt(rad) / (32.0 * self.m_dm * _PI))

    @staticmethod
    def _kallen(s: float, a: float, b: float) -> float:
        return s**2 + a**2 + b**2 - 2.0 * (s * a + s * b + a * b)

    def sigma_s(self, s: float) -> float:
        """One fixed SU(2)_L-component ``sv[ss]`` from the active VLL card."""

        ss = float(s)
        m_psi = self.r * self.m_dm
        mphi2 = 4.0 * self.m_dm**2 / max(1.0 + self.delta, _SAFE_FLOOR)
        a = (m_psi - self.m_l) ** 2
        b = (m_psi + self.m_l) ** 2
        rad = (ss - a) * (ss - b)
        if ss <= 4.0 * self.m_dm**2 or ss <= b or rad <= 0.0:
            return MIN_NUMBER
        den_res = (ss - mphi2) ** 2 + mphi2 * self.gammas**2
        den = 32.0 * _PI * (ss - 2.0 * self.m_dm**2) * den_res
        val = self.y**2 * self.lam**2 * (ss - m_psi**2 - self.m_l**2) * math.sqrt(rad) / den
        return _finite_positive(val)

    def sigma_s_no_sigma3(self, s: float) -> float:
        return self.sigma_s(s)

    def sigma_l(self, s: float) -> float:
        """Active VLL ``svl``: one-component s+t washout with interference."""

        ss = float(s)
        m_psi = self.r * self.m_dm
        mpsi2 = m_psi**2
        mphi2 = 4.0 * self.m_dm**2 / max(1.0 + self.delta, _SAFE_FLOOR)
        if ss <= mpsi2:
            return MIN_NUMBER
        d_s = 1.0 / max((ss - mphi2) ** 2 + mphi2 * self.gammas**2, _SAFE_FLOOR)
        t_min = 2.0 * mpsi2 - ss
        t_max = mpsi2**2 / ss
        if not (t_max > t_min):
            return MIN_NUMBER
        # The active VLL card returns Indeterminate when the bare t-channel
        # pole enters the physical interval.  RelicFlow keeps its finite
        # positive-rate contract and maps that point to the safe floor.
        if t_min <= mphi2 <= t_max:
            return MIN_NUMBER

        width_t = t_max - t_min
        log_ratio = math.log(max(abs(t_max - mphi2), _SAFE_FLOOR) / max(abs(t_min - mphi2), _SAFE_FLOOR))
        q = mphi2 - mpsi2
        integral_t2 = width_t + 2.0 * q * log_ratio - q**2 * (
            1.0 / (t_max - mphi2) - 1.0 / (t_min - mphi2)
        )
        integral_interference = 2.0 * (ss - mphi2) * d_s * (
            ss * width_t - (mpsi2**2 - ss * mphi2) * log_ratio
        )
        integral = 0.5 * self.lam**4 * (
            (ss - mpsi2) ** 2 * d_s * width_t + integral_t2 + integral_interference
        )
        return _finite_positive(integral / (16.0 * _PI * (ss - mpsi2) ** 2))

    def sigma1(self, s: float) -> float:
        """Active VLL ``Sigma1`` conversion rate, with the note's mL=0 card."""

        m2 = self.r * self.m_dm
        m4 = 0.0
        ss = float(s)
        pin2 = self._kallen(ss, self.m_dm**2, m2**2)
        pout2 = self._kallen(ss, self.m_dm**2, m4**2)
        if ss <= (self.m_dm + m2) ** 2 or pin2 <= 0.0 or pout2 <= 0.0:
            return MIN_NUMBER
        pin = math.sqrt(pin2) / (2.0 * math.sqrt(ss))
        pout = math.sqrt(pout2) / (2.0 * math.sqrt(ss))
        ex_in = (ss + self.m_dm**2 - m2**2) / (2.0 * math.sqrt(ss))
        ex_out = (ss + self.m_dm**2 - m4**2) / (2.0 * math.sqrt(ss))

        def integrand(costh: float) -> float:
            t = 2.0 * self.m_dm**2 - 2.0 * ex_in * ex_out + 2.0 * pin * pout * costh
            return self.y**2 * self.lam**2 * (4.0 * self.m_dm**2 - t) * (m2**2 - t) / max(
                (t - 4.0 * self.m_dm**2 / max(1.0 + self.delta, _SAFE_FLOOR)) ** 2,
                _SAFE_FLOOR,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            avg = 0.5 * float(integrate.quad(integrand, -1.0, 1.0, epsrel=1.0e-2, epsabs=0.0, limit=8)[0])
        sigma = (1.0 / (16.0 * _PI * ss)) * (pout / pin) * avg
        v_rel = math.sqrt(max(pin2, 0.0)) / max(ss - self.m_dm**2 - m2**2, _SAFE_FLOOR)
        return _finite_positive(sigma * v_rel)

    def sigma1_massless_bath(self, s: float) -> float:
        """Sigma1 kernel for the broken-phase light channel.

        This is the same conversion matrix element used by ``Sigma1`` with
        both the incoming and outgoing bath fermions taken massless.  It is
        used only by the optional Psi--n_R diagnostic, not by the main rate
        tables or the nBE/cBE evolution.
        """

        ss = float(s)
        m_dm = self.m_dm
        mphi2 = 4.0 * m_dm**2 / max(1.0 + self.delta, _SAFE_FLOOR)
        pin2 = self._kallen(ss, m_dm**2, 0.0)
        pout2 = pin2
        if ss <= m_dm**2 or pin2 <= 0.0 or pout2 <= 0.0:
            return MIN_NUMBER
        pin = math.sqrt(pin2) / (2.0 * math.sqrt(ss))
        pout = pin
        ex_in = (ss + m_dm**2) / (2.0 * math.sqrt(ss))
        ex_out = ex_in

        def integrand(costh: float) -> float:
            t = 2.0 * m_dm**2 - 2.0 * ex_in * ex_out + 2.0 * pin * pout * costh
            return self.y**2 * self.lam**2 * (4.0 * m_dm**2 - t) * (-t) / max(
                (t - mphi2) ** 2,
                _SAFE_FLOOR,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            avg = 0.5 * float(
                integrate.quad(
                    integrand,
                    -1.0,
                    1.0,
                    epsrel=1.0e-2,
                    epsabs=0.0,
                    limit=8,
                )[0]
            )
        sigma = (1.0 / (16.0 * _PI * ss)) * (pout / pin) * avg
        v_rel = math.sqrt(max(pin2, 0.0)) / max(ss - m_dm**2, _SAFE_FLOOR)
        return _finite_positive(sigma * v_rel)

    @lru_cache(maxsize=128)
    def thermal_svxl1_massless_bath(self, x: float) -> float:
        """Thermal ``Sigma1`` rate for a massless bath in the light channel.

        The integral is evaluated only at the diagnostic point (normally
        ``x_kd``).  Keeping it out of ``build_rate_set`` avoids adding a new
        table to every BP and makes the extra consistency checks cheap.
        """

        xx = max(float(x), 1.0e-8)
        temp = self.m_dm / xx
        q0 = self.m_dm**2
        threshold = q0
        near = threshold * (1.0 + 20.0 / xx)
        k2_dm = max(float(special.kv(2, xx)), _SAFE_FLOOR)
        # lim_{m->0} m^2 K_2(m/T) = 2 T^2.
        den = 8.0 * temp * self.m_dm**2 * k2_dm * (2.0 * temp**2)

        def kernel(ss: float) -> float:
            if xx < X_BEXP:
                z = math.sqrt(max(ss, 0.0)) / temp
                pref = (
                    (ss - q0)
                    * math.sqrt(max(ss - 2.0 * q0, 0.0))
                    * float(bessel_k1_approx(z))
                    / max(den, _SAFE_FLOOR)
                )
            else:
                st = ss / max(2.0 * q0, _SAFE_FLOOR)
                if st <= 1.0:
                    return 0.0
                poly = -15.0 + 24.0 * math.sqrt(st) * (-15.0 + 4.0 * X_BEXP)
                poly += 16.0 * st * (285.0 - 120.0 * X_BEXP + 32.0 * X_BEXP**2)
                pref = (
                    1.0
                    / max(2.0 * q0, _SAFE_FLOOR)
                    * _safe_exp(-2.0 * (-1.0 + math.sqrt(st)) * X_BEXP)
                    * math.sqrt(st - 1.0)
                    * (-1.0 + 2.0 * st)
                    * math.sqrt(1.0 / X_BEXP)
                    * poly
                    / (256.0 * math.sqrt(_PI) * st**1.25)
                )
            return pref * self.sigma1_massless_bath(ss)

        return self._integrate_segments(
            kernel,
            [threshold, near],
            10.0 ** (-float(self.settings.get("pg", 3))),
            limit=80,
        )

    def sigma2(self, s: float) -> float:
        """Active VLL ``Sigma2`` Psi Psi -> L L rate with t+u interference."""

        m2 = self.r * self.m_dm
        ss = float(s)
        pin2 = self._kallen(ss, m2**2, m2**2)
        if ss <= 4.0 * m2**2 or pin2 <= 0.0:
            return MIN_NUMBER
        mphi2 = 4.0 * self.m_dm**2 / max(1.0 + self.delta, _SAFE_FLOOR)
        t_min = m2**2 - ss / 2.0 - math.sqrt(max(ss * (ss - 4.0 * m2**2), 0.0)) / 2.0
        t_max = m2**2 - ss / 2.0 + math.sqrt(max(ss * (ss - 4.0 * m2**2), 0.0)) / 2.0

        def integrand(costh: float) -> float:
            t = t_min + (t_max - t_min) * (1.0 + costh) / 2.0
            u = 2.0 * m2**2 - ss - t
            den_t = max((t - mphi2) ** 2, _SAFE_FLOOR)
            den_u = max((u - mphi2) ** 2, _SAFE_FLOOR)
            den_tu = (t - mphi2) * (u - mphi2)
            interference = 0.0 if abs(den_tu) < _SAFE_FLOOR else 2.0 * (
                m2**2 * ss - (m2**2 - t) * (m2**2 - u)
            ) / den_tu
            return self.lam**4 / 4.0 * (
                (m2**2 - t) ** 2 / den_t
                + (m2**2 - u) ** 2 / den_u
                - interference
            ) * (t_max - t_min) / 2.0

        sigma = float(integrate.quad(integrand, -1.0, 1.0, epsrel=1.0e-2, epsabs=0.0, limit=8)[0]) / (
            32.0 * _PI * ss * (ss - 4.0 * m2**2)
        )
        v_rel = math.sqrt(max(self._kallen(ss, m2**2, m2**2), 0.0)) / max(ss - 2.0 * m2**2, _SAFE_FLOOR)
        return _finite_positive(sigma * v_rel)

    def _limits(
        self,
        x: float,
        base: float,
        resonance: float | None = None,
        extended_base: float | None = None,
    ) -> list[float]:
        near = base * (1.0 + 20.0 / max(x, 1.0e-12))
        if resonance is None:
            return [base, near]
        if resonance < near:
            return [base, resonance, near]
        upper_base = base if extended_base is None else float(extended_base)
        return [base, near, min(upper_base * (1.0 + 30.0 / max(x, 1.0e-12)), 2.0 * resonance)]

    def _integrate_segments(self, func: Callable[[float], float], limits: list[float], epsrel: float, limit: int = 80) -> float:
        total = 0.0
        for lo, hi in zip(limits[:-1], limits[1:]):
            if hi <= lo:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", integrate.IntegrationWarning)
                total += float(integrate.quad(func, lo, hi, epsrel=epsrel, epsabs=0.0, limit=limit)[0])
        return _finite_positive(total)

    def _thermal_identical(self, x: float, sigma: Callable[[float], float]) -> float:
        xx = max(float(x), 1.0e-8)
        temp = self.m_dm / xx
        limits = self._limits(xx, 4.0 * self.m_dm**2, 4.0 * self.m_dm**2 / (1.0 + self.delta) if self.delta < 0.0 else None)

        def kernel(ss: float) -> float:
            if xx < X_BEXP:
                z = math.sqrt(max(ss, 0.0)) / temp
                den = 8.0 * temp * self.m_dm**4 * max(float(special.kv(2, xx)) ** 2, _SAFE_FLOOR)
                pref = (ss - 2.0 * self.m_dm**2) * math.sqrt(max(ss - 4.0 * self.m_dm**2, 0.0)) * float(bessel_k1_approx(z)) / den
            else:
                st = ss / (4.0 * self.m_dm**2)
                if st <= 1.0:
                    return 0.0
                pref = (
                    1.0 / (4.0 * self.m_dm**2)
                    * _safe_exp(-2.0 * (-1.0 + math.sqrt(st)) * xx)
                    * math.sqrt(st - 1.0)
                    * (-1.0 + 2.0 * st)
                    * math.sqrt(1.0 / xx)
                    * (-15.0 + 24.0 * math.sqrt(st) * (-15.0 + 4.0 * xx) + 16.0 * st * (285.0 - 120.0 * xx + 32.0 * xx**2))
                    / (256.0 * math.sqrt(_PI) * st**1.25)
                )
            return pref * sigma(ss)

        return self._integrate_segments(kernel, limits, 10.0 ** (-float(self.settings.get("pg", 3))))

    def _thermal_pair(self, x: float, sigma: Callable[[float], float], kind: str) -> float:
        xx = max(float(x), 1.0e-8)
        temp = self.m_dm / xx
        if kind == "svxl":
            q0 = self.r**2 * self.m_dm**2 + self.m_l**2
            m1, m2 = self.m_l, self.r * self.m_dm
            high_norm = 2.0 * q0
        elif kind == "svxl1":
            q0 = self.r**2 * self.m_dm**2 + self.m_dm**2
            m1, m2 = self.m_dm, self.r * self.m_dm
            high_norm = 2.0 * q0
        else:
            q0 = 2.0 * self.r**2 * self.m_dm**2
            m1, m2 = self.r * self.m_dm, self.r * self.m_dm
            # This is the active thermalavgl.wl normalization, including its
            # asymmetric high-x denominator.
            high_norm = 2.0 * (self.r**2 * self.m_dm**2 + self.m_dm**2)
        # The active MMA thermalavgl.wl card uses ``resonancel`` for svxl:
        # include the pole when it lies above the svxl initial-state
        # threshold, including the BP01 delta > 0 case.  svxl1/svxl2 do not
        # carry this resonance breakpoint in the source card.
        resonance = None
        if kind == "svxl":
            candidate = 4.0 * self.m_dm**2 / (1.0 + self.delta)
            if candidate > 2.0 * q0:
                resonance = candidate
        # Only MMA's svxl card uses the physical 4*mDM^2 scale for this
        # extended endpoint.  Keep the default normalized endpoint for
        # svxl1/svxl2 and for the separate sv2x channel.
        extended_base = 4.0 * self.m_dm**2 if kind == "svxl" else None
        limits = self._limits(xx, 2.0 * q0, resonance, extended_base=extended_base)

        def kernel(ss: float) -> float:
            if xx < X_BEXP:
                k1 = float(bessel_k1_approx(math.sqrt(max(ss, 0.0)) / temp))
                k21 = max(float(special.kv(2, xx * m1 / self.m_dm)), _SAFE_FLOOR)
                k22 = max(float(special.kv(2, xx * m2 / self.m_dm)), _SAFE_FLOOR)
                den = 8.0 * temp * m1**2 * m2**2 * k21 * k22
                pref = (ss - q0) * math.sqrt(max(ss - 2.0 * q0, 0.0)) * k1 / den
            else:
                st = ss / max(high_norm, _SAFE_FLOOR)
                if st <= 1.0:
                    return 0.0
                poly = -15.0 + 24.0 * math.sqrt(st) * (-15.0 + 4.0 * X_BEXP) + 16.0 * st * (285.0 - 120.0 * X_BEXP + 32.0 * X_BEXP**2)
                pref = (
                    1.0 / max(high_norm, _SAFE_FLOOR)
                    * _safe_exp(-2.0 * (-1.0 + math.sqrt(st)) * X_BEXP)
                    * math.sqrt(st - 1.0)
                    * (-1.0 + 2.0 * st)
                    * math.sqrt(1.0 / X_BEXP)
                    * poly
                    / (256.0 * math.sqrt(_PI) * st**1.25)
                )
            return pref * sigma(ss)

        epsrel = 10.0 ** (-float(self.settings.get("pg", 3)))
        if resonance is not None and resonance > limits[0] and resonance < limits[-1]:
            # Resolve the narrow svxl Breit-Wigner peak with the same
            # width-scaled variable used by the native backend.  Splitting a
            # machine-precision integral at the pole can miss most of the
            # area and is not a reliable fallback for the inelastic channel.
            width_source = self.gammas if kind == "svxl" else self.gammap
            width = 2.0 * width_source * self.m_dm / math.sqrt(max(1.0 + self.delta, _SAFE_FLOOR))
            theta_a = math.atan((limits[0] - resonance) / width)
            theta_b = math.atan((limits[-1] - resonance) / width)

            def transformed(theta: float) -> float:
                cosine = math.cos(theta)
                ss = resonance + width * math.tan(theta)
                return kernel(ss) * width / max(cosine * cosine, _SAFE_FLOOR)

            nodes, weights = _gauss_legendre_nodes(256)
            midpoint = 0.5 * (theta_a + theta_b)
            half_width = 0.5 * (theta_b - theta_a)
            value = float(
                half_width
                * np.sum(
                    weights
                    * np.asarray(
                        [transformed(midpoint + half_width * node) for node in nodes],
                        dtype=float,
                    )
                )
            )
            return _finite_positive(value)
        return self._integrate_segments(kernel, limits, epsrel, limit=80)

    @lru_cache(maxsize=4096)
    def _thermal_svx_cached(self, x: float) -> float:
        return self._thermal_identical(float(x), self.sigma_s)

    @lru_cache(maxsize=4096)
    def _thermal_gs_cached(self, x: float) -> float:
        return self._thermal_identical(float(x), lambda _ss: self.gammas)

    @lru_cache(maxsize=4096)
    def _thermal_svxl_cached(self, x: float) -> float:
        return self._thermal_pair(float(x), self.sigma_l, "svxl")

    @lru_cache(maxsize=4096)
    def _thermal_svxl1_cached(self, x: float) -> float:
        return self._thermal_pair(float(x), self.sigma1, "svxl1")

    @lru_cache(maxsize=4096)
    def _thermal_svxl2_cached(self, x: float) -> float:
        return self._thermal_pair(float(x), self.sigma2, "svxl2")

    def thermal_svx(self, x: float) -> float:
        return self._thermal_svx_cached(float(x))

    def thermal_gs(self, x: float) -> float:
        return self._thermal_gs_cached(float(x))

    def thermal_svxl(self, x: float) -> float:
        return self._thermal_svxl_cached(float(x))

    def thermal_svxl1(self, x: float) -> float:
        return self._thermal_svxl1_cached(float(x))

    def thermal_svxl2(self, x: float) -> float:
        return self._thermal_svxl2_cached(float(x))

    def _thermal_sv2x_raw(self, x: float) -> float:
        xx = max(float(x), 1.0e-8)
        limits = self._limits(xx, 1.0, 4.0 * self.m_dm**2 / (1.0 + self.delta) / (4.0 * self.m_dm**2) if self.delta < 0.0 else None)
        epsrel = 10.0 ** (-float(self.settings.get("pg", 3)))
        if xx < X_BEXP:
            k2 = max(float(special.kv(2, xx)), _SAFE_FLOOR)
            expdbk2 = _bessel_k2_scaled(xx)

            def ff(ss: float) -> float:
                def integrand(ep: float) -> float:
                    root = math.sqrt(max((ss - 1.0) * (ep * ep - 1.0), 0.0))
                    a = math.sqrt(max(ss, 0.0)) * ep
                    if a <= root:
                        return 0.0
                    ratio = max((a - root) / max(a + root, _SAFE_FLOOR), _SAFE_FLOOR)
                    return (2.0 * math.sqrt(max(ss, 0.0)) * (2.0 * ss - 1.0) * xx**3 / 3.0) * expdbk2**2 * _safe_exp(-2.0 * math.sqrt(max(ss, 0.0)) * xx * ep + 2.0 * xx) * math.log(ratio)

                return float(integrate.quad(integrand, 1.0, 1.0 + 20.0 / xx, epsrel=epsrel, epsabs=0.0, limit=80)[0])

            def integrand(ss: float) -> float:
                first = (2.0 * math.sqrt(max(ss, 0.0)) * (2.0 * ss - 1.0) * xx**3) / (3.0 * k2**2)
                first *= math.sqrt(max((ss - 1.0) * ss, 0.0)) * float(bessel_k2_approx(2.0 * math.sqrt(max(ss, 0.0)) * xx)) / max(math.sqrt(max(ss, 0.0)) * xx, _SAFE_FLOOR)
                return (first + ff(ss)) * self.sigma_s(4.0 * self.m_dm**2 * ss)
        else:
            def integrand(ss: float) -> float:
                if ss <= 1.0:
                    return 0.0
                # Evaluate the DRAKE high-x polynomial in r=sqrt(s)-1.
                # Expanding the source expression directly in s causes severe
                # cancellation near s=1 (the integration support is O(1/x)).
                root_s = math.sqrt(ss)
                r = root_s - 1.0
                p4 = 512.0 * xx**2 - 1920.0 * xx + 4560.0
                p3 = 2048.0 * xx**2 - 6560.0 * xx + 14040.0
                p2 = 2560.0 * xx**2 - 6240.0 * xx + 11145.0
                p1 = 1024.0 * xx**2 - 832.0 * xx - 270.0
                p0 = 768.0 * xx - 2160.0
                poly = ((((p4 * r + p3) * r + p2) * r + p1) * r + p0)
                kernel = _safe_exp(-2.0 * r * xx) * math.sqrt((ss - 1.0) / ss) * (-1.0 + 2.0 * ss) * poly
                kernel /= 768.0 * math.sqrt(_PI) * ss**0.75 * math.sqrt(1.0 / xx)
                return self.sigma_s(4.0 * self.m_dm**2 * ss) * kernel
        return self._integrate_segments(integrand, limits, epsrel, limit=100)

    @lru_cache(maxsize=4096)
    def _thermal_sv2x_cached(self, x: float) -> float:
        return self._thermal_sv2x_raw(float(x))

    def thermal_sv2x(self, x: float) -> float:
        return self._thermal_sv2x_cached(float(x))

    @lru_cache(maxsize=4096)
    def rel_temp_correction(self, x: float) -> tuple[float, float]:
        bins = 220
        xx = max(float(x), 1.0e-12)
        try:
            from ..native.z4_native import get_backend

            native = get_backend()
            if native is not None:
                value = native.rel_temp(np.array([xx], dtype=float))[0]
                return float(value[0]), float(value[1])
        except Exception as exc:  # pragma: no cover - optional backend
            warnings.warn(f"Z4 native rel-temp correction unavailable; using Python: {exc}", RuntimeWarning)
        lmin = 1.0e-4 * math.sqrt(2.0 * xx)
        lmax = 8.0 * math.sqrt(2.0 * xx)
        ll = np.linspace(lmin, lmax, bins)
        dl = (lmax - lmin) / (bins - 1.0)
        fdm = np.exp(-(np.sqrt(xx * xx + ll * ll) - xx)) * _bessel_k2_scaled(xx)
        w34 = (1.0 / 3.0) * xx**-2 * dl * float(np.sum(ll**6 / (xx * xx + ll * ll) ** 1.5 * fdm))
        w34p = -(2.0 / xx + _der_bessel_k2_over_k2(xx)) * w34
        w34p -= (1.0 / xx) * dl * float(np.sum(ll**6 / (xx * xx + ll * ll) ** 2.5 * fdm))
        w34p -= (1.0 / (3.0 * xx)) * dl * float(np.sum(ll**6 / (xx * xx + ll * ll) ** 2.0 * fdm))
        return float(w34), float(w34p)

    def second_moment_scattering(self, xi: float, yeq: float, y: float, exp_dbk2: float, der_dbk2: float) -> tuple[float, float]:
        """Translation of VLL.wl BuildGetSecondMomentScat[150,10,24,25,10,50]."""

        dim = max(4, int(self.settings.get("fullCellDim", 150)))
        n_mu = max(1, int(self.settings.get("fullCellNMu", 10)))
        n_u = max(1, int(self.settings.get("fullCellNU", 24)))
        u_base = float(self.settings.get("fullCellUBase", 25.0))
        # The active VLL card binds GetSecondMomentScat to (...,25.,10.,50.,"C").
        u_tail = float(self.settings.get("fullCellUTail", 10.0))
        u_cap = float(self.settings.get("fullCellUCap", 50.0))

        # Optional scan-only late-time shortcut.  Once xDM is far beyond the
        # kinetic-decoupling region, the exact inelastic FullCel contribution
        # is numerically negligible compared with Hubble dilution.  Returning
        # zero here avoids constructing the late-time collision matrix while
        # retaining the complete FullCel integral at all earlier points.  This
        # is an explicit scattering-freezeout cutoff, not a Fokker--Planck
        # replacement; the default remains None (no cutoff).
        skip_xdm = self.settings.get("fullCellSkipXdm")
        try:
            skip_xdm_value = float(skip_xdm) if skip_xdm is not None else math.nan
        except (TypeError, ValueError):
            skip_xdm_value = math.nan
        y_safe = max(abs(float(y)), MIN_NUMBER)
        x_dm = float(xi) * float(yeq) / y_safe
        if math.isfinite(skip_xdm_value) and skip_xdm_value > 0.0 and x_dm >= skip_xdm_value:
            return 0.0, 0.0

        try:
            from ..native.z4_native import get_backend

            native = get_backend()
            if native is not None:
                native_params = np.array([self.r, self.m_dm, self.delta, self.y, self.lam, self.m_l], dtype=float)
                native_config = np.array([dim, n_mu, n_u, u_base, u_tail, u_cap], dtype=float)
                return native.full_cell(native_params, native_config, xi, yeq, y, exp_dbk2, der_dbk2)
        except Exception as exc:  # pragma: no cover - optional backend
            warnings.warn(f"Z4 native FullCel unavailable; using Python FullCel: {exc}", RuntimeWarning)

        yeqdy = float(yeq) / y_safe
        x_dm = float(xi) * yeqdy
        q_min = 1.0e-4 * math.sqrt(max(2.0 * xi / max(yeqdy, MIN_NUMBER), MIN_NUMBER))
        q_max = 8.0 * math.sqrt(max(2.0 * xi / max(yeqdy, MIN_NUMBER), MIN_NUMBER))
        q = np.linspace(q_min, q_max, dim)
        dq = (q_max - q_min) / max(dim - 1, 1)
        xq = np.sqrt(xi * xi + q * q)
        fdm = np.exp(-(yeqdy * xq - x_dm)) * float(exp_dbk2)
        mu_x, mu_w = np.polynomial.legendre.leggauss(n_mu)
        u01, uw01 = np.polynomial.legendre.leggauss(n_u)
        u01 = 0.5 * (u01 + 1.0)
        uw01 = 0.5 * uw01
        matrix = np.zeros((dim, dim), dtype=float)
        m_psi_hat = self.r * self.m_dm / (self.m_dm / xi)
        m_phi_hat = (2.0 * self.m_dm / math.sqrt(1.0 + self.delta)) / (self.m_dm / xi)
        m_l_hat = self.m_l / (self.m_dm / xi)
        g_y = self.y**2 * self.lam**2
        for i in range(dim - 1):
            qi = q[i]
            for j in range(i + 1, dim):
                qj = q[j]
                dx = xq[j] - xq[i]
                dx2 = dx * dx
                u_min2 = (m_psi_hat + dx) ** 2 - m_l_hat**2
                u_min = math.sqrt(max(u_min2, 0.0))
                u_max = min(max(u_min + u_tail, u_base), u_cap)
                if u_min >= u_max:
                    continue
                scale_u = u_max - u_min
                uu = u_min + scale_u * u01
                omega = np.sqrt(uu * uu + m_l_hat * m_l_hat)
                omega_p = omega - dx
                kp2 = np.where(omega_p > m_psi_hat, omega_p * omega_p - m_psi_hat * m_psi_hat, 0.0)
                z1 = 0.5 * omega
                z2 = 0.5 * omega_p
                ch1 = np.where(z1 > 50.0, 0.5 * np.exp(np.minimum(z1, 700.0)), np.cosh(z1))
                ch2 = np.where(z2 > 50.0, 0.5 * np.exp(np.minimum(z2, 700.0)), np.cosh(z2))
                stat = 1.0 / np.maximum(4.0 * ch1 * ch2, _SAFE_FLOOR)
                base = scale_u * uw01 * (uu / np.maximum(omega, _SAFE_FLOOR)) * stat
                base = np.where((omega_p > m_psi_hat) & (kp2 > 0.0), base, 0.0)
                pref0 = xi * xi / (32.0 * _PI * 2.0 * xq[i] * xq[j])
                m_ij = 0.0
                for mu, w_mu in zip(mu_x, mu_w):
                    qh2 = max(qi * qi + qj * qj - 2.0 * qi * qj * mu, 1.0e-12)
                    qh = math.sqrt(qh2)
                    t_hat = dx2 - qh2
                    # Spin/channel-summed chi L -> anti-chi Psi conversion
                    # kernel.  The factor 2 counts the two degenerate
                    # SU(2)_L bath components and belongs in |M|^2, not in
                    # the collision-rate prefactor.
                    m_sq = 2.0 * g_y * (4.0 * xi * xi - t_hat) * (m_psi_hat * m_psi_hat - t_hat) / max((t_hat - m_phi_hat * m_phi_hat) ** 2, _SAFE_FLOOR)
                    cos_th = (uu * uu + qh2 - kp2) / np.maximum(2.0 * uu * qh, _SAFE_FLOOR)
                    i_hat = float(np.sum(np.where(np.abs(cos_th) <= 1.0, base, 0.0)))
                    m_ij += w_mu * (pref0 / qh) * m_sq * i_hat
                m_ij *= 0.5
                matrix[i, j] = qj * qj * m_ij
                matrix[j, i] = qi * qi * m_ij
        for i in range(dim):
            matrix[i, i] = -float(np.sum(matrix[i, :] * np.exp((xq[i] - xq) / 2.0)))
        for i in range(dim - 1):
            for j in range(i + 1, dim):
                dxq = xq[i] - xq[j]
                matrix[i, j] *= math.exp(-dxq / 2.0)
                matrix[j, i] *= math.exp(dxq / 2.0)
        c = x_dm * xi**-7 * dq * dq / (6.0 * _PI**2)
        secmom = q**4 / xq
        avscatt = c * float(secmom @ (matrix @ fdm))
        der = ((x_dm * der_dbk2) - 1.0) * avscatt / y_safe + c * (yeqdy / y_safe) * float(secmom @ (matrix @ (xq * fdm)))
        return float(avscatt), float(der)

    def build_rate_set(self) -> Z4RateSet:
        return Z4RateSet(self, self.settings).build()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


class Z4Model:
    name = "Z4"

    def prepare_params(self, cards: dict) -> dict:
        params = dict(cards)
        params.setdefault("mS2", 10.0 * 4.0 * float(params.get("mDM", 20.57825138)) ** 2 / (1.0 + float(params.get("delta", 0.89583785))))
        return params

    def build_channels(self, params: dict) -> list[ThermalChannel]:
        raise RuntimeError("Z4 uses the model-local nBEl/cBEA runner; do not route it through the shared one-yield pipeline.")

    def build_relic_functions(self, params: dict, settings: dict, dof: Any) -> RelicFunctions:
        physics = Z4Physics(self.prepare_params(params), settings, dof)
        rates = physics.build_rate_set()
        return RelicFunctions(
            svx=rates.svx,
            yeq=physics.Yeq,
            channel_funcs={"z4_ann": rates.svx},
            metadata={"z4_physics": physics, "z4_rates": rates, "source": "current-project VLL.wl + nBEl.wl + cBEA.wl"},
        )
