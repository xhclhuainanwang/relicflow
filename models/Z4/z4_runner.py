"""Model-local Z4 runner: the current project's ``nBEl`` and ``cBEA`` path.

The source files are intentionally recorded in the output.  This runner is
not the shared one-yield solver: the active MMA cards evolve the abundance,
dark-sector temperature variable, and lepton asymmetry together, and use the
Euler/Trapezoidal implicit quadratic update from ``src\\nBEl.wl`` and
``src\\cBEA.wl``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from nbe.config import load_native_split
from nbe.constants import MIN_NUMBER, RHO_CRITICAL, T_TODAY
from nbe.dof import DofTable
from nbe.mathutils import bessel_k1_approx, bessel_k2_approx
from nbe.models.z4 import Z4Model, Z4Physics, Z4RateSet, _as_bool, _bessel_k2_scaled, _der_bessel_k2_over_k2


SOURCE_ROOT = Path(r"E:\work content\Z3\7.30\DRAKE_v1.0 3")
SOURCE_FILES = {
    "model": str(SOURCE_ROOT / "models" / "VLL" / "VLL.wl"),
    "nBE": str(SOURCE_ROOT / "src" / "nBEl.wl"),
    "cBE": str(SOURCE_ROOT / "src" / "cBEA.wl"),
    "thermal": str(SOURCE_ROOT / "src" / "thermalavgl.wl"),
    "rates": str(SOURCE_ROOT / "src" / "ratesl.wl"),
}

_SPHALERON_TSTAR_GEV_DEFAULT = 131.7
# The observed baryon asymmetry translated back through the SM sphaleron
# factor.  The corrected ``eps`` below is the input eps1 multiplied by the
# factor required to reproduce this observed B-L magnitude.
_Y_B_OBSERVED = 8.7e-11
_Y_BL_OBSERVED = (79.0 / 28.0) * _Y_B_OBSERVED
# The annihilation diagnostic uses one fixed SU(2)_L component in ``svx``.
# For chemical decoupling, include the two weak components and the two
# independent charge-conjugate channels.  ``nchi_eq`` already includes the
# Dirac X + Xbar internal states through gDM and must not be multiplied again.
_ANNIHILATION_WEAK_COMPONENT_FACTOR = 2.0
_ANNIHILATION_CHARGE_CONJUGATE_FACTOR = 2.0
_ANNIHILATION_TOTAL_FACTOR = (
    _ANNIHILATION_WEAK_COMPONENT_FACTOR
    * _ANNIHILATION_CHARGE_CONJUGATE_FACTOR
)
# The spin/channel-summed FullCel scattering kernel includes the two degenerate
# SU(2)_L weak components.  The independent charge-conjugate conversion
# channel is included once in the outer FullCel rate prefactor below.
_CONVERSION_CHARGE_CONJUGATE_FACTOR = 2.0

# Auxiliary Psi--n_R consistency checks from the bilingual note.  These are
# deliberately diagnostic-only: they do not alter the nBE/cBE evolution.
_PSI_LAMBDA_N_DEFAULT = 1.0e-6
_PSI_VEW_GEV_DEFAULT = 246.22
_PSI_EW_CROSSOVER_GEV_DEFAULT = 160.0
_PSI_MH_UNBROKEN_GEV_DEFAULT = 0.0
_PSI_MW_GEV_DEFAULT = 80.379
_PSI_MZ_GEV_DEFAULT = 91.1876
_PSI_MH_GEV_DEFAULT = 125.25
_PSI_GF_GEV_MINUS2_DEFAULT = 1.1663787e-5
_PSI_VUD_DEFAULT = 0.97420
_PSI_FPI_GEV_DEFAULT = 0.1302
_PSI_MPI_GEV_DEFAULT = 0.13957
_PSI_ME_GEV_DEFAULT = 5.1099895e-4
_PSI_MMU_GEV_DEFAULT = 1.056583755e-1
_PSI_ALPHA_EM_DEFAULT = 1.0 / 128.0
_PSI_NR_INTERNAL_DEGREES_DEFAULT = 2.0
_PSI_TNU_DECOUPLING_GEV_DEFAULT = 2.0e-3
_PSI_GSTAR_S_NU_DECOUPLING_DEFAULT = 10.75
_PSI_DECAY_GATE_DEFAULT = 10.0
_PSI_LIGHT_GATE_DEFAULT = 0.1


def _safe_den(value: float, floor: float = MIN_NUMBER) -> float:
    value = float(value)
    if abs(value) >= floor:
        return value
    return floor if value >= 0.0 else -floor


def _safe_y(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) and abs(value) >= MIN_NUMBER else (MIN_NUMBER if value >= 0.0 else -MIN_NUMBER)


def _safe_err(num: float, den: float) -> float:
    return abs(float(num)) / max(abs(float(den)), 1.0e-140)


def _safe_div(num: float, den: float, floor: float) -> float:
    return float(num) / _safe_den(den, max(abs(float(floor)), 1.0e-120))


def _corrected_eps_from_ybl(ybl_cbe_xstar: float | None, eps1: float) -> float | None:
    """Return the theory-corrected CP asymmetry needed by the observation."""

    if ybl_cbe_xstar is None:
        return None
    value = float(ybl_cbe_xstar)
    if not math.isfinite(value) or abs(value) <= MIN_NUMBER:
        return None
    return float(eps1) * float(_Y_BL_OBSERVED / abs(value))


def _resonance_C(physics: Z4Physics) -> dict[str, float]:
    """Evaluate the paper's dimensionless resonance coefficient C."""

    mchi = float(physics.m_dm)
    delta = float(physics.delta)
    r = float(physics.r)
    mphi = 2.0 * mchi / math.sqrt(max(1.0 + delta, MIN_NUMBER))
    gamma_phi = float(physics.gammas)
    gamma_phi_bar = mphi * gamma_phi / max(mchi**2, MIN_NUMBER)
    b_coeff = 4.0 * delta / max(1.0 + delta, MIN_NUMBER)
    c_prefactor = r**2 / _safe_den(2.0 * (4.0 - r**2))
    c_value = c_prefactor - _safe_div(
        2.0 * b_coeff,
        b_coeff**2 + gamma_phi_bar**2,
        MIN_NUMBER,
    )
    return {
        "mPhi_GeV": mphi,
        "GammaPhi_GeV": gamma_phi,
        "GammaPhi_bar": gamma_phi_bar,
        "B": b_coeff,
        "C": float(c_value),
    }


def _format_number(value: Any) -> str:
    """Project-wide console format: fixed 4 decimals or scientific 4 digits."""

    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan" if math.isnan(number) else ("inf" if number > 0.0 else "-inf")
    if number == 0.0:
        return "0.0000"
    if abs(number) < 1.0 or abs(number) >= 1.0e5:
        return f"{number:.4e}"
    return f"{number:.4f}"


def _parse_mma_result(path: Path | None) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "Oh2_nBE": None,
        "Oh2_cBE": None,
        "Ydl_nBE": None,
        "Ydl_cBE": None,
        "Ys_nBE": None,
        "Ys_cBE": None,
    }
    if path is None or not path.exists():
        return values
    text = path.read_text(encoding="utf-8", errors="ignore").replace("*^", "e")
    for key in values:
        # The MMA character file currently uses both ``key = value`` and
        # tab/space-separated ``key value`` records.  Accept either form so
        # BP comparisons do not silently become NA.
        match = re.search(rf"(?m)^\s*{re.escape(key)}\s*(?:=\s*)?([^\s,\]]+)", text)
        if match:
            try:
                values[key] = float(match.group(1).rstrip(";"))
            except ValueError:
                pass
    # Keep the historical aliases used by older callers.
    values["Ydl"] = values["Ydl_nBE"]
    values["cYdl"] = values["Ydl_cBE"]
    values["Ys"] = values["Ys_nBE"]
    values["cYs"] = values["Ys_cBE"]
    return values


def _settings_with_defaults(settings: dict[str, Any]) -> dict[str, Any]:
    out = dict(settings)
    defaults = {
        "xmin": 0.1,
        "cxmin": 0.1,
        "xmax": 1.5e5,
        "xlock1": 2.0,
        "sphaleronTstarGeV": _SPHALERON_TSTAR_GEV_DEFAULT,
        "Ydleqi0": 0.0,
        "errh": 1.01,
        "errch": 1.01,
        "epss": 1.0e-1,
        "epsl": 1.0e-2,
        "errY": 1.0,
        "errcY": 3.0,
        "nBEerr": 1.0e-1,
        "nBEsafety": 0.9,
        "nBEhinit": 1.0e-1,
        "cBEerr": 1.0e-1,
        "cBEsafety": 0.9,
        "cBEhinit": 1.0e-1,
        "cBEerrNewton": 1.0e-2,
        "cBEerrMaxNewton": 0.9,
        "Nx": 50,
        "Nx1": 20,
        "iacc": 0.1,
        "iacc1": 0.6,
        "imax": 5,
        "imax1": 5,
        "pg": 3,
        "RelThermalAv": True,
        "FullCel": True,
        "KDonly": False,
        "nBEuseTrapezoidal": True,
        "cBEuseTrapezoidal": False,
        "thermalWorkers": 1,
        "z4RateCache": True,
        "z4RateCacheDir": None,
        "fullCellDim": 150,
        "fullCellNMu": 10,
        "fullCellNU": 24,
        "fullCellUBase": 25.0,
        # Matches the active VLL.wl binding, not the unused builder default.
        "fullCellUTail": 10.0,
        "fullCellUCap": 50.0,
        # Non-elastic VLL scattering retains the complete second-moment
        # collision integral by default.  The scan profile may enable the
        # explicit late-time FullCel freezeout cutoff below; this is not an FP
        # replacement and is disabled for the formal/default BP settings.
        "fullCellSkipXdm": None,
        "cBENewtonMaxIter": 50,
        "cBEMaxStepRetries": 80,
        # Note-level Psi--n_R consistency diagnostics.  These values are
        # evaluated after the cosmological solve and never enter nBE/cBE.
        "psiDiagnostics": True,
        "lambdaN": _PSI_LAMBDA_N_DEFAULT,
        "vEW_GeV": _PSI_VEW_GEV_DEFAULT,
        "ewCrossover_GeV": _PSI_EW_CROSSOVER_GEV_DEFAULT,
        "mHUnbroken_GeV": _PSI_MH_UNBROKEN_GEV_DEFAULT,
        "mW_GeV": _PSI_MW_GEV_DEFAULT,
        "mZ_GeV": _PSI_MZ_GEV_DEFAULT,
        "mh_GeV": _PSI_MH_GEV_DEFAULT,
        "GF_GeV_minus2": _PSI_GF_GEV_MINUS2_DEFAULT,
        "Vud": _PSI_VUD_DEFAULT,
        "fPi_GeV": _PSI_FPI_GEV_DEFAULT,
        "mPi_GeV": _PSI_MPI_GEV_DEFAULT,
        "nRInternalDegrees": _PSI_NR_INTERNAL_DEGREES_DEFAULT,
        "TnuDec_GeV": _PSI_TNU_DECOUPLING_GEV_DEFAULT,
        "gStarSNuDec": _PSI_GSTAR_S_NU_DECOUPLING_DEFAULT,
        "psiDecayGate": _PSI_DECAY_GATE_DEFAULT,
        "psiLightGate": _PSI_LIGHT_GATE_DEFAULT,
        "psiLateGate": 0.1,
        "psiHubbleShiftGate": 0.01,
        "psiLateGridSize": 192,
        # Formal/BP runs use the exact massless-bath check.  The scan profile
        # may select the existing tabulated rate and records that provenance.
        "psiLightRateMode": "exact",
    }
    for key, value in defaults.items():
        out.setdefault(key, value)
    return out


def _coefficients(x: float, physics: Z4Physics, rates: Z4RateSet) -> dict[str, float]:
    """Active coefficient block shared by the current nBEl and cBEA files."""

    assert rates.svx is not None and rates.svxl is not None and rates.svxl1 is not None and rates.svxl2 is not None and rates.gs is not None
    s = physics.entropy(x)
    den_h = physics.hubble(x) * x
    gfac = 1.0 + physics.gtilde(x)
    yeq = physics.Yeq(x)
    yeq_safe2 = max(yeq**2, (yeq + 1.0e-120) ** 2)
    seq = physics.Yseq(x)
    y_g = physics.Yg(x)
    eta_inv = _safe_div(1.0, physics.eta(x), MIN_NUMBER)
    ksi_over_eta = physics.ksi(x) * eta_inv
    svx = float(rates.svx(x))
    svxl = float(rates.svxl(x))
    svxl1 = float(rates.svxl1(x))
    svxl2 = float(rates.svxl2(x))
    gs = float(rates.gs(x))
    floor_h = den_h * 1.0e-120
    floor_y = den_h * (abs(yeq) + 1.0e-120) ** 2

    lsc = _safe_div(gs * seq * physics.brc * gfac, den_h * yeq_safe2, floor_y)
    llc = -(eta_inv * physics.eps1 / 2.0) * lsc * physics.br_l + _safe_div(0.5 * s * eta_inv * physics.eps1 * svx * gfac, den_h, floor_h)
    kll = (
        gfac * ksi_over_eta * physics.br_l * gs * seq * (physics.eps1**2 * physics.br_l + physics.brc) / _safe_den(den_h * 4.0 * y_g)
        - gfac * ksi_over_eta * s / _safe_den(den_h * y_g) * (physics.YLeq(x) * physics.Ypeq(x) * 2.0 * svxl + svx / 4.0 * yeq_safe2 + 2.0 * svxl2 * physics.Ypeq(x) ** 2)
    )
    kls = eta_inv * gs * physics.eps1 * physics.br_l * gfac / _safe_den(2.0 * den_h)
    cl = _safe_div(eta_inv * physics.eps1 / 2.0 * physics.br_l * gs * gfac, den_h, floor_h) * seq * (1.0 - 2.0 * physics.br_l - physics.brc) - _safe_div(0.5 * s * eta_inv * physics.eps1 * svx * gfac, den_h, floor_h) * yeq_safe2
    lsc = _safe_div(gs * seq * physics.brc * gfac, den_h * yeq_safe2, floor_y)
    kss = -gs * gfac / _safe_den(den_h)
    cs = gs * gfac * seq * physics.br_l / _safe_den(den_h)
    lcc = -_safe_div(s * svx * gfac, den_h, floor_h) - _safe_div(physics.brc**2 * gs * seq * gfac, den_h * yeq_safe2, floor_y)
    kcs = _safe_div(physics.brc * gs * gfac, den_h, floor_h)
    kcl = -_safe_div(ksi_over_eta * s * physics.eps1 * svx * gfac / _safe_den(2.0 * y_g), den_h, floor_h) * yeq_safe2 - _safe_div(ksi_over_eta * physics.eps1 * gfac * physics.brc * physics.br_l * gs * seq / _safe_den(2.0 * y_g), den_h, floor_h)
    cc = _safe_div(s * svx * gfac, den_h, floor_h) * yeq_safe2 - _safe_div(physics.brc * physics.br_l * gs * seq * gfac, den_h, floor_h)
    klc = -ksi_over_eta * s * svxl1 * physics.Ypeq(x) * gfac / _safe_den(den_h * y_g)
    return {
        "s": s,
        "den_h": den_h,
        "gfac": gfac,
        "Yeq": yeq,
        "YeqSafe2": yeq_safe2,
        "Yseq": seq,
        "Yg": y_g,
        "svx": svx,
        "svxl": svxl,
        "svxl1": svxl1,
        "svxl2": svxl2,
        "gs": gs,
        "lsc": lsc,
        "llc": llc,
        "kll": kll,
        "kls": kls,
        "Cl": cl,
        "kss": kss,
        "Cs": cs,
        "lcc": lcc,
        "kcs": kcs,
        "kcl": kcl,
        "Cc": cc,
        "klc": klc,
    }


def _abundance_roots(
    h: float,
    yi: float,
    ysi: float,
    ydli: float,
    prev: dict[str, float],
    cur: dict[str, float],
    physics: Z4Physics,
    *,
    lcc_override: float | None = None,
    llc_override: float | None = None,
    klc_override: float | None = None,
) -> dict[str, float]:
    """Port the algebraic quadratic elimination used in nBEl/cBEA."""

    yeq_p = cur["Yeq"]
    llc_cur = cur["llc"] if llc_override is None else float(llc_override)
    klc_cur = cur["klc"] if klc_override is None else float(klc_override)
    den_s_t = _safe_den(2.0 - h * cur["kss"])
    den_l_t = _safe_den(2.0 - h * cur["kll"] - h * klc_cur * yeq_p)
    us = h * cur["lsc"]
    as_t = 2.0 * ysi + h * cur["Cs"] + h * prev["lsc"] * yi**2 + h * prev["kss"] * ysi + h * prev["Cs"]
    bs = as_t / den_s_t
    vs = us / den_s_t
    al = (2.0 + h * prev["kll"]) * ydli + h * cur["Cl"] + h * prev["llc"] * yi**2 + h * prev["kls"] * ysi + h * prev["Cl"] + h * prev["klc"] * ydli * yi
    bl = al / den_l_t
    vl = h * llc_cur / den_l_t
    lls = h * cur["kls"] / den_l_t
    rs = vs * lls
    # cBEA.wl uses the current-step coefficients kcsi/kcli here (the values
    # carried from the previously accepted point), while Ccip1 is evaluated
    # at the endpoint.  These are prev[kcs]/prev[kcl] in the Python split.
    ac = 2.0 * yi + h * cur["Cc"] + h * prev["lcc"] * yi**2 + h * prev["kcs"] * ysi + h * prev["kcl"] * ydli + h * prev["Cc"]
    eac = yi + h * cur["Cc"]
    eas = ysi + h * cur["Cs"]
    ebs = eas / _safe_den(1.0 - h * cur["kss"])
    evs = us / _safe_den(1.0 - h * cur["kss"])
    eal = ydli + h * cur["Cl"]
    den_l_e = _safe_den(1.0 - h * cur["kll"] - h * klc_cur * yeq_p)
    ebl = eal / den_l_e
    evl = h * llc_cur / den_l_e
    ells = h * cur["kls"] / den_l_e
    ers = evs * ells
    edc = eac + h * cur["kcs"] * ebs + h * cur["kcl"] * (ebl + ells * ebs)
    lcc = cur["lcc"] if lcc_override is None else float(lcc_override)
    u = h * lcc
    uuc = u + h * cur["kcs"] * vs + h * cur["kcl"] * vl + h * cur["kcl"] * rs
    dc = ac + h * cur["kcs"] * bs + h * cur["kcl"] * (bl + lls * bs)
    e_uuc = u + h * cur["kcs"] * evs + h * cur["kcl"] * evl + h * cur["kcl"] * ers
    if 1.0 - uuc * dc < 0.0 or 0.25 - e_uuc * edc < 0.0:
        raise _StepRetry("negative implicit quadratic discriminant")
    y_trap = dc / _safe_den(1.0 + math.sqrt(max(1.0 - uuc * dc, 0.0)))
    y_euler = edc / _safe_den(0.5 + math.sqrt(max(0.25 - e_uuc * edc, 0.0)))
    return {
        "TrapY": y_trap,
        "EulY": y_euler,
        "TrapYs": cur["Yseq"],
        "EulYs": cur["Yseq"],
        "TrapYdl": bl + vl * y_trap**2 + lls * cur["Yseq"],
        "EulYdl": ebl + evl * y_euler**2 + ells * cur["Yseq"],
        "dc": dc,
        "edc": edc,
        "Ac": ac,
        "eAc": eac,
        "uuc": uuc,
        "euuc": e_uuc,
        "bs": bs,
        "vs": vs,
        "bl": bl,
        "vl": vl,
        "lls": lls,
        "rs": rs,
    }


class _StepRetry(RuntimeError):
    pass


def _cbe_temperature_coefficients(
    x: float,
    y_temp: float,
    physics: Z4Physics,
    rates: Z4RateSet,
    cur: dict[str, float],
) -> tuple[float, float, float]:
    """Return the xDM-dependent cBEA coefficients for one trial y.

    In ``cBEA.wl`` these coefficients are refreshed once before the Euler or
    trapezoidal Newton loop, and once after the accepted temperature.  They
    are not recomputed at every Newton trial.
    """

    assert rates.svx is not None and rates.svxl1 is not None
    x_dm = x * physics.yeq(x) / _safe_y(y_temp)
    svx_dm = float(rates.svx(x_dm))
    lcc = -_safe_div(
        cur["s"] * svx_dm * cur["gfac"],
        cur["den_h"],
        cur["den_h"] * 1.0e-120,
    ) - cur["lsc"] * physics.brc
    eta_inv = _safe_div(1.0, physics.eta(x), MIN_NUMBER)
    ksi_over_eta = physics.ksi(x) * eta_inv
    llc = (
        -(eta_inv * physics.eps1 / 2.0) * cur["lsc"] * physics.br_l
        + _safe_div(
            0.5 * cur["s"] * eta_inv * physics.eps1 * svx_dm * cur["gfac"],
            cur["den_h"],
            cur["den_h"] * 1.0e-120,
        )
    )
    klc = -(
        ksi_over_eta
        * cur["s"]
        * float(rates.svxl1(x_dm))
        * physics.Ypeq(x)
        * cur["gfac"]
        / _safe_den(cur["den_h"] * physics.Yg(x))
    )
    return float(lcc), float(llc), float(klc)


def _step_update(h: float, err: float, safety: float, growth: float) -> float:
    if err == 0.0:
        return growth * h
    return min(safety / math.sqrt(max(err, MIN_NUMBER)), growth) * h


def solve_nbel(physics: Z4Physics, rates: Z4RateSet, settings: dict[str, Any]) -> dict[str, Any]:
    assert rates.svx is not None
    xmin = float(settings["xmin"])
    xmax = float(settings["xmax"])
    h = float(settings["nBEhinit"])
    eps = float(settings["nBEerr"])
    safety = float(settings["nBEsafety"])
    err_y_max = float(settings["errY"])
    use_trap = _as_bool(settings.get("nBEuseTrapezoidal"), True)
    yi = physics.Yeq(xmin)
    ysi = physics.Yseq(xmin)
    ydli = float(settings.get("Ydleqi0", 0.0))
    prev = _coefficients(xmin, physics, rates)
    rows_nbe = [[xmin, yi]]
    rows_ydl = [[xmin, ydli]]
    rows_ys = [[xmin, ysi]]
    attempts = 0
    while xmin < xmax:
        hh = min(h, xmax - xmin)
        accepted = False
        while not accepted:
            attempts += 1
            if attempts > int(settings.get("nBEMaxStepRetries", 200000)):
                raise RuntimeError("nBEl exceeded the step retry limit")
            x_next = xmin + hh
            if x_next == xmin:
                raise RuntimeError("nBEl step-size overflow")
            cur = _coefficients(x_next, physics, rates)
            try:
                roots = _abundance_roots(
                    hh,
                    yi,
                    ysi,
                    ydli,
                    prev,
                    cur,
                    physics,
                )
            except _StepRetry:
                hh *= 0.5
                continue
            if use_trap:
                trap_y, euler_y = roots["TrapY"], roots["EulY"]
            else:
                trap_y = euler_y = roots["EulY"]
                roots["TrapYdl"] = roots["EulYdl"]
                roots["TrapYs"] = roots["EulYs"]
            err = _safe_err(trap_y - euler_y, trap_y * eps)
            if err > err_y_max:
                shrink = safety * math.sqrt(err_y_max / max(err, MIN_NUMBER))
                hh = max(min(shrink, safety), 0.1) * hh
                continue
            accepted = True
        xmin = x_next
        yi = _safe_y(euler_y)
        ydli = float(roots["EulYdl"])
        ysi = float(roots["EulYs"])
        prev = cur
        rows_nbe.append([xmin, float(roots["TrapY"])])
        rows_ydl.append([xmin, ydli])
        rows_ys.append([xmin, ysi])
        h = _step_update(hh, err, safety, float(settings["errh"]))
    entropy_today = float(physics.dof.iheff(T_TODAY)) * 2.0 * math.pi**2 / 45.0 * T_TODAY**3
    oh2 = physics.m_dm * entropy_today / RHO_CRITICAL * float(rows_nbe[-1][1])
    return {
        "xy": rows_nbe,
        "xy_ydl": rows_ydl,
        "xy_ys": rows_ys,
        "oh2": float(oh2),
        "Yc": float(rows_nbe[-1][1]),
        "Ydl": abs(float(rows_ydl[-1][1])),
        "Ys": float(rows_ys[-1][1]),
        "attempts": attempts,
    }


def _gamma_fp_proxy(physics: Z4Physics, rates: Z4RateSet, x: float) -> float:
    # The active VLL card runs FullCel=True.  This path exists only for an
    # explicit FullCel=False override and preserves the dimensions of gam/s
    # in cBEA using the already tabulated visible-width average.
    assert rates.gs is not None
    return physics.entropy(x) * float(rates.gs(x))


def _rate_derivative(rate: Callable[[float], float], x: float) -> float:
    """Use the log-table derivative without another expensive thermal integral."""

    derivative = getattr(rate, "derivative", None)
    if callable(derivative):
        return float(derivative(float(x)))
    xx = max(float(x), 1.0e-12)
    step = max(xx * 1.0e-5, 1.0e-8)
    return float((rate(xx + step) - rate(max(xx - step, 1.0e-12))) / (2.0 * step))


def _cbe_rhs(
    x: float,
    y_temp: float,
    y_ab: float,
    yi_temp: float,
    yi_ab: float,
    h: float,
    physics: Z4Physics,
    rates: Z4RateSet,
    cur: dict[str, float],
    prev: dict[str, float],
    settings: dict[str, Any],
    ysi: float,
    ydli: float,
    *,
    full_cell: bool,
    y_ab_fixed: float | None = None,
    y_ab_roots: dict[str, float] | None = None,
    abundance_kind: str = "Eul",
) -> tuple[float, float, float, dict[str, float]]:
    y_safe = _safe_y(y_temp)
    x_dm = x * physics.yeq(x) / y_safe
    x_dm = max(float(x_dm), 1.0e-12)
    svx_dm = float(rates.svx(x_dm))
    assert rates.sv2x is not None and rates.gs is not None
    sv2x_dm = float(rates.sv2x(x_dm))
    # cBEA keeps the abundance quadratic fixed while Newton solves the
    # temperature equation, then recomputes it once after the accepted
    # temperature is known.  ``y_ab_fixed`` implements that active ordering;
    # the fallback retains the same post-Newton recomputation when this helper
    # is called directly.
    if y_ab_fixed is None:
        lcc_dm = -_safe_div(cur["s"] * svx_dm * cur["gfac"], cur["den_h"], cur["den_h"] * 1.0e-120) - cur["lsc"] * physics.brc
        try:
            roots = _abundance_roots(
                h,
                yi_ab,
                ysi,
                ydli,
                prev,
                cur,
                physics,
                lcc_override=lcc_dm,
            )
        except _StepRetry:
            roots = _abundance_roots(
                h,
                yi_ab,
                ysi,
                ydli,
                prev,
                cur,
                physics,
                lcc_override=cur["lcc"],
            )
        y_ab_next = float(roots["TrapY"])
    else:
        roots = dict(y_ab_roots or {})
        y_ab_next = float(y_ab_fixed)
    y_ab_next = _safe_y(y_ab_next)
    lam = cur["s"] / _safe_den(cur["den_h"]) * cur["gfac"]
    prod1 = lam * y_safe * y_ab_next / 4.0
    prod2 = svx_dm - sv2x_dm - (cur["Yeq"] / y_ab_next) ** 2 * (cur["svx"] - physics.yeq(x) / y_safe * float(rates.sv2x(x)))
    w34, w34p = physics.rel_temp_correction(x_dm)
    if full_cell:
        # This is the non-elastic VLL path: always evaluate the complete
        # second-moment collision integral.  In particular, do not replace it
        # by a Fokker--Planck term or set it to zero at late xDM.
        scat_pref = _CONVERSION_CHARGE_CONJUGATE_FACTOR * physics.m_dm**3 / _safe_den(physics.hubble(x) * cur["s"] ** (2.0 / 3.0) * x) * cur["gfac"]
        # cBEA passes the lower-case thermal equilibrium variable yeq[mDM,x]
        # here; the uppercase Yeq is the abundance and must not enter the
        # phase-space q rescaling.
        scat = physics.second_moment_scattering(x, physics.yeq(x), y_safe, _bessel_k2_scaled(x_dm), _der_bessel_k2_over_k2(x_dm))
        rhs = cur["gfac"] * (y_safe / x) * w34 + scat_pref * scat[0] + prod1 * prod2
        scat_jac = scat_pref * scat[1]
    else:
        gam_ds = _gamma_fp_proxy(physics, rates, x) / _safe_den(cur["s"])
        rhs = cur["gfac"] * (y_safe / x) * w34 + lam * gam_ds * (1.0 - w34 / 2.0) * (physics.yeq(x) - y_safe) + prod1 * prod2
        scat_jac = -lam * gam_ds * (1.0 - w34 / 2.0)

    dsvx_dm = _rate_derivative(rates.svx, x_dm)
    dsv2x_dm = _rate_derivative(rates.sv2x, x_dm)
    dersvx_dm = -(x_dm / y_safe) * dsvx_dm
    dersv2x_dm = -(x_dm / y_safe) * dsv2x_dm
    if abundance_kind == "Trap":
        root_y = max(float(roots.get("TrapY", y_ab_next)), MIN_NUMBER)
        discriminant = max(1.0 - float(roots.get("uuc", 0.0)) * float(roots.get("dc", 0.0)), MIN_NUMBER)
        root_ac = float(roots.get("Ac", 0.0))
    else:
        root_y = max(float(roots.get("EulY", y_ab_next)), MIN_NUMBER)
        discriminant = max(0.25 - float(roots.get("euuc", 0.0)) * float(roots.get("edc", 0.0)), MIN_NUMBER)
        root_ac = float(roots.get("eAc", 0.0))
    # cBEA.wl retains the explicit Euler-root sensitivity in DerYip1E.  In
    # the active VLL card lambdaip1 = llip1*svxip1, so its svxip1 denominator
    # cancels and the source expression reduces to the form below.
    der_y_ab = 0.0
    if abundance_kind == "Eul" and root_y > MIN_NUMBER and discriminant > MIN_NUMBER:
        der_y_ab = -(
            root_y**2
            * h
            * lam
            * dersvx_dm
            / (2.0 * math.sqrt(discriminant))
        )
    lep_jac = 0.0
    if abundance_kind == "Eul" and root_y > MIN_NUMBER:
        # cBEA.wl retains this VLL lepton-temperature coupling in the Euler
        # FullCel RHS.  EulYsip1 is the previous-step Yseqi, while Yseqip1 is
        # the endpoint coefficient; using cur["Yseq"] for both is not the
        # source ordering.
        lep_source = physics.brc * cur["gs"] * (ysi - physics.br_l * cur["Yseq"])
        lep_term = -cur["gfac"] * lep_source * y_safe / _safe_den(cur["den_h"] * root_y)
        rhs += lep_term
        lep_jac = cur["gfac"] * lep_source * (
            y_safe * der_y_ab / root_y**2 - 1.0 / root_y
        ) / _safe_den(cur["den_h"])
    prod2_jac = (
        dersvx_dm
        - dersv2x_dm
        + 2.0 * (cur["Yeq"] / root_y) ** 2 * (der_y_ab / root_y) * (cur["svx"] - physics.yeq(x) / y_safe * float(rates.sv2x(x)))
        - (cur["Yeq"] / root_y) ** 2 * (float(rates.sv2x(x)) * physics.yeq(x) / y_safe**2)
    )
    jac_hint = cur["gfac"] * (w34 - x_dm * w34p) / x + scat_jac + prod1 * ((1.0 / y_safe + der_y_ab / root_y) * prod2 + prod2_jac) + lep_jac
    return float(rhs), float(jac_hint), y_ab_next, roots


def _solve_temperature_newton(
    x: float,
    h: float,
    yi_temp: float,
    yi_ab: float,
    prev_rhs: float,
    prev: dict[str, float],
    cur: dict[str, float],
    physics: Z4Physics,
    rates: Z4RateSet,
    settings: dict[str, Any],
    ysi: float,
    ydli: float,
    *,
    full_cell: bool,
    trapezoidal: bool,
) -> tuple[float, float, float, float, float, dict[str, float]]:
    y = float(yi_temp)
    max_iter = max(4, int(settings.get("cBENewtonMaxIter", 50)))
    err_newton = float(settings["cBEerrNewton"])
    err_max = float(settings["cBEerrMaxNewton"])
    newton_damping = min(max(float(settings.get("cBENewtonDamping", 1.0)), 1.0e-3), 1.0)
    alpha = h / 2.0 if trapezoidal else h
    last_roots: dict[str, float] = {}
    last_rhs = prev_rhs
    last_jac = 0.0
    last_yab = yi_ab
    # cBEA updates xDM-dependent coefficients once before entering Newton;
    # the roots are fixed during the iterations and rebuilt after acceptance.
    lcc_initial, llc_initial, klc_initial = _cbe_temperature_coefficients(
        x, yi_temp, physics, rates, cur
    )
    fixed_roots = _abundance_roots(
        h,
        yi_ab,
        ysi,
        ydli,
        prev,
        cur,
        physics,
        lcc_override=lcc_initial,
        llc_override=llc_initial,
        klc_override=klc_initial,
    )
    fixed_yab = fixed_roots["TrapY" if trapezoidal else "EulY"]
    for _ in range(max_iter):
        rhs, jac_hint, y_ab_next, _ = _cbe_rhs(
            x, y, yi_ab, yi_temp, yi_ab, h, physics, rates, cur, prev, settings, ysi, ydli,
            full_cell=full_cell, y_ab_fixed=fixed_yab, y_ab_roots=fixed_roots,
            abundance_kind="Trap" if trapezoidal else "Eul",
        )
        # The active cBEA.wl path supplies an analytical Jacobian.  The
        # translated FullCel derivative now provides the same scattering
        # piece, so a default FullCel Newton iteration needs one native call,
        # not three centered-difference calls.  Keep the centered fallback for
        # the optional non-FullCel route, where the full source Jacobian has a
        # separate Fokker--Planck correction block.
        jac = float(jac_hint)
        if not full_cell or not math.isfinite(jac):
            dy = max(abs(y) * 1.0e-5, 1.0e-12)
            rhs_p = _cbe_rhs(
                x, y + dy, yi_ab, yi_temp, yi_ab, h, physics, rates, cur, prev, settings, ysi, ydli,
                full_cell=full_cell, y_ab_fixed=fixed_yab, y_ab_roots=fixed_roots,
                abundance_kind="Trap" if trapezoidal else "Eul",
            )[0]
            rhs_m = _cbe_rhs(
                x, y - dy, yi_ab, yi_temp, yi_ab, h, physics, rates, cur, prev, settings, ysi, ydli,
                full_cell=full_cell, y_ab_fixed=fixed_yab, y_ab_roots=fixed_roots,
                abundance_kind="Trap" if trapezoidal else "Eul",
            )[0]
            jac = (rhs_p - rhs_m) / (2.0 * dy)
            if not math.isfinite(jac):
                jac = jac_hint
        if trapezoidal:
            residual = y - yi_temp - alpha * (prev_rhs + rhs)
        else:
            # cBEA.wl's active SkipTrapezoidal branch is implicit Euler:
            # it contains only the endpoint RHS, not the previous-step RHS.
            residual = y - yi_temp - h * rhs
        delta = residual / _safe_den(1.0 - alpha * jac)
        if newton_damping < 1.0:
            delta *= newton_damping
        y -= delta
        error = abs(delta) / max(abs(y), MIN_NUMBER)
        last_rhs, last_jac, last_yab, last_roots = rhs, jac, y_ab_next, fixed_roots
        if error <= err_newton:
            return float(y), float(last_yab), float(last_rhs), float(error), float(last_jac), last_roots
        if error > err_max and newton_damping >= 1.0:
            raise _StepRetry("cBE Newton correction exceeded cBEerrMaxNewton")
    raise _StepRetry("cBE Newton iteration did not converge")


def solve_cbea(physics: Z4Physics, rates: Z4RateSet, settings: dict[str, Any]) -> dict[str, Any]:
    assert rates.svx is not None and rates.sv2x is not None and rates.svxl1 is not None
    xi = float(settings.get("cxmin", settings["xmin"]))
    xmax = float(settings["xmax"])
    h = float(settings["cBEhinit"])
    eps = float(settings["cBEerr"])
    safety = float(settings["cBEsafety"])
    err_y_max = float(settings["errY"])
    full_cell = _as_bool(settings.get("FullCel"), True)
    use_trap = _as_bool(settings.get("cBEuseTrapezoidal"), False)
    x_lock = float(settings.get("cBEYeqlock", settings.get("xlock1", 2.0)))
    if not full_cell:
        raise ValueError("Z4/VLL cBE requires FullCel=True; Fokker--Planck approximation is disabled for inelastic scattering.")
    yi_ab = physics.Yeq(xi)
    yi_temp = physics.yeq(xi)
    ysi = physics.Yseq(xi)
    ydli = float(settings.get("Ydleqi0", 0.0))
    prev = _coefficients(xi, physics, rates)
    prev_rhs = 0.0
    rows = [[xi, yi_ab, yi_temp]]
    rows_ydl = [[xi, ydli]]
    rows_ys = [[xi, ysi]]
    rows_gamma: list[list[float]] = []
    rows_gamma_thermal: list[list[float]] = []
    thermal_root_bracketed = False
    attempts = 0
    last_retry_reason = "unknown"
    while xi < xmax:
        hh = min(h, xmax - xi)
        while True:
            attempts += 1
            if attempts > int(settings.get("cBEMaxStepRetries", 200000)):
                raise RuntimeError(
                    "cBEA exceeded the step retry limit at "
                    f"x={_format_number(xi)}, h={_format_number(hh)}; "
                    f"last reason: {last_retry_reason}"
                )
            x_next = xi + hh
            if x_next == xi:
                raise RuntimeError("cBEA step-size overflow")
            cur = _coefficients(x_next, physics, rates)
            cur["hubble"] = physics.hubble(x_next)
            lock_temp = x_next <= x_lock
            try:
                if lock_temp:
                    y_t, y_e = physics.yeq(x_next), physics.yeq(x_next)
                    y_ab_t = y_ab_e = physics.Yeq(x_next)
                    rhs_t = rhs_e = 0.0
                    newton_err_t = newton_err_e = 0.0
                    roots_t = _abundance_roots(
                        hh,
                        yi_ab,
                        ysi,
                        ydli,
                        prev,
                        cur,
                        physics,
                    )
                    roots_e = roots_t
                elif not use_trap:
                    # This is the active cBEA branch for the current VLL card:
                    # cBEuseTrapezoidal=False jumps directly to Euler, just as
                    # cBEA.wl does at SkipTrapezoidal.  MMA nevertheless keeps
                    # the pre-Euler Yip1T/yi predictor for step control; it does
                    # not set the cBE error to zero.
                    lcc_initial, llc_initial, klc_initial = _cbe_temperature_coefficients(
                        x_next, yi_temp, physics, rates, cur
                    )
                    predictor_roots = _abundance_roots(
                        hh,
                        yi_ab,
                        ysi,
                        ydli,
                        prev,
                        cur,
                        physics,
                        lcc_override=lcc_initial,
                        llc_override=llc_initial,
                        klc_override=klc_initial,
                    )
                    y_t = _safe_y(yi_temp)
                    y_ab_t = _safe_y(predictor_roots["TrapY"])
                    roots_t = predictor_roots
                    rhs_t = _cbe_rhs(
                        x_next, y_t, yi_ab, yi_temp, yi_ab, hh, physics, rates, cur, prev, settings, ysi, ydli,
                        full_cell=full_cell, y_ab_fixed=y_ab_t, y_ab_roots=predictor_roots,
                        abundance_kind="Trap",
                    )[0]
                    y_e, y_ab_e, rhs_e, newton_err_e, _, roots_e = _solve_temperature_newton(
                        x_next, hh, yi_temp, yi_ab, prev_rhs, prev, cur, physics, rates, settings, ysi, ydli, full_cell=full_cell, trapezoidal=False
                    )
                    newton_err_t = 0.0
                else:
                    y_t, y_ab_t, rhs_t, newton_err_t, _, roots_t = _solve_temperature_newton(
                        x_next, hh, yi_temp, yi_ab, prev_rhs, prev, cur, physics, rates, settings, ysi, ydli, full_cell=full_cell, trapezoidal=True
                    )
                    y_e, y_ab_e, rhs_e, newton_err_e, _, roots_e = _solve_temperature_newton(
                        x_next, hh, yi_temp, yi_ab, prev_rhs, prev, cur, physics, rates, settings, ysi, ydli, full_cell=full_cell, trapezoidal=False
                    )
            except _StepRetry as exc:
                last_retry_reason = str(exc)
                hh *= 0.5
                continue
            if not lock_temp:
                # After Newton, cBEA updates lcc with svx[x*yeq/y] and
                # rebuilds the abundance quadratic before accepting the step.
                x_dm_e = x_next * physics.yeq(x_next) / _safe_y(y_e)
                svx_dm_e = float(rates.svx(x_dm_e))
                lcc_final = -_safe_div(
                    cur["s"] * svx_dm_e * cur["gfac"],
                    cur["den_h"],
                    cur["den_h"] * 1.0e-120,
                ) - cur["lsc"] * physics.brc
                eta_inv_e = _safe_div(1.0, physics.eta(x_next), MIN_NUMBER)
                ksi_over_eta_e = physics.ksi(x_next) * eta_inv_e
                llc_final = (
                    -(eta_inv_e * physics.eps1 / 2.0) * cur["lsc"] * physics.br_l
                    + _safe_div(
                        0.5 * cur["s"] * eta_inv_e * physics.eps1 * svx_dm_e * cur["gfac"],
                        cur["den_h"],
                        cur["den_h"] * 1.0e-120,
                    )
                )
                klc_final = -(
                    ksi_over_eta_e
                    * cur["s"]
                    * float(rates.svxl1(x_dm_e))
                    * physics.Ypeq(x_next)
                    * cur["gfac"]
                    / _safe_den(cur["den_h"] * physics.Yg(x_next))
                )
                try:
                    final_roots = _abundance_roots(
                        hh,
                        yi_ab,
                        ysi,
                        ydli,
                        prev,
                        cur,
                        physics,
                        lcc_override=lcc_final,
                        llc_override=llc_final,
                        klc_override=klc_final,
                    )
                except _StepRetry as exc:
                    last_retry_reason = str(exc)
                    hh *= 0.5
                    continue
                y_ab_e = _safe_y(final_roots["EulY"])
                roots_e = final_roots
                # cBEA carries these accepted, temperature-dependent
                # coefficients into the next step.  Keeping the equilibrium
                # values here changes the lepton history while leaving the
                # total abundance deceptively close.
                cur["lcc"] = float(lcc_final)
                cur["llc"] = float(llc_final)
                cur["klc"] = float(klc_final)
                if use_trap:
                    y_ab_t = _safe_y(final_roots["TrapY"])
                    roots_t = final_roots
            err_ab = _safe_err(y_ab_t - y_ab_e, y_ab_t * eps)
            err_temp = _safe_err(y_t - y_e, y_t * eps)
            err = max(err_ab, err_temp)
            if err > err_y_max:
                last_retry_reason = (
                    f"error={_format_number(err)} > errY={_format_number(err_y_max)}"
                )
                shrink = safety * math.sqrt(err_y_max / max(err, MIN_NUMBER))
                hh = max(min(shrink, safety), 0.1) * hh
                continue
            break
        xi = x_next
        yi_ab = _safe_y(y_ab_e)
        yi_temp = _safe_y(y_e)
        ysi = float(roots_e["EulYs"])
        ydli = float(roots_e["EulYdl"])
        prev = cur
        prev_rhs = float(rhs_e)
        rows.append([xi, yi_ab, yi_temp])
        rows_ydl.append([xi, ydli])
        rows_ys.append([xi, ysi])
        if full_cell:
            # The source records a derived full-cell scattering rate only when
            # its denominator is nonzero; the native output keeps the same
            # diagnostic with gamma/H units.
            den = cur["s"] * (physics.yeq(xi) - yi_temp)
            if abs(den) > 1.0e-50:
                rows_gamma.append([xi, abs(rhs_e / den) * cur["s"]])
            # This is the tGammaThOverHFull diagnostic used by the active
            # main_cbe_VLL.wls driver for xk = xkdScaJac.  Evaluate the
            # equilibrium full-cell Jacobian only until its first descending
            # crossing of one; later points cannot change that physical root
            # and skipping them keeps the general BP path fast.
            if xi >= 1.0 and not thermal_root_bracketed:
                gamma_th_over_h = _thermal_gamma_over_h(physics, xi)
                rows_gamma_thermal.append([xi, gamma_th_over_h])
                if (
                    len(rows_gamma_thermal) >= 2
                    and rows_gamma_thermal[-2][1] >= 1.0
                    and gamma_th_over_h <= 1.0
                ):
                    thermal_root_bracketed = True
        h = _step_update(hh, err, safety, float(settings["errch"]))
    entropy_today = float(physics.dof.iheff(T_TODAY)) * 2.0 * math.pi**2 / 45.0 * T_TODAY**3
    oh2 = physics.m_dm * entropy_today / RHO_CRITICAL * yi_ab
    return {
        "xy": rows,
        "xy_ydl": rows_ydl,
        "xy_ys": rows_ys,
        "oh2": float(oh2),
        "Yc": float(yi_ab),
        "Ydl": abs(float(ydli)),
        "Ys": float(ysi),
        "gamma_scattering": rows_gamma,
        "gamma_thermal_over_h": rows_gamma_thermal,
        "attempts": attempts,
        "full_cell": full_cell,
        "use_trapezoidal": use_trap,
    }


def _find_crossing(rows: list[list[float]], target: float, start_x: float = -math.inf) -> float | None:
    pairs = [(a, b) for a, b in zip(rows[:-1], rows[1:]) if a[0] >= start_x or b[0] >= start_x]
    for left, right in pairs:
        y1, y2 = abs(left[1]), abs(right[1])
        if (y1 - target) * (y2 - target) <= 0.0:
            if y2 == y1:
                return float(right[0])
            return float(left[0] + (target - y1) * (right[0] - left[0]) / (y2 - y1))
    return None


def _thermal_gamma_over_h(physics: Z4Physics, x: float) -> float:
    """Return the total conversion Jacobian gamma_conv/tilde-H.

    The active ``main_cbe_VLL.wls`` path constructs this from
    ``scatpref * GetSecondMomentScat[..., yeq, yeq, ...][[2]]`` and then uses
    the root of this quantity equal to one as ``xk``.  Calling the native
    FullCel implementation directly here also keeps this diagnostic
    independent of the late-time ``fullCellSkipXdm`` approximation used by
    the cBE evolution itself.
    """

    xx = float(x)
    gfac = 1.0 + physics.gtilde(xx)
    prefactor = _CONVERSION_CHARGE_CONJUGATE_FACTOR * physics.m_dm**3 / _safe_den(
        physics.hubble(xx) * physics.entropy(xx) ** (2.0 / 3.0) * xx
    ) * gfac
    yeq = physics.yeq(xx)
    _, jacobian = physics.second_moment_scattering(
        xx,
        yeq,
        yeq,
        _bessel_k2_scaled(xx),
        _der_bessel_k2_over_k2(xx),
    )
    value = abs(-xx * prefactor * float(jacobian) / 2.0)
    return float(value) if math.isfinite(value) else 0.0


def _root_from_table(rows: list[list[float]], target: float = 1.0) -> float | None:
    """Linearly interpolate the first descending table crossing of target."""

    valid = [
        (float(row[0]), float(row[1]))
        for row in rows
        if len(row) >= 2 and math.isfinite(float(row[0])) and math.isfinite(float(row[1]))
    ]
    for (x1, y1), (x2, y2) in zip(valid[:-1], valid[1:]):
        if (y1 - target) * (y2 - target) <= 0.0:
            if y2 == y1:
                return x2
            return float(x1 + (target - y1) * (x2 - x1) / (y2 - y1))
    return None


def _signed_value_at_x(rows: list[list[float]], x_target: float) -> float | None:
    """Linearly interpolate the signed trajectory value at ``x_target``.

    The nBEl/cBEA trajectories retain the sign of the lepton asymmetry.  This
    helper deliberately does not apply ``abs``: the sphaleron conversion
    diagnostic must be able to distinguish the signed ``Y_{B-L}`` value from
    the absolute-value quantity used by the legacy final-yield summary.
    """

    valid = [
        (float(row[0]), float(row[1]))
        for row in rows
        if len(row) >= 2
        and math.isfinite(float(row[0]))
        and math.isfinite(float(row[1]))
    ]
    if not valid:
        return None
    valid.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in valid], dtype=float)
    ys = np.asarray([item[1] for item in valid], dtype=float)
    target = float(x_target)
    if not math.isfinite(target) or target < xs[0] or target > xs[-1]:
        return None
    return float(np.interp(target, xs, ys))


def _reconstructed_ybl_at_x(
    rows: list[list[float]],
    physics: Z4Physics,
    x_target: float,
) -> float | None:
    """Return the visible-sector B-L value reconstructed from ``Y_deltaL``."""

    y_delta_l = _signed_value_at_x(rows, x_target)
    if y_delta_l is None:
        return None
    return float(physics.ybl_sm(float(x_target), y_delta_l))


def _reconstructed_ybl_rows(
    rows: list[list[float]],
    physics: Z4Physics,
) -> list[list[float]]:
    """Apply the dynamic spectator map to a saved asymmetry trajectory."""

    out: list[list[float]] = []
    for row in rows:
        if len(row) < 2:
            continue
        x_value, y_delta_l = float(row[0]), float(row[1])
        if math.isfinite(x_value) and math.isfinite(y_delta_l):
            out.append([x_value, float(physics.ybl_sm(x_value, y_delta_l))])
    return out


def _vll_epsm(params: dict[str, Any]) -> float:
    """Return the VLL/MMA reference CP-asymmetry scale ``epsm``."""

    r = float(params["r"])
    delta = float(params["delta"])
    lam = float(params["lam"])
    r2 = r / 2.0
    kinematic = 7.0 - 15.0 * r2**2 + 9.0 * r2**4 - r2**6
    return float(
        2.0 * lam**2 / (3.0 * math.pi * math.sqrt(5.0))
        * (1.0 + delta)
        / 4.0
        * kinematic
    )


def _xkd_departure_from_cbe(rows: list[list[float]], physics: Z4Physics) -> float | None:
    """Translate cBEA.wl's internal first 5-percent temperature departure xkd1."""

    for row in rows:
        if len(row) < 3:
            continue
        x, _abundance, y_temp = map(float, row[:3])
        y_eq = physics.yeq(x)
        if x > 1.0 and y_eq > 0.0 and abs(1.0 - y_temp / y_eq) >= 0.05:
            return float(x)
    return None


def _root_scan(
    func: Callable[[float], float],
    xmin: float,
    xmax: float,
    *,
    which: str = "first",
) -> float | None:
    xs = np.geomspace(max(xmin, 1.0e-6), max(xmax, xmin * 1.0001), 180)
    vals = []
    for x in xs:
        try:
            val = float(func(float(x)))
        except (ArithmeticError, ValueError, OverflowError):
            val = math.nan
        vals.append(val)
    brackets = [
        (float(xa), float(xb), float(va), float(vb))
        for xa, xb, va, vb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:])
        if math.isfinite(va) and math.isfinite(vb) and va * vb <= 0.0
    ]
    if not brackets:
        return None
    lo, hi, va, vb = brackets[-1] if which == "last" else brackets[0]
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        vm = float(func(mid))
        if va * vm <= 0.0:
            hi = mid
            vb = vm
        else:
            lo = mid
            va = vm
    return math.sqrt(lo * hi)


def _psi_bessel_ratio_k1_k2(z: float) -> float:
    """Return K1(z)/K2(z) using the shared stable Bessel approximations."""

    zz = max(float(z), 1.0e-12)
    k1 = float(bessel_k1_approx(zz))
    k2 = float(bessel_k2_approx(zz))
    return _safe_div(k1, k2, MIN_NUMBER)


def _psi_fermion_eq_density(temp: float, internal_degrees: float) -> float:
    """Massless Fermi--Dirac equilibrium density for the n_R diagnostic."""

    tt = max(float(temp), 0.0)
    return max(
        float(internal_degrees),
        0.0,
    ) * 3.0 * 1.202056903159594 / (4.0 * math.pi**2) * tt**3


def _psi_vev_at_temperature(temp: float, settings: dict[str, Any]) -> float:
    """Smooth toy crossover profile used only by the diagnostic layer."""

    tt = max(float(temp), 0.0)
    vev0 = max(float(settings.get("vEW_GeV", _PSI_VEW_GEV_DEFAULT)), 0.0)
    tc = max(float(settings.get("ewCrossover_GeV", _PSI_EW_CROSSOVER_GEV_DEFAULT)), MIN_NUMBER)
    if tt >= tc:
        return 0.0
    return vev0 * math.sqrt(max(1.0 - (tt / tc) ** 2, 0.0))


def _psi_mixing_at_temperature(
    physics: Z4Physics,
    temp: float,
    settings: dict[str, Any],
) -> dict[str, float]:
    """Evaluate the note's broken-phase neutral mixing formulas."""

    lambda_n = abs(float(settings.get("lambdaN", _PSI_LAMBDA_N_DEFAULT)))
    vev = _psi_vev_at_temperature(temp, settings)
    m_psi = float(physics.r * physics.m_dm)
    m_d = lambda_n * vev / math.sqrt(2.0)
    m_n = math.sqrt(m_psi**2 + m_d**2)
    s_theta = _safe_div(m_d, m_n, MIN_NUMBER)
    c_theta = _safe_div(m_psi, m_n, MIN_NUMBER)
    return {
        "v_GeV": float(vev),
        "mD_GeV": float(m_d),
        "MN_GeV": float(m_n),
        "sTheta": float(s_theta),
        "cTheta": float(c_theta),
        "deltaMmix_GeV": float(m_n - m_psi),
        "lambdaNuN_eff": float(physics.lam * s_theta),
    }


def _psi_ew_splitting(m_psi: float, settings: dict[str, Any]) -> float:
    """Return the one-loop charged-neutral splitting used in the note.

    This helper is intentionally independent of the cosmological solver.  The
    fixed Gauss-Legendre rule avoids adding a SciPy dependency to the runner;
    it is only evaluated when the prepared collider diagnostic is explicitly
    requested.
    """

    mass = max(float(m_psi), MIN_NUMBER)
    m_z = max(float(settings.get("mZ_GeV", _PSI_MZ_GEV_DEFAULT)), MIN_NUMBER)
    alpha_em = max(float(settings.get("alphaEM", _PSI_ALPHA_EM_DEFAULT)), 0.0)
    rho = (mass / m_z) ** 2
    nodes, weights = np.polynomial.legendre.leggauss(96)
    z_values = 0.5 * (nodes + 1.0)
    quadrature_weights = 0.5 * weights
    denominator = max(rho, MIN_NUMBER)
    logarithm = np.log1p(z_values / (denominator * (1.0 - z_values) ** 2))
    integral = float(np.sum(quadrature_weights * (2.0 - z_values) * logarithm))
    return float(alpha_em * m_z / 2.0 * math.sqrt(rho) / math.pi * integral)


def _psi_leptonic_three_body_width(delta_m: float, lepton_mass: float, settings: dict[str, Any]) -> float:
    """Return the note's leading-order width for psi -> N l nu_l."""

    gap = float(delta_m)
    mass = max(float(lepton_mass), 0.0)
    if gap <= mass or gap <= 0.0:
        return 0.0
    b = min(max(mass / gap, 0.0), 1.0 - 1.0e-15)
    b2 = b * b
    b4 = b2 * b2
    root = math.sqrt(max(1.0 - b2, 0.0))
    if b < 1.0e-6:
        phase_polynomial = 1.0 - 4.5 * b2 - 4.0 * b4
    else:
        phase_polynomial = (
            1.0
            - 4.5 * b2
            - 4.0 * b4
            + 15.0 * b4 / max(2.0 * root, MIN_NUMBER) * math.atanh(root)
        )
    gf = max(float(settings.get("GF_GeV_minus2", _PSI_GF_GEV_MINUS2_DEFAULT)), 0.0)
    return float(gf**2 / (15.0 * math.pi**3) * gap**5 * root * phase_polynomial)


def build_psi_collider_diagnostics(
    physics: Z4Physics,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Prepare collider-only Psi diagnostics without changing the scan.

    The function is deliberately not called by ``_psi_diagnostics`` or by the
    MC+CMA scanner yet.  It provides the note-level quantities that can be
    evaluated algebraically after a CMA point is selected: the EW splitting,
    the physical charged-neutral gap, the leptonic three-body widths, total
    widths, lifetimes, branching fractions, and the disappearing-track ratio.
    """

    m_psi = max(float(physics.r * physics.m_dm), 0.0)
    lambda_n = abs(float(settings.get("lambdaN", _PSI_LAMBDA_N_DEFAULT)))
    mixing = _psi_mixing_at_temperature(physics, 0.0, settings)
    m_n = max(float(mixing["MN_GeV"]), 0.0)
    delta_m_mix = float(mixing["deltaMmix_GeV"])
    delta_m_ew = _psi_ew_splitting(m_psi, settings)
    delta_m = float(delta_m_ew - delta_m_mix)
    m_c = m_n + delta_m

    v_ew = max(float(settings.get("vEW_GeV", _PSI_VEW_GEV_DEFAULT)), MIN_NUMBER)
    m_w = max(float(settings.get("mW_GeV", _PSI_MW_GEV_DEFAULT)), MIN_NUMBER)
    m_z = max(float(settings.get("mZ_GeV", _PSI_MZ_GEV_DEFAULT)), MIN_NUMBER)
    m_h = max(float(settings.get("mh_GeV", _PSI_MH_GEV_DEFAULT)), 0.0)
    g_ew = 2.0 * m_w / v_ew
    c_w = _safe_div(m_w, m_z, MIN_NUMBER)

    def vector_width(parent_mass: float, vector_mass: float, prefactor: float) -> float:
        if parent_mass <= vector_mass:
            return 0.0
        ratio = (vector_mass / parent_mass) ** 2
        return float(
            prefactor
            * parent_mass**3
            / max(vector_mass**2, MIN_NUMBER)
            * (1.0 - ratio) ** 2
            * (1.0 + 2.0 * ratio)
        )

    s_theta = float(mixing["sTheta"])
    c_theta = float(mixing["cTheta"])
    gamma_w = vector_width(m_c, m_w, g_ew**2 * s_theta**2 / (64.0 * math.pi))
    gamma_z = vector_width(
        m_n,
        m_z,
        g_ew**2 * s_theta**2 * c_theta**2 / (128.0 * math.pi * max(c_w**2, MIN_NUMBER)),
    )
    h_ratio = (m_h / max(m_n, MIN_NUMBER)) ** 2
    gamma_h = (
        0.0
        if m_n <= m_h
        else lambda_n**2 * c_theta**2 * m_n / (64.0 * math.pi) * (1.0 - h_ratio) ** 2
    )

    gf = max(float(settings.get("GF_GeV_minus2", _PSI_GF_GEV_MINUS2_DEFAULT)), 0.0)
    v_ud = abs(float(settings.get("Vud", _PSI_VUD_DEFAULT)))
    f_pi = max(float(settings.get("fPi_GeV", _PSI_FPI_GEV_DEFAULT)), 0.0)
    m_pi = max(float(settings.get("mPi_GeV", _PSI_MPI_GEV_DEFAULT)), 0.0)
    pion_open = delta_m > m_pi
    if pion_open:
        pion_phase = math.sqrt(max(1.0 - (m_pi / delta_m) ** 2, 0.0))
        gamma_pion = gf**2 * v_ud**2 * f_pi**2 / math.pi * delta_m**3 * pion_phase
    else:
        gamma_pion = 0.0

    gamma_e_nu = _psi_leptonic_three_body_width(
        delta_m,
        max(float(settings.get("mElectron_GeV", _PSI_ME_GEV_DEFAULT)), 0.0),
        settings,
    )
    gamma_mu_nu = _psi_leptonic_three_body_width(
        delta_m,
        max(float(settings.get("mMuon_GeV", _PSI_MMU_GEV_DEFAULT)), 0.0),
        settings,
    )
    charged_total = gamma_w + gamma_pion + gamma_e_nu + gamma_mu_nu
    neutral_total = gamma_z + gamma_h
    cascade_total = gamma_pion + gamma_e_nu + gamma_mu_nu

    def branching(width: float, total: float) -> float:
        return 0.0 if total <= 0.0 else float(width / total)

    return {
        "status": "prepared_not_wired",
        "mPsi_GeV": float(m_psi),
        "mD_zeroT_GeV": float(mixing["mD_GeV"]),
        "MN_zeroT_GeV": float(m_n),
        "MC_GeV": float(m_c),
        "sTheta_zeroT": float(s_theta),
        "cTheta_zeroT": float(c_theta),
        "deltaMmix_zeroT_GeV": delta_m_mix,
        "deltaM_EW_GeV": float(delta_m_ew),
        "deltaM_charged_minus_neutral_GeV": float(delta_m),
        "threshold_Wn_open": bool(m_c > m_w),
        "threshold_Zn_open": bool(m_n > m_z),
        "threshold_hn_open": bool(m_n > m_h),
        "gamma_Wn_GeV": float(gamma_w),
        "gamma_Zn_GeV": float(gamma_z),
        "gamma_hn_GeV": float(gamma_h),
        "gamma_pion_GeV": float(gamma_pion),
        "gamma_e_nu_GeV": float(gamma_e_nu),
        "gamma_mu_nu_GeV": float(gamma_mu_nu),
        "gamma_charged_total_GeV": float(charged_total),
        "gamma_neutral_total_GeV": float(neutral_total),
        "ctau_charged_mm": (
            math.inf if charged_total <= 0.0 else float(1.97327e-13 / charged_total)
        ),
        "ctau_neutral_mm": (
            math.inf if neutral_total <= 0.0 else float(1.97327e-13 / neutral_total)
        ),
        "Br_Wn": branching(gamma_w, charged_total),
        "Br_pion": branching(gamma_pion, charged_total),
        "Br_e_nu": branching(gamma_e_nu, charged_total),
        "Br_mu_nu": branching(gamma_mu_nu, charged_total),
        "Br_Zn": branching(gamma_z, neutral_total),
        "Br_hn": branching(gamma_h, neutral_total),
        "pion_open": bool(pion_open),
        "R_DT": math.inf if cascade_total <= 0.0 else float(gamma_w / cascade_total),
        "formula_scope": "collider-only; not wired into nBE/cBE or MC+CMA",
    }


def _psi_diagnostics(
    physics: Z4Physics,
    rates: Z4RateSet,
    nbe: dict[str, Any],
    cbe: dict[str, Any],
    lep: dict[str, Any],
    settings: dict[str, Any],
    xstar: float,
) -> dict[str, Any]:
    """Evaluate note-level Psi--n_R checks once after the cosmological solve.

    The diagnostics are intentionally separated from the nBE/cBE state
    evolution.  All expensive cosmological quantities are reused from the
    accepted trajectories; only the massless-bath light rate is integrated at
    the single kinetic-decoupling point when the broken phase is active.
    """

    if not _as_bool(settings.get("psiDiagnostics"), True):
        return {"enabled": False, "status": "disabled_by_settings"}

    m_psi = float(physics.r * physics.m_dm)
    lambda_n = abs(float(settings.get("lambdaN", _PSI_LAMBDA_N_DEFAULT)))
    m_h_unbroken = max(float(settings.get("mHUnbroken_GeV", _PSI_MH_UNBROKEN_GEV_DEFAULT)), 0.0)
    decay_gate = max(float(settings.get("psiDecayGate", _PSI_DECAY_GATE_DEFAULT)), 0.0)
    light_gate = max(float(settings.get("psiLightGate", _PSI_LIGHT_GATE_DEFAULT)), 0.0)
    g_nr = max(float(settings.get("nRInternalDegrees", _PSI_NR_INTERNAL_DEGREES_DEFAULT)), 0.0)
    weak_factor = _ANNIHILATION_WEAK_COMPONENT_FACTOR
    charge_factor = _ANNIHILATION_CHARGE_CONJUGATE_FACTOR
    spin_factor = max(float(getattr(physics, "g_psi", 2.0)), 1.0)

    phase_decay = max(1.0 - (m_h_unbroken / max(m_psi, MIN_NUMBER)) ** 2, 0.0)
    gamma_rest = lambda_n**2 * m_psi / (32.0 * math.pi) * phase_decay**2

    def gamma_decay_thermal(x: float) -> float:
        temp = physics.temperature(float(x))
        return gamma_rest * _psi_bessel_ratio_k1_k2(m_psi / max(temp, MIN_NUMBER))

    def gamma_decay_density(x: float) -> float:
        temp = physics.temperature(float(x))
        n_psi_one = physics.neq(m_psi, m_psi / max(temp, MIN_NUMBER))
        return weak_factor * charge_factor * spin_factor * n_psi_one * gamma_decay_thermal(x)

    x_rows = np.asarray(nbe["xy"], dtype=float)
    x_min = max(float(x_rows[0, 0]), 1.0e-8)
    x_max = max(float(x_rows[-1, 0]), x_min * 1.0001)
    x_lep_start = float(lep.get("xw") or x_min)
    x_lep_end = float(lep.get("xlep") or xstar or x_max)
    x_lep_start = min(max(x_lep_start, x_min), x_max)
    x_lep_end = min(max(x_lep_end, x_lep_start), x_max)
    if x_lep_end > x_lep_start * 1.0001:
        x_decay_grid = np.geomspace(x_lep_start, x_lep_end, 48)
    else:
        x_decay_grid = np.asarray([x_lep_start], dtype=float)
    decay_ratios = np.asarray(
        [gamma_decay_thermal(float(x)) / max(physics.hubble(float(x)), MIN_NUMBER) for x in x_decay_grid],
        dtype=float,
    )
    min_decay_index = int(np.argmin(decay_ratios))
    min_decay_ratio = float(decay_ratios[min_decay_index])
    min_decay_x = float(x_decay_grid[min_decay_index])

    def gamma_inverse_decay(x: float) -> float:
        temp = physics.temperature(float(x))
        n_nr = _psi_fermion_eq_density(temp, g_nr)
        return _safe_div(gamma_decay_density(float(x)), n_nr, MIN_NUMBER)

    inverse_decay_residual = lambda x: gamma_inverse_decay(x) - physics.hubble(x)
    # The first crossing is thermalisation as the Universe cools, while the
    # last crossing is the late inverse-decay decoupling point.  They are not
    # interchangeable when Psi is still relativistic at the first crossing.
    x_nr_therm = _root_scan(inverse_decay_residual, x_min, x_max, which="first")
    x_nr_dec = _root_scan(inverse_decay_residual, x_min, x_max, which="last")
    t_nr_therm = None if x_nr_therm is None else float(physics.temperature(x_nr_therm))
    t_nr_dec = None if x_nr_dec is None else float(physics.temperature(x_nr_dec))
    gamma_d_at_nr_dec = None if x_nr_dec is None else float(gamma_decay_density(x_nr_dec))
    gamma_id_at_nr_dec = None if x_nr_dec is None else float(gamma_inverse_decay(x_nr_dec))
    hubble_at_nr_dec = None if x_nr_dec is None else float(physics.hubble(x_nr_dec))
    gstar_s_nr_dec = None if t_nr_dec is None else float(physics.dof.iheff(t_nr_dec))
    t_nu_dec = max(float(settings.get("TnuDec_GeV", _PSI_TNU_DECOUPLING_GEV_DEFAULT)), MIN_NUMBER)
    gstar_s_nu_dec = max(float(settings.get("gStarSNuDec", _PSI_GSTAR_S_NU_DECOUPLING_DEFAULT)), MIN_NUMBER)
    delta_neff = (
        None
        if gstar_s_nr_dec is None
        else (g_nr / 2.0) * (gstar_s_nu_dec / max(gstar_s_nr_dec, MIN_NUMBER)) ** (4.0 / 3.0)
    )

    x_kd = lep.get("xkd")
    if x_kd is None or not math.isfinite(float(x_kd)) or float(x_kd) <= 0.0:
        x_kd = None
        temp_kd = None
        mixing_kd = {
            "v_GeV": 0.0,
            "mD_GeV": 0.0,
            "MN_GeV": m_psi,
            "sTheta": 0.0,
            "cTheta": 1.0,
            "deltaMmix_GeV": 0.0,
            "lambdaNuN_eff": 0.0,
        }
        gamma_light = 0.0
        r_light = 0.0
        light_rate_mode = "not_evaluated_no_xkd"
    else:
        x_kd = float(x_kd)
        temp_kd = float(physics.temperature(x_kd))
        mixing_kd = _psi_mixing_at_temperature(physics, temp_kd, settings)
        s_theta_kd = mixing_kd["sTheta"]
        if s_theta_kd <= 0.0:
            gamma_light = 0.0
            r_light = 0.0
            light_rate_mode = "broken_phase_closed_v_equals_zero"
        else:
            # One neutral bath component and one charge channel are used in
            # the microscopic rate.  The two charge-conjugate channels are
            # then summed externally; there is no extra 2_weak here.
            n_neutral_one = _psi_fermion_eq_density(temp_kd, 1.0)
            light_rate_mode_setting = str(settings.get("psiLightRateMode", "exact")).strip().lower()
            if light_rate_mode_setting in {"proxy", "fast", "scan"}:
                assert rates.svxl1 is not None
                sv_light = float(rates.svxl1(x_kd))
                light_rate_mode = "proxy_existing_svxl1"
            else:
                sv_light = physics.thermal_svxl1_massless_bath(x_kd)
                light_rate_mode = "exact_massless_bath_single_xkd_point"
            gamma_light = charge_factor * s_theta_kd**2 * n_neutral_one * sv_light
            h_tilde = physics.hubble(x_kd) * (1.0 + physics.gtilde(x_kd))
            r_light = _safe_div(gamma_light, h_tilde, MIN_NUMBER)

    zero_temp_mixing = _psi_mixing_at_temperature(physics, 0.0, settings)
    v_ew = max(float(settings.get("vEW_GeV", _PSI_VEW_GEV_DEFAULT)), 0.0)
    m_w = max(float(settings.get("mW_GeV", _PSI_MW_GEV_DEFAULT)), MIN_NUMBER)
    m_z = max(float(settings.get("mZ_GeV", _PSI_MZ_GEV_DEFAULT)), MIN_NUMBER)
    m_h = max(float(settings.get("mh_GeV", _PSI_MH_GEV_DEFAULT)), 0.0)
    g_ew = 2.0 * m_w / max(v_ew, MIN_NUMBER)
    c_w = _safe_div(m_w, m_z, MIN_NUMBER)
    m_c = m_psi
    m_n = zero_temp_mixing["MN_GeV"]
    s_theta = zero_temp_mixing["sTheta"]
    c_theta = zero_temp_mixing["cTheta"]

    def vector_width(parent_mass: float, vector_mass: float, prefactor: float) -> float:
        if parent_mass <= vector_mass:
            return 0.0
        ratio = (vector_mass / parent_mass) ** 2
        return prefactor * parent_mass**3 / max(vector_mass**2, MIN_NUMBER) * (1.0 - ratio) ** 2 * (1.0 + 2.0 * ratio)

    gamma_w = vector_width(m_c, m_w, g_ew**2 * s_theta**2 / (64.0 * math.pi))
    gamma_z = vector_width(
        m_n,
        m_z,
        g_ew**2 * s_theta**2 * c_theta**2 / (128.0 * math.pi * max(c_w**2, MIN_NUMBER)),
    )
    h_ratio = (m_h / max(m_n, MIN_NUMBER)) ** 2
    gamma_h = 0.0 if m_n <= m_h else lambda_n**2 * c_theta**2 * m_n / (64.0 * math.pi) * (1.0 - h_ratio) ** 2

    gf = max(float(settings.get("GF_GeV_minus2", _PSI_GF_GEV_MINUS2_DEFAULT)), 0.0)
    v_ud = abs(float(settings.get("Vud", _PSI_VUD_DEFAULT)))
    f_pi = max(float(settings.get("fPi_GeV", _PSI_FPI_GEV_DEFAULT)), 0.0)
    m_pi = max(float(settings.get("mPi_GeV", _PSI_MPI_GEV_DEFAULT)), 0.0)
    delta_m = m_c - m_n
    if delta_m > m_pi:
        pion_phase = math.sqrt(max(1.0 - (m_pi / delta_m) ** 2, 0.0))
        gamma_pion = gf**2 * v_ud**2 * f_pi**2 / math.pi * delta_m**3 * pion_phase
    else:
        gamma_pion = 0.0

    charged_total = gamma_w + gamma_pion
    neutral_total = gamma_z + gamma_h
    ctau_charged_mm = _safe_div(1.97327e-13, charged_total, MIN_NUMBER) if charged_total > 0.0 else math.inf
    ctau_neutral_mm = _safe_div(1.97327e-13, neutral_total, MIN_NUMBER) if neutral_total > 0.0 else math.inf
    pion_open = delta_m > m_pi
    pion_dominance_pass = (not pion_open) or gamma_w > 10.0 * max(gamma_pion, MIN_NUMBER)

    # Late energy injection into the decoupled n_R bath.  This is deliberately
    # a separate diagnostic: it does not feed back into the nBE/cBE evolution.
    # The parent abundance is the equilibrium estimate, so the result is
    # labelled as such rather than being called a rigorous upper bound.
    late_x_start = x_nr_dec
    late_x_end = x_max
    late_grid_size = max(int(settings.get("psiLateGridSize", 192)), 16)
    late_comoving_energy = 0.0
    late_q_peak = 0.0
    r_late = 0.0
    delta_neff_late = 0.0
    delta_neff_total = delta_neff
    max_rho_n_over_rho_rad = 0.0
    x_at_max_rho_n_over_rho_rad = None
    max_delta_h_over_h = 0.0
    x_at_max_delta_h_over_h = None
    thermal_comoving_energy = None

    def two_body_massless_energy(parent_mass: float, visible_mass: float) -> float:
        parent = max(float(parent_mass), MIN_NUMBER)
        return max(parent**2 - max(float(visible_mass), 0.0) ** 2, 0.0) / (2.0 * parent)

    if late_x_start is not None and late_x_start < late_x_end:
        late_x_grid = np.geomspace(float(late_x_start), float(late_x_end), late_grid_size)
        late_x_grid = np.unique(np.concatenate(([float(late_x_start)], late_x_grid)))
        q_over_xh_s43 = np.zeros_like(late_x_grid)
        q_values = np.zeros_like(late_x_grid)
        e_w = two_body_massless_energy(m_c, m_w)
        e_z = two_body_massless_energy(m_n, m_z)
        e_h = two_body_massless_energy(m_n, m_h)
        for index, x_value in enumerate(late_x_grid):
            temp = float(physics.temperature(float(x_value)))
            n_psi_one = physics.neq(m_psi, m_psi / max(temp, MIN_NUMBER))
            if temp >= max(float(settings.get("ewCrossover_GeV", _PSI_EW_CROSSOVER_GEV_DEFAULT)), MIN_NUMBER):
                # In the unbroken phase the boost factor in the decay rate and
                # the average daughter-energy boost cancel in Q_n.
                q_n = weak_factor * charge_factor * spin_factor * n_psi_one * gamma_rest * two_body_massless_energy(m_psi, m_h_unbroken)
            else:
                # Below the crossover, sum the physical n-producing W/Z/h
                # channels over the two weak components; the pion cascade does
                # not inject a light n directly and is excluded here.
                q_n = charge_factor * spin_factor * n_psi_one * (gamma_w * e_w + gamma_z * e_z + gamma_h * e_h)
            entropy_density = max(float(physics.entropy(float(x_value))), MIN_NUMBER)
            h_tilde = physics.hubble(float(x_value)) * (1.0 + physics.gtilde(float(x_value)))
            q_values[index] = max(float(q_n), 0.0)
            q_over_xh_s43[index] = q_values[index] / max(
                float(x_value) * h_tilde * entropy_density ** (4.0 / 3.0),
                MIN_NUMBER,
            )
        increments = 0.5 * (q_over_xh_s43[1:] + q_over_xh_s43[:-1]) * np.diff(late_x_grid)
        cumulative_late = np.concatenate(([0.0], np.cumsum(increments)))
        late_comoving_energy = float(cumulative_late[-1])
        late_q_peak = float(np.max(q_values))
        temp_nr_dec = max(float(t_nr_dec), MIN_NUMBER)
        entropy_nr_dec = max(float(physics.entropy(float(x_nr_dec))), MIN_NUMBER)
        rho_n_thermal_nr_dec = (7.0 / 8.0) * (math.pi**2 / 30.0) * g_nr * temp_nr_dec**4
        thermal_comoving_energy = rho_n_thermal_nr_dec / entropy_nr_dec ** (4.0 / 3.0)
        r_late = _safe_div(late_comoving_energy, thermal_comoving_energy, MIN_NUMBER)
        delta_neff_late = 0.0 if delta_neff is None else float(r_late * delta_neff)
        delta_neff_total = None if delta_neff is None else float(delta_neff + delta_neff_late)

        # Track the extra radiation and the corresponding exact Hubble shift
        # from T_n=T before decoupling and entropy-redshifted T_n afterwards.
        energy_x_grid = np.geomspace(x_min, x_max, late_grid_size)
        energy_x_grid = np.unique(np.concatenate((energy_x_grid, [float(x_nr_dec)])))
        cumulative_energy = np.interp(energy_x_grid, late_x_grid, cumulative_late, left=0.0, right=late_comoving_energy)
        gstar_s_nr_dec = max(float(physics.dof.iheff(temp_nr_dec)), MIN_NUMBER)
        ratios = []
        hubble_shifts = []
        for x_value, late_comoving in zip(energy_x_grid, cumulative_energy):
            temp = max(float(physics.temperature(float(x_value))), MIN_NUMBER)
            gstar_s = max(float(physics.dof.iheff(temp)), MIN_NUMBER)
            t_n = temp if x_value <= float(x_nr_dec) else temp * (gstar_s / gstar_s_nr_dec) ** (1.0 / 3.0)
            rho_n_thermal = (7.0 / 8.0) * (math.pi**2 / 30.0) * g_nr * t_n**4
            entropy_density = max(float(physics.entropy(float(x_value))), MIN_NUMBER)
            rho_n_total = rho_n_thermal + float(late_comoving) * entropy_density ** (4.0 / 3.0)
            g_eff = max(float(physics.dof.isqrtgeff(temp)) ** 2, MIN_NUMBER)
            rho_rad = (math.pi**2 / 30.0) * g_eff * temp**4
            ratio = _safe_div(rho_n_total, rho_rad, MIN_NUMBER)
            ratios.append(ratio)
            hubble_shifts.append(math.sqrt(max(1.0 + ratio, 1.0)) - 1.0)
        max_ratio_index = int(np.argmax(ratios))
        max_hubble_index = int(np.argmax(hubble_shifts))
        max_rho_n_over_rho_rad = float(ratios[max_ratio_index])
        x_at_max_rho_n_over_rho_rad = float(energy_x_grid[max_ratio_index])
        max_delta_h_over_h = float(hubble_shifts[max_hubble_index])
        x_at_max_delta_h_over_h = float(energy_x_grid[max_hubble_index])

    ybl_conservation = {
        "available": False,
        "residual": None,
        "max_abs_residual": None,
        "definition": "Y_(B-L)^SM - Y_DeltaPsi - Y_Deltan = constant",
        "reason": "The active nBE/cBE state does not independently evolve Y_DeltaPsi or Y_Deltan; a reconstructed residual would be tautological.",
    }
    late_gate_threshold = max(float(settings.get("psiLateGate", 0.1)), 0.0)
    hubble_shift_gate_threshold = max(float(settings.get("psiHubbleShiftGate", 0.01)), 0.0)
    late_gate_pass = bool(r_late < late_gate_threshold)
    hubble_shift_gate_pass = bool(max_delta_h_over_h < hubble_shift_gate_threshold)
    complete_consistency_gates_pass = bool(
        min_decay_ratio > decay_gate
        and r_light < light_gate
        and pion_dominance_pass
        and late_gate_pass
        and hubble_shift_gate_pass
        and ybl_conservation["available"]
    )
    return {
        "enabled": True,
        "status": "evaluated",
        "lambdaN": lambda_n,
        "mPsi_GeV": m_psi,
        "decay": {
            "width_one_component_GeV": float(gamma_rest),
            "phase_space_factor": float(phase_decay),
            "weak_component_factor": weak_factor,
            "charge_conjugate_factor": charge_factor,
            "spin_internal_factor": spin_factor,
            "total_decay_density_definition": "gPsi_spin*2_weak*2_charge*nPsi_one*<GammaPsi_one>",
            "x_leptogenesis_start": x_lep_start,
            "x_leptogenesis_end": x_lep_end,
            "min_thermal_width_over_H": min_decay_ratio,
            "x_at_min_thermal_width_over_H": min_decay_x,
            "gate_threshold": decay_gate,
            "gate_pass": bool(min_decay_ratio > decay_gate),
        },
        "inverse_decay": {
            "nR_internal_degrees": g_nr,
            "gamma_D_total_at_x_n_dec_GeV4": gamma_d_at_nr_dec,
            "Gamma_ID_at_x_n_dec_GeV": gamma_id_at_nr_dec,
            "H_at_x_n_dec_GeV": hubble_at_nr_dec,
            "Gamma_ID_over_H_at_x_n_dec": (
                None
                if gamma_id_at_nr_dec is None or hubble_at_nr_dec is None
                else _safe_div(gamma_id_at_nr_dec, hubble_at_nr_dec, MIN_NUMBER)
            ),
            "x_n_dec": x_nr_dec,
            "T_n_dec_GeV": t_nr_dec,
            "x_n_therm": x_nr_therm,
            "T_n_therm_GeV": t_nr_therm,
            "gStarS_at_T_n_dec": gstar_s_nr_dec,
            "T_nu_dec_reference_GeV": t_nu_dec,
            "gStarS_at_T_nu_dec_reference": gstar_s_nu_dec,
            "DeltaNeff": delta_neff,
            "DeltaNeff_definition": "(g_nR/2)*(gStarS(T_nu_dec)/gStarS(T_n_dec))^(4/3), with g_nR counting n_R plus anti-n_R",
            "condition": "gamma_D_total/n_n_eq = H",
        },
        "late_injection": {
            "x_start": late_x_start,
            "x_end": late_x_end,
            "equilibrium_parent_estimate": True,
            "late_comoving_energy": float(late_comoving_energy),
            "thermal_comoving_energy_at_x_n_dec": thermal_comoving_energy,
            "R_late": float(r_late),
            "DeltaNeff_late": float(delta_neff_late),
            "DeltaNeff_total": delta_neff_total,
            "Q_n_peak_GeV5": float(late_q_peak),
            "max_rho_n_over_rho_rad": float(max_rho_n_over_rho_rad),
            "x_at_max_rho_n_over_rho_rad": x_at_max_rho_n_over_rho_rad,
            "max_delta_H_over_H": float(max_delta_h_over_h),
            "x_at_max_delta_H_over_H": x_at_max_delta_h_over_h,
            "late_gate_threshold": late_gate_threshold,
            "late_gate_pass": late_gate_pass,
            "hubble_shift_gate_threshold": hubble_shift_gate_threshold,
            "hubble_shift_gate_pass": hubble_shift_gate_pass,
            "formula": "d(rho_n/s^(4/3))/dx = Q_n/[x*Htilde*s^(4/3)]",
        },
        "YBL_conservation": ybl_conservation,
        "mixing_at_xkd": {
            "xkd": x_kd,
            "Tkd_GeV": temp_kd,
            **mixing_kd,
            "R_light": float(r_light),
            "gamma_light_GeV": float(gamma_light),
            "light_gate_threshold": light_gate,
            "light_gate_pass": bool(r_light < light_gate),
            "rate_mode": light_rate_mode,
            "rate_definition": "2_charge*sTheta^2*n_neutral_one*<sigma v>_massless_bath at xkd",
        },
        "mixing_zero_temperature": zero_temp_mixing,
        "collider_widths": {
            "gamma_Wn_GeV": float(gamma_w),
            "gamma_Zn_GeV": float(gamma_z),
            "gamma_hn_GeV": float(gamma_h),
            "gamma_pion_GeV": float(gamma_pion),
            "deltaM_charged_minus_neutral_GeV": float(delta_m),
            "pion_open": bool(pion_open),
            "W_to_pion_gate_pass": bool(pion_dominance_pass),
            "ctau_charged_mm": float(ctau_charged_mm),
            "ctau_neutral_mm": float(ctau_neutral_mm),
            "individual_state_widths": True,
            "goldstone_ratio_W_Z_h": [float(gamma_w), float(gamma_z), float(gamma_h)],
            "W_over_Z": _safe_div(gamma_w, gamma_z, MIN_NUMBER),
            "Z_over_h": _safe_div(gamma_z, gamma_h, MIN_NUMBER),
        },
        "all_gates_pass": bool(min_decay_ratio > decay_gate and r_light < light_gate and pion_dominance_pass),
        "all_consistency_gates_pass": complete_consistency_gates_pass,
        "formula_scope": "diagnostic-only; late injection uses an equilibrium-Psi estimate; YBL_SM is reconstructed from evolved YDeltaL",
    }


def _lep_diagnostics(physics: Z4Physics, rates: Z4RateSet, nbe: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    assert rates.svx is not None and rates.svxl is not None and rates.svxl2 is not None
    rows = nbe["xy"]
    x_min, x_max = float(rows[0][0]), float(rows[-1][0])

    def gamma_a(x: float) -> float:
        # ``nchi_eq`` is the total X + Xbar density.  The annihilation
        # channel multiplicity is defined for one X density, so convert back
        # by gDM before applying the weak and charge-channel factors.
        n_single_x_eq = physics.nchi_eq(x) / max(float(physics.g_dm), 1.0)
        return _ANNIHILATION_TOTAL_FACTOR * n_single_x_eq * float(rates.svx(x))

    def gamma_w(x: float) -> float:
        return physics.entropy(x) * float(rates.svxl(x)) * physics.YLeq(x) * physics.Ypeq(x) / physics.Yg(x)

    def gamma_wx(x: float) -> float:
        return physics.entropy(x) * float(rates.svx(x)) * physics.Yeq(x) ** 2 / physics.Yg(x)

    def gamma_washout_eff(x: float) -> float:
        eta_inv = _safe_div(1.0, physics.eta(x), MIN_NUMBER)
        return physics.ksi(x) * eta_inv * (
            gamma_w(x)
            + gamma_wx(x) * (1.0 + y_interp(x) / max(physics.Yeq(x), MIN_NUMBER))
            + physics.entropy(x) * float(rates.svxl2(x)) * physics.Ypeq(x) ** 2 / physics.Yg(x)
        )

    xc = _root_scan(lambda x: gamma_a(x) - physics.hubble(x), x_min, x_max)
    nbe_interp = np.asarray(rows, dtype=float)

    def y_interp(x: float) -> float:
        return float(np.interp(x, nbe_interp[:, 0], nbe_interp[:, 1]))

    xw = _root_scan(lambda x: gamma_washout_eff(x) - physics.hubble(x), x_min, x_max)
    ydl_rows = nbe["xy_ydl"]
    ybl_rows = _reconstructed_ybl_rows(ydl_rows, physics)
    final_ybl = abs(float(ybl_rows[-1][1])) if ybl_rows else 0.0
    x_after = xw if xw is not None else x_min
    after = [abs(float(row[1])) for row in ybl_rows if row[0] > x_after]
    xlep = None
    if after and final_ybl > 0.0:
        target = 1.1 * final_ybl if after[0] >= final_ybl else 0.9 * final_ybl
        xlep = _find_crossing(ybl_rows, target, x_after)
        if xlep is None and target > final_ybl:
            xlep = _find_crossing(ybl_rows, 0.9 * final_ybl, x_after)
    return {"xc": xc, "xw": xw, "xlep": xlep if xlep is not None else xw, "definition": "translated diagnostics from main_cbe_VLL.wls"}


def run_z4_from_cards(
    model_name: str,
    cards: dict[str, Any],
    dof_file: str | Path,
    out_path: str | Path,
    compare_mma_txt: str | Path | None = None,
    inputs_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    settings = _settings_with_defaults(cards)
    params = Z4Model().prepare_params(cards)
    if str(model_name).strip().lower() not in {"z4", "vll"}:
        raise ValueError(f"Expected Z4/VLL model, got {model_name}")
    dof = DofTable.from_file(dof_file)
    physics = Z4Physics(params, settings, dof)
    mpsi_gev = float(physics.r * physics.m_dm)
    resonance = _resonance_C(physics)
    mphi_gev = resonance["mPhi_GeV"]
    t_rates0 = time.perf_counter()
    rates = physics.build_rate_set()
    t_rates = time.perf_counter() - t_rates0
    nbe = solve_nbel(physics, rates, settings)
    t_nbe = time.perf_counter() - t_rates0 - t_rates
    cbe = solve_cbea(physics, rates, settings)
    t_cbe = time.perf_counter() - t_rates0 - t_rates - t_nbe
    lep = _lep_diagnostics(physics, rates, nbe, settings)
    tstar_gev = float(settings.get("sphaleronTstarGeV", _SPHALERON_TSTAR_GEV_DEFAULT))
    if not math.isfinite(tstar_gev) or tstar_gev <= 0.0:
        raise ValueError("sphaleronTstarGeV must be a finite positive temperature in GeV")
    xstar = float(physics.m_dm / tstar_gev)
    ydelta_l_nbe_xstar = _signed_value_at_x(nbe["xy_ydl"], xstar)
    ydelta_l_cbe_xstar = _signed_value_at_x(cbe["xy_ydl"], xstar)
    ybl_nbe_xstar = _reconstructed_ybl_at_x(nbe["xy_ydl"], physics, xstar)
    ybl_cbe_xstar = _reconstructed_ybl_at_x(cbe["xy_ydl"], physics, xstar)
    spectator_ratio_xstar = float(physics.spectator_ratio(xstar))
    xi_xstar = float(physics.ksi(xstar))
    eta_xstar = float(physics.eta(xstar))
    xi_over_eta_xstar = _safe_div(xi_xstar, eta_xstar, MIN_NUMBER)
    ydelta_l_nbe_final_signed = float(nbe["xy_ydl"][-1][1])
    ydelta_l_cbe_final_signed = float(cbe["xy_ydl"][-1][1])
    ybl_nbe_final_signed = float(physics.ybl_sm(nbe["xy_ydl"][-1][0], ydelta_l_nbe_final_signed))
    ybl_cbe_final_signed = float(physics.ybl_sm(cbe["xy_ydl"][-1][0], ydelta_l_cbe_final_signed))
    xlep_value = lep.get("xlep")
    xlep = None
    if xlep_value is not None:
        try:
            xlep_candidate = float(xlep_value)
            if math.isfinite(xlep_candidate):
                xlep = xlep_candidate
        except (TypeError, ValueError):
            pass
    ydelta_l_nbe_xlep = _signed_value_at_x(nbe["xy_ydl"], xlep) if xlep is not None else None
    ydelta_l_cbe_xlep = _signed_value_at_x(cbe["xy_ydl"], xlep) if xlep is not None else None
    ybl_nbe_xlep = _reconstructed_ybl_at_x(nbe["xy_ydl"], physics, xlep) if xlep is not None else None
    ybl_cbe_xlep = _reconstructed_ybl_at_x(cbe["xy_ydl"], physics, xlep) if xlep is not None else None
    eps = _corrected_eps_from_ybl(ybl_cbe_xstar, float(params["eps1"]))
    epsm = _vll_epsm(params)
    lep.update(
        {
            "Tstar_GeV": tstar_gev,
            "xstar": xstar,
            "mPsi_GeV": mpsi_gev,
            "mPhi_GeV": mphi_gev,
            "GammaPhi_GeV": resonance["GammaPhi_GeV"],
            "GammaPhi_bar": resonance["GammaPhi_bar"],
            "B": resonance["B"],
            "C": resonance["C"],
            "YDeltaL_nbe_xlep": ydelta_l_nbe_xlep,
            "YDeltaL_cbe_xlep": ydelta_l_cbe_xlep,
            "YDeltaL_nbe_xstar": ydelta_l_nbe_xstar,
            "YDeltaL_cbe_xstar": ydelta_l_cbe_xstar,
            "YDeltaL_nbe_final": ydelta_l_nbe_final_signed,
            "YDeltaL_cbe_final": ydelta_l_cbe_final_signed,
            "YBL_nbe_xlep": ybl_nbe_xlep,
            "YBL_cbe_xlep": ybl_cbe_xlep,
            "YBL_nbe_xstar": ybl_nbe_xstar,
            "YBL_cbe_xstar": ybl_cbe_xstar,
            "Rpsi_xstar": spectator_ratio_xstar,
            "xi_xstar": xi_xstar,
            "eta_xstar": eta_xstar,
            "xi_over_eta_xstar": xi_over_eta_xstar,
            "YBL_nbe_final": ybl_nbe_final_signed,
            "YBL_cbe_final": ybl_cbe_final_signed,
            "YBL_observed": _Y_BL_OBSERVED,
            "eps": eps,
            "eps_definition": (
                "eps1*abs(YBL_observed)/abs(YBL_cBE(xstar)); theory-corrected "
                "CP asymmetry required to reach the observed B-L magnitude"
            ),
            "YBL_xstar_definition": (
                "Y_{B-L}^{SM}(xstar) reconstructed from signed Y_{Delta L}(xstar) "
                "using the dynamic spectator map; no absolute value applied"
            ),
            "YBL_xstar_in_range": ybl_nbe_xstar is not None and ybl_cbe_xstar is not None,
        }
    )
    lep["xkd"] = _root_from_table(cbe["gamma_thermal_over_h"], 1.0)
    lep["xkd_departure"] = _xkd_departure_from_cbe(cbe["xy"], physics)
    lep["xkd_definition"] = (
        "MMA main_cbe_VLL.wls xk=xkdScaJac: first interpolated root of "
        "tGammaThOverHFull=gamma_conv/tilde H=1, including the independent "
        "charge-conjugate conversion channel; the separate xkd_departure is "
        "the cBEA.wl xkd1 first accepted point with abs(1-y/yeq)>=0.05"
    )
    psi_checks = _psi_diagnostics(
        physics,
        rates,
        nbe,
        cbe,
        lep,
        settings,
        xstar,
    )
    mma = _parse_mma_result(Path(compare_mma_txt) if compare_mma_txt else None)
    ref_nbe = mma["Oh2_nBE"]
    ref_cbe = mma["Oh2_cBE"]
    nbe_rel = None if ref_nbe in (None, 0.0) else abs(nbe["oh2"] - ref_nbe) / abs(ref_nbe)
    cbe_rel = None if ref_cbe in (None, 0.0) else abs(cbe["oh2"] - ref_cbe) / abs(ref_cbe)
    total = time.perf_counter() - t0
    assert rates.tables is not None
    rate_backend = str((rates.timings or {}).get("backend", "python"))
    mma_observables = {
        "r": float(params["r"]),
        "mDM": float(params["mDM"]),
        "mDM_GeV": float(params["mDM"]),
        "delta": float(params["delta"]),
        "lam": float(params["lam"]),
        "Oh2_nBE": float(nbe["oh2"]),
        "Oh2_cBE": float(cbe["oh2"]),
        "xc": lep.get("xc"),
        "xk": lep.get("xkd"),
        "xw": lep.get("xw"),
        "y": float(params["y"]),
        "Ydl": ydelta_l_nbe_final_signed,
        "cYdl": ydelta_l_cbe_final_signed,
        "Ydl_nBE_signed_final": ydelta_l_nbe_final_signed,
        "cYdl_cBE_signed_final": ydelta_l_cbe_final_signed,
        "YDeltaL_nBE_signed_final": ydelta_l_nbe_final_signed,
        "YDeltaL_cBE_signed_final": ydelta_l_cbe_final_signed,
        "Ydl_nBE_abs_final": float(nbe["Ydl"]),
        "cYdl_cBE_abs_final": float(cbe["Ydl"]),
        "YBL_nBE_signed_final": ybl_nbe_final_signed,
        "YBL_cBE_signed_final": ybl_cbe_final_signed,
        "eps1": float(params["eps1"]),
        "epsm": epsm,
        "xlep": lep.get("xlep"),
        "Tstar_GeV": tstar_gev,
        "xstar": xstar,
        "mPsi_GeV": mpsi_gev,
        "mPhi_GeV": mphi_gev,
        "GammaPhi_GeV": resonance["GammaPhi_GeV"],
        "GammaPhi_bar": resonance["GammaPhi_bar"],
        "B": resonance["B"],
        "C": resonance["C"],
        "YBL_nBE_xlep": ybl_nbe_xlep,
        "YBL_cBE_xlep": ybl_cbe_xlep,
        "YBL_nBE_xstar": ybl_nbe_xstar,
        "YBL_cBE_xstar": ybl_cbe_xstar,
        "YBL_observed": _Y_BL_OBSERVED,
        "eps": eps,
        "excluded_small_scale": ["Mkd", "Mfs1", "ks", "dak", "zzl1"],
        "definition": (
            "MMA-compatible non-small-scale BP observables; YBL *_signed fields "
            "retain the trajectory sign, while *_abs fields are legacy summaries"
        ),
    }
    sample_x = [1.0, 3.0, 10.0, 30.0, 100.0]
    out: dict[str, Any] = {
        "inputs": {
            "model": "Z4",
            "requested_model": model_name,
            "dof_file": str(dof_file),
            "runner": str(Path(__file__).resolve()),
            "source_files": SOURCE_FILES,
            **(inputs_meta or {}),
            "source": "current_project_MMA_only",
        },
        "params": params,
        "settings": settings,
        "relicflow": {
            "algorithm": "nBEl + cBEA",
            "model_alias": "Z4 <- current-project models/VLL",
            "active_branch": {"gammas": "gBrL+gBrc", "Brc": physics.brc, "BrL": physics.br_l, "Sigma3": "disabled by active sv expression"},
            "full_cell": {"requested": _as_bool(settings.get("FullCel"), True), "implemented": True, "backend": rate_backend + "+translated VLL BuildGetSecondMomentScat", "skip_xdm": settings.get("fullCellSkipXdm"), "charge_conjugate_factor": _CONVERSION_CHARGE_CONJUGATE_FACTOR},
            "solver_contract": {"nBE_state": "Euler; tnBE output stores Trapezoidal Y", "cBE_state": "Euler; cBE default disables Trapezoidal Newton", "safe_floor": 1.0e-120},
            "thermal_backend": rate_backend,
            "thermal_unique_x": int((rates.timings or {}).get("native_unique_x", 0.0)),
            "rate_cache": {
                "enabled": bool((rates.timings or {}).get("cache_enabled", False)),
                "hit": bool((rates.timings or {}).get("cache_hit", False)),
                "path": (rates.timings or {}).get("cache_path"),
                "fingerprint": (rates.timings or {}).get("cache_fingerprint"),
            },
            "thermal_table_settings": {key: settings[key] for key in ["Nx", "Nx1", "iacc", "iacc1", "imax", "imax1", "pg"] if key in settings},
            "psi_diagnostics": {
                "enabled": bool(psi_checks.get("enabled", False)),
                "lambdaN": float(settings.get("lambdaN", _PSI_LAMBDA_N_DEFAULT)),
                "formula_scope": psi_checks.get("formula_scope", "disabled"),
                "light_rate_mode": psi_checks.get("mixing_at_xkd", {}).get("rate_mode"),
            },
        },
        "result": {
            "oh2_nbe": float(nbe["oh2"]),
            "oh2_cbe": float(cbe["oh2"]),
            "Oh2_nBE": float(nbe["oh2"]),
            "Oh2_cBE": float(cbe["oh2"]),
            "Yc_nbe": float(nbe["Yc"]),
            "Ydl_nbe": float(nbe["Ydl"]),
            "YDeltaL_nbe": float(nbe["Ydl"]),
            "Ys_nbe": float(nbe["Ys"]),
            "Yc_cbe": float(cbe["Yc"]),
            "Ydl_cbe": float(cbe["Ydl"]),
            "YDeltaL_cbe": float(cbe["Ydl"]),
            "Ys_cbe": float(cbe["Ys"]),
            "cYdl_over_Ydl": None if nbe["Ydl"] == 0.0 else float(cbe["Ydl"] / nbe["Ydl"]),
            "n_points_nbe": len(nbe["xy"]),
            "n_points_cbe": len(cbe["xy"]),
            "runtime_s": float(total),
            "epsm": epsm,
            "xstar": xstar,
            "mPsi_GeV": mpsi_gev,
            "mPhi_GeV": mphi_gev,
            "GammaPhi_GeV": resonance["GammaPhi_GeV"],
            "GammaPhi_bar": resonance["GammaPhi_bar"],
            "B": resonance["B"],
            "C": resonance["C"],
            "YBL_nbe_xlep": ybl_nbe_xlep,
            "YBL_cbe_xlep": ybl_cbe_xlep,
            "YBL_nbe_xstar": ybl_nbe_xstar,
            "YBL_cbe_xstar": ybl_cbe_xstar,
            "Rpsi_xstar": spectator_ratio_xstar,
            "xi_xstar": xi_xstar,
            "eta_xstar": eta_xstar,
            "xi_over_eta_xstar": xi_over_eta_xstar,
        },
        "mma_observables": mma_observables,
        "lep": lep,
        "psi_checks": psi_checks,
        "reference": {"file": str(compare_mma_txt) if compare_mma_txt else None, "Oh2_nBE_mma": ref_nbe, "Oh2_cBE_mma": ref_cbe, "nbe_rel_err": nbe_rel, "cbe_rel_err": cbe_rel},
        "comparison": {"oh2_nbe_rel_err": nbe_rel, "oh2_cbe_rel_err": cbe_rel},
        "samples": {
            "svx": {str(int(x)): float(rates.svx(x)) for x in sample_x},
            "sv2x": {str(int(x)): float(rates.sv2x(x)) for x in sample_x},
            "svxl": {str(int(x)): float(rates.svxl(x)) for x in sample_x},
            "svxl1": {str(int(x)): float(rates.svxl1(x)) for x in sample_x},
            "svxl2": {str(int(x)): float(rates.svxl2(x)) for x in sample_x},
            "gs": {str(int(x)): float(rates.gs(x)) for x in sample_x},
            "Yeq": {str(int(x)): float(physics.Yeq(x)) for x in sample_x},
            "yeq": {str(int(x)): float(physics.yeq(x)) for x in sample_x},
        },
        "tables": {key: value.tolist() for key, value in rates.tables.items()},
        "sv_tables": {key: int(len(value)) for key, value in rates.tables.items()},
        "timing_s": {"load_dof": 0.0, "build_rates": float(t_rates), "solve_nBE": float(t_nbe), "solve_cBE": float(t_cbe), "total": float(total)},
        "xy_nbe": nbe["xy"],
        "xy_nbe_ydl": nbe["xy_ydl"],
        "xy_nbe_ys": nbe["xy_ys"],
        "xy_cbe": cbe["xy"],
        "xy_cbe_ydl": cbe["xy_ydl"],
        "xy_cbe_ys": cbe["xy_ys"],
        "tGammaScaFull": cbe["gamma_scattering"],
        "tGammaThOverHFull": cbe["gamma_thermal_over_h"],
        "units": {
            "x": "mDM/T",
            "Tstar_GeV": "GeV",
            "xstar": "dimensionless",
            "svx": "GeV^-2",
            "gamma": "GeV",
            "oh2": "dimensionless",
            "Y": "dimensionless",
            "YBL_xstar": "dimensionless",
            "y": "mDM*Tchi/s^(2/3)",
        },
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def print_z4_report(out: dict[str, Any], output_path: str | Path) -> None:
    result = out["result"]
    lep = out.get("lep", {})
    print("Z4 model summary")
    print(f"  output: {output_path}")
    print("  [relic and asymmetry]")
    print("  Oh2_nBE = " + _format_number(result["oh2_nbe"]))
    print("  Oh2_cBE = " + _format_number(result["oh2_cbe"]))
    print("  Ydl_nBE = " + _format_number(result["Ydl_nbe"]))
    print("  Ydl_cBE = " + _format_number(result["Ydl_cbe"]))
    print("  cYdl/Ydl = " + _format_number(result["cYdl_over_Ydl"]))

    print("  [thermal diagnostics]")
    print("  x_c = " + _format_number(lep.get("xc")))
    print("  x_kd = " + _format_number(lep.get("xkd")))
    print("  charge_conjugate_factor = " + _format_number(_CONVERSION_CHARGE_CONJUGATE_FACTOR))
    print("  x_kd departure(5%) = " + _format_number(lep.get("xkd_departure")))
    print("  x_w = " + _format_number(lep.get("xw")))
    print("  x_lep = " + _format_number(lep.get("xlep")))
    print("  mPsi = " + _format_number(lep.get("mPsi_GeV")) + " GeV")
    print("  mPhi = " + _format_number(lep.get("mPhi_GeV")) + " GeV")
    print("  GammaPhi = " + _format_number(lep.get("GammaPhi_GeV")) + " GeV")
    print("  GammaPhi_bar = " + _format_number(lep.get("GammaPhi_bar")))
    print("  B = " + _format_number(lep.get("B")))
    print("  C = " + _format_number(lep.get("C")))

    print("  [sphaleron diagnostic]")
    print("  T_star = " + _format_number(lep.get("Tstar_GeV")) + " GeV")
    print("  x_star = " + _format_number(lep.get("xstar")))
    print("  YBL_nBE(x_star) = " + _format_number(lep.get("YBL_nbe_xstar")))
    print("  YBL_cBE(x_star) = " + _format_number(lep.get("YBL_cbe_xstar")))
    print("  YBL_observed = " + _format_number(lep.get("YBL_observed")))
    print("  eps = " + _format_number(lep.get("eps")))

    print("  eps_max = " + _format_number(out.get("mma_observables", {}).get("epsm")))

    psi = out.get("psi_checks", {})
    if psi.get("enabled"):
        decay = psi.get("decay", {})
        inverse = psi.get("inverse_decay", {})
        late = psi.get("late_injection", {})
        ybl_conservation = psi.get("YBL_conservation", {})
        mixing = psi.get("mixing_at_xkd", {})
        collider = psi.get("collider_widths", {})
        print("  [Psi-nR consistency checks]")
        print("  lambdaN = " + _format_number(psi.get("lambdaN")))
        print("  GammaPsi(one component) = " + _format_number(decay.get("width_one_component_GeV")) + " GeV")
        print("  min <GammaPsi>/H = " + _format_number(decay.get("min_thermal_width_over_H")))
        print("  Psi decay gate = " + ("PASS" if decay.get("gate_pass") else "FAIL"))
        print("  x_n_dec = " + _format_number(inverse.get("x_n_dec")))
        print("  T_n_dec = " + _format_number(inverse.get("T_n_dec_GeV")) + " GeV")
        print("  x_n_therm = " + _format_number(inverse.get("x_n_therm")))
        print("  T_n_therm = " + _format_number(inverse.get("T_n_therm_GeV")) + " GeV")
        print("  gamma_D(total)@x_n_dec = " + _format_number(inverse.get("gamma_D_total_at_x_n_dec_GeV4")) + " GeV^4")
        print("  Gamma_ID/H@x_n_dec = " + _format_number(_safe_div(inverse.get("Gamma_ID_at_x_n_dec_GeV") or 0.0, inverse.get("H_at_x_n_dec_GeV") or 1.0, MIN_NUMBER)))
        print("  Delta_Neff(nR) = " + _format_number(inverse.get("DeltaNeff")))
        print("  Delta_Neff(late) = " + _format_number(late.get("DeltaNeff_late")))
        print("  Delta_Neff(total) = " + _format_number(late.get("DeltaNeff_total")))
        print("  R_late = " + _format_number(late.get("R_late")))
        print("  max rho_n/rho_rad = " + _format_number(late.get("max_rho_n_over_rho_rad")))
        print("  max delta H/H = " + _format_number(late.get("max_delta_H_over_H")))
        print("  late injection gate = " + ("PASS" if late.get("late_gate_pass") else "FAIL"))
        print("  Hubble shift gate = " + ("PASS" if late.get("hubble_shift_gate_pass") else "FAIL"))
        print("  YBL conservation test = " + ("AVAILABLE" if ybl_conservation.get("available") else "NOT_INDEPENDENTLY_TRACKED"))
        print("  sTheta(T_kd) = " + _format_number(mixing.get("sTheta")))
        print("  R_light(T_kd) = " + _format_number(mixing.get("R_light")))
        print("  light feedback gate = " + ("PASS" if mixing.get("light_gate_pass") else "FAIL"))
        print("  Gamma_Wn = " + _format_number(collider.get("gamma_Wn_GeV")) + " GeV")
        print("  Gamma_Zn = " + _format_number(collider.get("gamma_Zn_GeV")) + " GeV")
        print("  Gamma_hn = " + _format_number(collider.get("gamma_hn_GeV")) + " GeV")
        print("  Gamma_pion = " + _format_number(collider.get("gamma_pion_GeV")) + " GeV")
        print("  Gamma_Wn/Gamma_Zn = " + _format_number(collider.get("W_over_Z")))
        print("  Gamma_Zn/Gamma_hn = " + _format_number(collider.get("Z_over_h")))
        print("  Psi diagnostic gates = " + ("PASS" if psi.get("all_gates_pass") else "FAIL"))
        print("  complete spectator consistency = " + ("PASS" if psi.get("all_consistency_gates_pass") else "INCOMPLETE/FAIL"))

    print("  [runtime]")
    for key, value in out.get("timing_s", {}).items():
        print(f"  {key}: {_format_number(value)} s")
    ref = out.get("reference", {})
    if ref.get("file"):
        print("  MMA nBE rel.err = " + _format_number(ref.get("nbe_rel_err")))
        print("  MMA cBE rel.err = " + _format_number(ref.get("cbe_rel_err")))


def main() -> None:
    ap = argparse.ArgumentParser(description="RelicFlow native Z4/VLL nBEl/cBEA runner.")
    ap.add_argument("--params-file", default=str(Path(__file__).resolve().parent / "parameters_params.json"))
    ap.add_argument("--settings-file", default=str(Path(__file__).resolve().parent / "settings_Z4_default.json"))
    ap.add_argument("--dof-file", default=str(Path(__file__).resolve().parents[2] / "data" / "dof_Drees_etal.dat"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "Z4_out.json"))
    ap.add_argument("--compare-mma-txt", default=None)
    args = ap.parse_args()
    model, cards = load_native_split(args.params_file, args.settings_file)
    out = run_z4_from_cards(model or "Z4", cards, args.dof_file, args.out, args.compare_mma_txt, {"params_file": args.params_file, "settings_file": args.settings_file})
    print_z4_report(out, args.out)


if __name__ == "__main__":
    main()
