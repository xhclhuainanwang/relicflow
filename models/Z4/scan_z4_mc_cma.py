"""MC + CMA scan driver for the RelicFlow Z4/VLL model.

The scan follows the two-stage style of ``Type_one_A4_scan.py`` while keeping
the model-specific work in the existing Z4 runner.  Every accepted point is
evaluated with the complete inelastic FullCel cBE solver; no Fokker--Planck
proxy is used here.

The five scan variables are

    mchi  : [1e3, 1e4] GeV, logarithmic
    r     : [0.1, 1.95], logarithmic
    delta : [1e-4, 3], logarithmic
    y     : [1e-4, 3], logarithmic
    lam   : [1e-4, 3], logarithmic

The MC stage explores all five variables.  The default ``basic`` CMA stage
holds the best MC values of (mchi, r, delta) fixed and optimizes
log10(y), log10(lam).  The optional ``ordering`` mode also optimizes
    log10(mchi), log10(r), log10(delta) and adds a soft score for
    xlep > xc > xkd.  The optional ``epsm`` mode opens only log10(delta)
    and adds a soft score for abs(eps) < epsm.
Candidates with mPsi > mPhi are rejected before a RelicFlow solve.
All CMA modes share the base soft score for xstar > xlep; each named mode
then adds only its own mode-specific score.
"""
#& 'C:\Python314\python.exe' 'E:\Tools\relicflow\models\Z4\scan_z4_mc_cma.py' --cma-mode ordering
#& 'C:\Python314\python.exe' 'E:\Tools\relicflow\models\Z4\scan_z4_mc_cma.py' --cma-mode epsm

#& 'C:\Python314\python.exe' 'E:\Tools\relicflow\models\Z4\scan_z4_mc_cma.py'
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


# This file is intended to be run directly from VS Code, PowerShell, or an
# arbitrary working directory.  Add the RelicFlow root before importing nbe.
RELICFLOW_ROOT = Path(__file__).resolve().parents[2]
if str(RELICFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(RELICFLOW_ROOT))

from nbe.config import load_native_split  # noqa: E402
from models.Z4.z4_runner import run_z4_from_cards  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PARAMS_FILE = SCRIPT_PATH.parent / "BP3_params.json"
DEFAULT_SETTINGS_FILE = SCRIPT_PATH.parent / "settings_Z4_scan.json"
DEFAULT_DOF_FILE = RELICFLOW_ROOT / "data" / "dof_Drees_etal.dat"

SCAN_BOUNDS: dict[str, tuple[float, float]] = {
    "mchi": (2.0e3, 1.0e4),
    "r": (1.0e-1, 1.95),
    "delta": (1.0e-4, 3.0),
    "y": (1.0e-4, 3.0),
    "lam": (1.0e-4, 3.0),
}
LOG_KEYS = frozenset({"mchi", "r", "delta", "y", "lam"})
MC_KEYS = tuple(SCAN_BOUNDS)
BASIC_CMA_KEYS = ("y", "lam")
ORDERING_CMA_KEYS = ("mchi", "r", "delta", "y", "lam")
EPSM_CMA_KEYS = ("delta", "y", "lam")
# Kept as the basic-mode aliases so the existing mode's contract is explicit.
CMA_KEYS = BASIC_CMA_KEYS
FIXED_CMA_KEYS = ("mchi", "r", "delta")

Y_B_OBSERVED = 8.7e-11
Y_BL_OBSERVED = (79.0 / 28.0) * Y_B_OBSERVED
OBJECTIVE_HUGE = 1.0e100
EPS1_FIXED = 1.0e-5
OH2_WINDOW = (1.1e-1, 1.3e-1)
EPS_LIMIT = 1.0
EPS_PENALTY_WEIGHT = 1.0
XSTAR_ORDER_WEIGHT = 25.0
XSTAR_ORDER_INVALID_PENALTY = 1.0e6
ORDERING_MODE = "ordering"
ORDERING_CMA_ITERATIONS = 32
ORDERING_INVALID_PENALTY = 1.0e6
EPSM_MODE = "epsm"
EPSM_INVALID_PENALTY = 1.0e6

CMA_CSV_COLUMNS = [
    "cma_seed",
    "cma_iteration",
    "cma_member",
    "objective",
    "mchi",
    "r",
    "delta",
    "y",
    "lam",
    "mpsi",
    "mphi",
    "mL",
    "eps1",
    "Oh2_nBE",
    "Oh2_cBE",
    "Ydl_nBE",
    "Ydl_cBE",
    "YDeltaL_nBE",
    "YDeltaL_cBE",
    "YDeltaL_nBE_final",
    "YDeltaL_cBE_final",
    "YBL_nBE_final",
    "YBL_cBE_final",
    "cYdl_over_Ydl",
    "xc",
    "xkd",
    "xkd_departure",
    "xw",
    "xlep",
    "YBL_nBE_xlep",
    "YBL_cBE_xlep",
    "xstar",
    "YDeltaL_nBE_xstar",
    "YDeltaL_cBE_xstar",
    "YBL_nBE_xstar",
    "YBL_cBE_xstar",
    "Rpsi_xstar",
    "xi_xstar",
    "eta_xstar",
    "xi_over_eta_xstar",
    "YBL_observed",
    "eps",
    "epsm",
    # Psi--n_R consistency diagnostics. These are numeric-only columns so
    # accepted CMA rows remain directly machine-readable.
    "psi_lambdaN",
    "psi_width_one_component_GeV",
    "psi_phase_space_factor",
    "psi_weak_component_factor",
    "psi_charge_conjugate_factor",
    "psi_spin_internal_factor",
    "psi_min_thermal_width_over_H",
    "psi_x_at_min_thermal_width_over_H",
    "psi_decay_gate_threshold",
    "psi_decay_gate_pass",
    "psi_nR_internal_degrees",
    "psi_gamma_D_total_x_n_dec_GeV4",
    "psi_Gamma_ID_x_n_dec_GeV",
    "psi_H_x_n_dec_GeV",
    "psi_Gamma_ID_over_H_x_n_dec",
    "psi_x_n_dec",
    "psi_T_n_dec_GeV",
    "psi_x_n_therm",
    "psi_T_n_therm_GeV",
    "psi_gStarS_T_n_dec",
    "psi_Tnu_dec_reference_GeV",
    "psi_gStarS_Tnu_dec_reference",
    "DeltaNeff",
    "psi_DeltaNeff_late",
    "psi_DeltaNeff_total",
    "psi_R_late",
    "psi_Q_n_peak_GeV5",
    "psi_max_rho_n_over_rho_rad",
    "psi_x_at_max_rho_n_over_rho_rad",
    "psi_max_delta_H_over_H",
    "psi_x_at_max_delta_H_over_H",
    "psi_late_equilibrium_parent_estimate",
    "psi_late_gate_pass",
    "psi_hubble_shift_gate_pass",
    "psi_YBL_conservation_available",
    "psi_YBL_conservation_residual",
    "psi_all_consistency_gates_pass",
    "psi_T_kd_GeV",
    "psi_v_T_kd_GeV",
    "psi_mD_T_kd_GeV",
    "psi_MN_T_kd_GeV",
    "psi_sTheta_T_kd",
    "psi_cTheta_T_kd",
    "psi_deltaMmix_T_kd_GeV",
    "psi_lambdaNuN_eff_T_kd",
    "psi_R_light_T_kd",
    "psi_gamma_light_T_kd_GeV",
    "psi_light_gate_threshold",
    "psi_light_gate_pass",
    "psi_light_rate_exact",
    "psi_sTheta_zeroT",
    "psi_cTheta_zeroT",
    "psi_deltaMmix_zeroT_GeV",
    "psi_Gamma_Wn_GeV",
    "psi_Gamma_Zn_GeV",
    "psi_Gamma_hn_GeV",
    "psi_Gamma_pion_GeV",
    "psi_deltaM_charged_neutral_GeV",
    "psi_pion_open",
    "psi_W_to_pion_gate_pass",
    "psi_ctau_charged_mm",
    "psi_ctau_neutral_mm",
    "psi_Gamma_Wn_over_Gamma_Zn",
    "psi_Gamma_Zn_over_Gamma_hn",
    "psi_all_gates_pass",
    "mPsi_GeV",
    "mPhi_GeV",
    "GammaPhi_GeV",
    "GammaPhi_bar",
    "B",
    "C",
    "runtime_s",
]


def _format_number(value: Any) -> str:
    """Format console/CSV numbers using the project-wide four-digit rule."""

    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")
    magnitude = abs(number)
    if 1.0 <= magnitude < 1.0e5:
        return f"{number:.4f}"
    return f"{number:.4e}"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lhs_unit(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    """Latin-hypercube points in the open unit cube."""

    cut = (np.arange(count, dtype=float)[:, None] + rng.random((count, dimension))) / float(count)
    points = cut.copy()
    for column in range(dimension):
        rng.shuffle(points[:, column])
    return points


def _scan_bounds_for_mode(cma_mode: str) -> dict[str, tuple[float, float]]:
    bounds = dict(SCAN_BOUNDS)
    if cma_mode == ORDERING_MODE:
        bounds["mchi"] = (1.0e2, bounds["mchi"][1])
    return bounds


def _unit_to_value(
    unit: float,
    key: str,
    scan_bounds: dict[str, tuple[float, float]] = SCAN_BOUNDS,
) -> float:
    low, high = scan_bounds[key]
    clipped = float(np.clip(unit, 0.0, 1.0))
    if key in LOG_KEYS:
        return float(10.0 ** (math.log10(low) + clipped * (math.log10(high) - math.log10(low))))
    return float(low + clipped * (high - low))


def _sample_mc(
    rng: np.random.Generator,
    count: int,
    scan_bounds: dict[str, tuple[float, float]] = SCAN_BOUNDS,
) -> list[dict[str, float]]:
    units = _lhs_unit(rng, count, len(MC_KEYS))
    return [
        {
            key: _unit_to_value(units[index, column], key, scan_bounds)
            for column, key in enumerate(MC_KEYS)
        }
        for index in range(count)
    ]


def _derived_masses(candidate: dict[str, float]) -> tuple[float, float]:
    mchi = float(candidate["mchi"])
    r = float(candidate["r"])
    delta = float(candidate["delta"])
    mpsi = r * mchi
    mphi = 2.0 * mchi / math.sqrt(max(1.0 + delta, np.finfo(float).tiny))
    return mpsi, mphi


def _mass_prefilter(candidate: dict[str, float]) -> tuple[bool, str, float, float]:
    values = [_finite_float(candidate.get(key)) for key in MC_KEYS]
    if any(value is None for value in values):
        return False, "nonfinite_scan_parameter", math.nan, math.nan
    mpsi, mphi = _derived_masses(candidate)
    if mpsi > mphi:
        return False, "mpsi_gt_mphi", mpsi, mphi
    return True, "", mpsi, mphi


def _get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    return mapping.get(key, default) if isinstance(mapping, dict) else default


def _candidate_base_row(
    candidate: dict[str, float],
    base_cards: dict[str, Any],
    stage: str,
    mc_index: int | None,
    cma_seed: int | None,
    cma_iteration: int | None,
    cma_member: int | None,
    status: str,
    reason: str,
    objective: float | None,
    mpsi: float,
    mphi: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "mc_index": mc_index,
        "cma_seed": cma_seed,
        "cma_iteration": cma_iteration,
        "cma_member": cma_member,
        "status": status,
        "reason": reason,
        "post_filter": "",
        "objective": objective,
        "mchi": candidate.get("mchi"),
        "r": candidate.get("r"),
        "delta": candidate.get("delta"),
        "y": candidate.get("y"),
        "lam": candidate.get("lam"),
        "mpsi": mpsi,
        "mphi": mphi,
        "mL": base_cards.get("mL"),
        "eps1": base_cards.get("eps1"),
    }


def _row_from_output(
    candidate: dict[str, float],
    out: dict[str, Any],
    base_row: dict[str, Any],
    objective: float,
    reason: str = "",
) -> dict[str, Any]:
    result = out.get("result", {})
    lep = out.get("lep", {})
    observables = out.get("mma_observables", {})
    psi = out.get("psi_checks", {})
    psi_decay = _get(psi, "decay", {})
    psi_inverse = _get(psi, "inverse_decay", {})
    psi_mix = _get(psi, "mixing_at_xkd", {})
    psi_mix0 = _get(psi, "mixing_zero_temperature", {})
    psi_collider = _get(psi, "collider_widths", {})
    psi_late = _get(psi, "late_injection", {})
    psi_ybl = _get(psi, "YBL_conservation", {})
    light_rate_mode = str(_get(psi_mix, "rate_mode", ""))
    xlep = _finite_float(_get(lep, "xlep"))
    ybl_nbe_xlep = _get(lep, "YBL_nbe_xlep")
    ybl_cbe_xlep = _get(lep, "YBL_cbe_xlep")
    row = dict(base_row)
    row.update(
        {
            "status": "ok",
            "reason": reason,
            "objective": objective,
            "mchi": candidate["mchi"],
            "r": candidate["r"],
            "delta": candidate["delta"],
            "y": candidate["y"],
            "lam": candidate["lam"],
            "Oh2_nBE": _get(result, "Oh2_nBE"),
            "Oh2_cBE": _get(result, "Oh2_cBE"),
            "Ydl_nBE": _get(result, "Ydl_nbe"),
            "Ydl_cBE": _get(result, "Ydl_cbe"),
            "YDeltaL_nBE": _get(result, "YDeltaL_nbe", _get(result, "Ydl_nbe")),
            "YDeltaL_cBE": _get(result, "YDeltaL_cbe", _get(result, "Ydl_cbe")),
            "YDeltaL_nBE_final": _get(lep, "YDeltaL_nbe_final"),
            "YDeltaL_cBE_final": _get(lep, "YDeltaL_cbe_final"),
            "YBL_nBE_final": _get(lep, "YBL_nbe_final"),
            "YBL_cBE_final": _get(lep, "YBL_cbe_final"),
            "cYdl_over_Ydl": _get(result, "cYdl_over_Ydl"),
            "xc": _get(lep, "xc"),
            "xkd": _get(lep, "xkd"),
            "xkd_departure": _get(lep, "xkd_departure"),
            "xw": _get(lep, "xw"),
            "xlep": xlep,
            "YBL_nBE_xlep": ybl_nbe_xlep,
            "YBL_cBE_xlep": ybl_cbe_xlep,
            "xstar": _get(lep, "xstar", _get(result, "xstar")),
            "YDeltaL_nBE_xstar": _get(lep, "YDeltaL_nbe_xstar"),
            "YDeltaL_cBE_xstar": _get(lep, "YDeltaL_cbe_xstar"),
            "YBL_nBE_xstar": _get(lep, "YBL_nbe_xstar", _get(result, "YBL_nBE_xstar")),
            "YBL_cBE_xstar": _get(lep, "YBL_cbe_xstar", _get(result, "YBL_cBE_xstar")),
            "Rpsi_xstar": _get(lep, "Rpsi_xstar", _get(result, "Rpsi_xstar")),
            "xi_xstar": _get(lep, "xi_xstar", _get(result, "xi_xstar")),
            "eta_xstar": _get(lep, "eta_xstar", _get(result, "eta_xstar")),
            "xi_over_eta_xstar": _get(lep, "xi_over_eta_xstar", _get(result, "xi_over_eta_xstar")),
            "YBL_observed": _get(lep, "YBL_observed", Y_BL_OBSERVED),
            "eps": _get(lep, "eps"),
            "eps_max": _get(result, "epsm", _get(observables, "epsm")),
            "psi_lambdaN": _get(psi, "lambdaN"),
            "psi_width_one_component_GeV": _get(psi_decay, "width_one_component_GeV"),
            "psi_phase_space_factor": _get(psi_decay, "phase_space_factor"),
            "psi_weak_component_factor": _get(psi_decay, "weak_component_factor"),
            "psi_charge_conjugate_factor": _get(psi_decay, "charge_conjugate_factor"),
            "psi_spin_internal_factor": _get(psi_decay, "spin_internal_factor"),
            "psi_min_thermal_width_over_H": _get(psi_decay, "min_thermal_width_over_H"),
            "psi_x_at_min_thermal_width_over_H": _get(psi_decay, "x_at_min_thermal_width_over_H"),
            "psi_decay_gate_threshold": _get(psi_decay, "gate_threshold"),
            "psi_decay_gate_pass": _get(psi_decay, "gate_pass"),
            "psi_nR_internal_degrees": _get(psi_inverse, "nR_internal_degrees"),
            "psi_gamma_D_total_x_n_dec_GeV4": _get(psi_inverse, "gamma_D_total_at_x_n_dec_GeV4"),
            "psi_Gamma_ID_x_n_dec_GeV": _get(psi_inverse, "Gamma_ID_at_x_n_dec_GeV"),
            "psi_H_x_n_dec_GeV": _get(psi_inverse, "H_at_x_n_dec_GeV"),
            "psi_Gamma_ID_over_H_x_n_dec": _get(psi_inverse, "Gamma_ID_over_H_at_x_n_dec"),
            "psi_x_n_dec": _get(psi_inverse, "x_n_dec"),
            "psi_T_n_dec_GeV": _get(psi_inverse, "T_n_dec_GeV"),
            "psi_x_n_therm": _get(psi_inverse, "x_n_therm"),
            "psi_T_n_therm_GeV": _get(psi_inverse, "T_n_therm_GeV"),
            "psi_gStarS_T_n_dec": _get(psi_inverse, "gStarS_at_T_n_dec"),
            "psi_Tnu_dec_reference_GeV": _get(psi_inverse, "T_nu_dec_reference_GeV"),
            "psi_gStarS_Tnu_dec_reference": _get(psi_inverse, "gStarS_at_T_nu_dec_reference"),
            "DeltaNeff": _get(psi_inverse, "DeltaNeff"),
            "psi_DeltaNeff_late": _get(psi_late, "DeltaNeff_late"),
            "psi_DeltaNeff_total": _get(psi_late, "DeltaNeff_total"),
            "psi_R_late": _get(psi_late, "R_late"),
            "psi_Q_n_peak_GeV5": _get(psi_late, "Q_n_peak_GeV5"),
            "psi_max_rho_n_over_rho_rad": _get(psi_late, "max_rho_n_over_rho_rad"),
            "psi_x_at_max_rho_n_over_rho_rad": _get(psi_late, "x_at_max_rho_n_over_rho_rad"),
            "psi_max_delta_H_over_H": _get(psi_late, "max_delta_H_over_H"),
            "psi_x_at_max_delta_H_over_H": _get(psi_late, "x_at_max_delta_H_over_H"),
            "psi_late_equilibrium_parent_estimate": _get(psi_late, "equilibrium_parent_estimate"),
            "psi_late_gate_pass": _get(psi_late, "late_gate_pass"),
            "psi_hubble_shift_gate_pass": _get(psi_late, "hubble_shift_gate_pass"),
            "psi_YBL_conservation_available": _get(psi_ybl, "available"),
            "psi_YBL_conservation_residual": _get(psi_ybl, "max_abs_residual"),
            "psi_all_consistency_gates_pass": _get(psi, "all_consistency_gates_pass"),
            "psi_T_kd_GeV": _get(psi_mix, "Tkd_GeV"),
            "psi_v_T_kd_GeV": _get(psi_mix, "v_GeV"),
            "psi_mD_T_kd_GeV": _get(psi_mix, "mD_GeV"),
            "psi_MN_T_kd_GeV": _get(psi_mix, "MN_GeV"),
            "psi_sTheta_T_kd": _get(psi_mix, "sTheta"),
            "psi_cTheta_T_kd": _get(psi_mix, "cTheta"),
            "psi_deltaMmix_T_kd_GeV": _get(psi_mix, "deltaMmix_GeV"),
            "psi_lambdaNuN_eff_T_kd": _get(psi_mix, "lambdaNuN_eff"),
            "psi_R_light_T_kd": _get(psi_mix, "R_light"),
            "psi_gamma_light_T_kd_GeV": _get(psi_mix, "gamma_light_GeV"),
            "psi_light_gate_threshold": _get(psi_mix, "light_gate_threshold"),
            "psi_light_gate_pass": _get(psi_mix, "light_gate_pass"),
            "psi_light_rate_exact": 1.0 if light_rate_mode.startswith("exact") else 0.0,
            "psi_sTheta_zeroT": _get(psi_mix0, "sTheta"),
            "psi_cTheta_zeroT": _get(psi_mix0, "cTheta"),
            "psi_deltaMmix_zeroT_GeV": _get(psi_mix0, "deltaMmix_GeV"),
            "psi_Gamma_Wn_GeV": _get(psi_collider, "gamma_Wn_GeV"),
            "psi_Gamma_Zn_GeV": _get(psi_collider, "gamma_Zn_GeV"),
            "psi_Gamma_hn_GeV": _get(psi_collider, "gamma_hn_GeV"),
            "psi_Gamma_pion_GeV": _get(psi_collider, "gamma_pion_GeV"),
            "psi_deltaM_charged_neutral_GeV": _get(psi_collider, "deltaM_charged_minus_neutral_GeV"),
            "psi_pion_open": _get(psi_collider, "pion_open"),
            "psi_W_to_pion_gate_pass": _get(psi_collider, "W_to_pion_gate_pass"),
            "psi_ctau_charged_mm": _get(psi_collider, "ctau_charged_mm"),
            "psi_ctau_neutral_mm": _get(psi_collider, "ctau_neutral_mm"),
            "psi_Gamma_Wn_over_Gamma_Zn": _get(psi_collider, "W_over_Z"),
            "psi_Gamma_Zn_over_Gamma_hn": _get(psi_collider, "Z_over_h"),
            "psi_all_gates_pass": _get(psi, "all_gates_pass"),
            "mPsi_GeV": _get(lep, "mPsi_GeV", _get(result, "mPsi_GeV")),
            "mPhi_GeV": _get(lep, "mPhi_GeV", _get(result, "mPhi_GeV")),
            "GammaPhi_GeV": _get(lep, "GammaPhi_GeV", _get(result, "GammaPhi_GeV")),
            "GammaPhi_bar": _get(lep, "GammaPhi_bar", _get(result, "GammaPhi_bar")),
            "B": _get(lep, "B", _get(result, "B")),
            "C": _get(lep, "C", _get(result, "C")),
            "runtime_s": _get(result, "runtime_s"),
        }
    )
    return row


def _ordering_penalty(out: dict[str, Any]) -> tuple[float, float, bool, str]:
    """Return the two normalized violations of xlep > xc > xkd."""

    lep = out.get("lep", {})
    xlep = _finite_float(_get(lep, "xlep"))
    xc = _finite_float(_get(lep, "xc"))
    xkd = _finite_float(_get(lep, "xkd"))
    if xlep is None or xc is None or xkd is None:
        return ORDERING_INVALID_PENALTY, ORDERING_INVALID_PENALTY, False, "ordering_diagnostic_missing"
    scale = max(1.0, abs(xlep), abs(xc), abs(xkd))
    lep_before_c = max(0.0, xc - xlep) / scale
    c_before_kd = max(0.0, xkd - xc) / scale
    satisfied = xlep > xc and xc > xkd
    return (
        float(lep_before_c * lep_before_c),
        float(c_before_kd * c_before_kd),
        satisfied,
        "ordering_satisfied" if satisfied else "ordering_violation",
    )


def _xstar_after_xlep_penalty(out: dict[str, Any]) -> tuple[float, bool, str]:
    """Return the normalized violation of the base condition xstar > xlep."""

    result = out.get("result", {})
    lep = out.get("lep", {})
    xstar = _finite_float(_get(lep, "xstar", _get(result, "xstar")))
    xlep = _finite_float(_get(lep, "xlep"))
    if xstar is None or xlep is None:
        return XSTAR_ORDER_INVALID_PENALTY, False, "xstar_order_diagnostic_missing"
    scale = max(1.0, abs(xstar), abs(xlep))
    violation = max(0.0, xlep - xstar) / scale
    penalty = violation * violation
    satisfied = xstar > xlep
    return float(penalty), satisfied, "xstar_after_xlep" if satisfied else "xstar_before_xlep"


def _epsm_penalty(out: dict[str, Any]) -> tuple[float, bool, str, float | None]:
    """Return the normalized violation of abs(eps) < epsm."""

    result = out.get("result", {})
    lep = out.get("lep", {})
    observables = out.get("mma_observables", {})
    eps = _finite_float(_get(lep, "eps"))
    epsm = _finite_float(_get(result, "epsm", _get(observables, "epsm")))
    if eps is None or epsm is None or epsm <= 0.0:
        return EPSM_INVALID_PENALTY, False, "epsm_diagnostic_invalid", None
    ratio = abs(eps) / epsm
    excess = max(0.0, ratio - 1.0)
    return (
        float(excess * excess),
        ratio < 1.0,
        "eps_below_epsm" if ratio < 1.0 else "eps_above_epsm",
        float(ratio),
    )


def _objective(
    out: dict[str, Any],
    omega_low: float,
    omega_high: float,
    omega_log_scale: float,
    cma_mode: str = "basic",
    ordering_xlep_weight: float = 0.0,
    ordering_xc_xkd_weight: float = 0.0,
    epsm_weight: float = 0.0,
    xstar_weight: float = XSTAR_ORDER_WEIGHT,
) -> tuple[float, str]:
    """Score relic density and softly penalize points with ``abs(eps) > 1``.

    Points inside [omega_low, omega_high] have zero objective.  Outside the
    interval, the logarithmic distance to the nearest boundary is minimized.
    The epsilon penalty is zero below the limit and quadratic in the excess
    above it.  Every CMA mode includes a normalized soft penalty for violating
    xstar > xlep.  In ``ordering`` mode, an additional normalized soft penalty
    is added for violating xlep > xc > xkd.  In ``epsm`` mode, an additional
    normalized penalty is added for abs(eps) > epsm.  Y_B-L remains diagnostic
    only.
    """

    result = out.get("result", {})
    lep = out.get("lep", {})
    oh2 = _finite_float(_get(result, "Oh2_cBE"))
    if oh2 is None or oh2 <= 0.0:
        return OBJECTIVE_HUGE, "invalid_Oh2_cBE"
    eps = _finite_float(_get(lep, "eps"))
    if eps is None:
        return OBJECTIVE_HUGE, "invalid_eps"
    if omega_low <= 0.0 or omega_high <= omega_low or omega_log_scale <= 0.0:
        return OBJECTIVE_HUGE, "invalid_objective_scale"
    if omega_low <= oh2 <= omega_high:
        relic_term = 0.0
        relic_reason = "inside_Oh2_window"
    else:
        boundary = omega_low if oh2 < omega_low else omega_high
        relic_term = math.log10(oh2 / boundary) / omega_log_scale
        relic_reason = "below_Oh2_window" if oh2 < omega_low else "above_Oh2_window"
    eps_excess = max(0.0, abs(eps) - EPS_LIMIT)
    eps_term = EPS_PENALTY_WEIGHT * eps_excess * eps_excess
    objective = relic_term * relic_term + eps_term
    reason = relic_reason + ("+eps_gt_1" if eps_term > 0.0 else "")
    xstar_penalty, _, xstar_reason = _xstar_after_xlep_penalty(out)
    objective += float(xstar_weight) * xstar_penalty
    reason += "+" + xstar_reason
    if cma_mode == ORDERING_MODE:
        lep_before_c_penalty, c_before_kd_penalty, _, order_reason = _ordering_penalty(out)
        objective += float(ordering_xlep_weight) * lep_before_c_penalty
        objective += float(ordering_xc_xkd_weight) * c_before_kd_penalty
        reason += "+" + order_reason
    elif cma_mode == EPSM_MODE:
        epsm_penalty, _, epsm_reason, _ = _epsm_penalty(out)
        objective += float(epsm_weight) * epsm_penalty
        reason += "+" + epsm_reason
    return float(objective), reason


def _post_filter_status(row: dict[str, Any], require_cma: bool) -> str:
    """Return the final post-CMA status without applying the epsm bound."""

    if row.get("status") != "ok":
        return ""
    if require_cma and row.get("stage") != "CMA":
        return "not_CMA_stage"
    oh2 = _finite_float(row.get("Oh2_cBE"))
    eps = _finite_float(row.get("eps"))
    if oh2 is None or not (OH2_WINDOW[0] <= oh2 <= OH2_WINDOW[1]):
        return "reject_Oh2_window"
    if eps is None:
        return "reject_invalid_eps"
    if abs(eps) > 1.0:
        return "reject_eps_gt_1"
    return "accepted"


class ScanEvaluator:
    """Evaluate candidates through the existing Z4 runner in a private temp dir."""

    def __init__(
        self,
        base_cards: dict[str, Any],
        dof_file: Path,
        scan_root: Path,
        use_rate_cache: bool,
        params_file: Path,
        settings_file: Path,
    ) -> None:
        self.base_cards = dict(base_cards)
        # mS2 is a derived quantity in Z4Model.prepare_params.  Recompute it
        # for every (mchi, delta) point rather than inheriting a stale card.
        self.base_cards.pop("mS2", None)
        self.base_cards["z4RateCache"] = bool(use_rate_cache)
        # eps1 is fixed for this scan, even when the selected base params
        # card was written for a different benchmark point.
        self.base_cards["eps1"] = EPS1_FIXED
        # Accepted CMA rows must carry the note-level Psi--n_R checks.  The
        # scan-specific light-rate proxy avoids a fresh quadrature per point;
        # formal/BP runs retain the exact setting from their own cards.
        self.base_cards["psiDiagnostics"] = True
        self.base_cards["psiLightRateMode"] = "proxy"
        self.dof_file = dof_file
        self.scan_root = scan_root
        self.params_file = params_file
        self.settings_file = settings_file
        self.work_dir = Path(tempfile.mkdtemp(prefix="z4_mc_cma_", dir=str(scan_root)))
        self.counter = 0

    def close(self) -> None:
        # The work directory is scanner-owned.  Individual temporary output
        # files are removed immediately after each call; keeping this method
        # as a no-op makes the evaluator lifecycle explicit to callers.
        return None

    def evaluate(
        self,
        candidate: dict[str, float],
        stage: str,
        mc_index: int | None = None,
        cma_seed: int | None = None,
        cma_iteration: int | None = None,
        cma_member: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        allowed, reason, mpsi, mphi = _mass_prefilter(candidate)
        base_row = _candidate_base_row(
            candidate,
            self.base_cards,
            stage,
            mc_index,
            cma_seed,
            cma_iteration,
            cma_member,
            "pending",
            reason,
            None,
            mpsi,
            mphi,
        )
        if not allowed:
            base_row["status"] = "prefilter"
            return None, base_row

        cards = dict(self.base_cards)
        cards.update(
            {
                "mDM": float(candidate["mchi"]),
                "r": float(candidate["r"]),
                "delta": float(candidate["delta"]),
                "y": float(candidate["y"]),
                "lam": float(candidate["lam"]),
            }
        )
        output_path = self.work_dir / f"eval_{self.counter:07d}.json"
        self.counter += 1
        try:
            out = run_z4_from_cards(
                model_name="Z4",
                cards=cards,
                dof_file=self.dof_file,
                out_path=output_path,
                inputs_meta={
                    "scan_script": str(SCRIPT_PATH),
                    "scan_stage": stage,
                    "scan_mc_index": mc_index,
                    "scan_cma_seed": cma_seed,
                    "scan_cma_iteration": cma_iteration,
                    "scan_cma_member": cma_member,
                    "scan_params_file": str(self.params_file),
                    "scan_settings_file": str(self.settings_file),
                },
            )
        except Exception as exc:  # keep a failed point in the audit CSV
            message = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:240]}"
            base_row["status"] = "error"
            base_row["reason"] = message
            return None, base_row
        finally:
            # The returned Python object already contains the complete result;
            # do not accumulate one large trajectory JSON per scan candidate.
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
        return out, base_row


def _write_cma_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write only clean, accepted CMA observables; never write status text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CMA_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in CMA_CSV_COLUMNS:
                source_key = "eps_max" if key == "epsm" else key
                formatted[key] = _format_number(row.get(source_key))
            writer.writerow(formatted)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _make_cards(params_file: Path, settings_file: Path) -> dict[str, Any]:
    model, cards = load_native_split(params_file, settings_file)
    if model and str(model).strip().lower() not in {"z4", "vll"}:
        raise ValueError(f"Expected a Z4/VLL card pair, got model={model!r}")
    return dict(cards)


def _cma_keys_for_mode(cma_mode: str) -> tuple[str, ...]:
    if cma_mode == ORDERING_MODE:
        return ORDERING_CMA_KEYS
    if cma_mode == EPSM_MODE:
        return EPSM_CMA_KEYS
    return BASIC_CMA_KEYS


def _make_cma_candidate(
    seed: dict[str, float],
    vector: list[float] | np.ndarray,
    cma_keys: tuple[str, ...],
    scan_bounds: dict[str, tuple[float, float]] = SCAN_BOUNDS,
) -> dict[str, float]:
    candidate = dict(seed)
    for index, key in enumerate(cma_keys):
        low, high = scan_bounds[key]
        log_value = float(np.clip(vector[index], math.log10(low), math.log10(high)))
        candidate[key] = float(10.0**log_value)
    return candidate


def _save_best_output(path: Path, out: dict[str, Any], row: dict[str, Any]) -> None:
    payload = dict(out)
    payload.setdefault("scan", {})
    payload["scan"].update(
        {
            "selected_by": "MC+CMA objective",
            "objective": row.get("objective"),
            "stage": row.get("stage"),
            "mc_index": row.get("mc_index"),
            "cma_seed": row.get("cma_seed"),
            "cma_iteration": row.get("cma_iteration"),
            "candidate": {key: row.get(key) for key in MC_KEYS},
        }
    )
    _write_json(path, payload)


def run_scan(args: argparse.Namespace) -> int:
    cma_mode = str(args.cma_mode)
    cma_iterations = ORDERING_CMA_ITERATIONS if cma_mode == ORDERING_MODE else int(args.cma_iterations)
    cma_keys = _cma_keys_for_mode(cma_mode)
    scan_bounds = _scan_bounds_for_mode(cma_mode)
    if float(args.ordering_xlep_weight) < 0.0:
        raise ValueError("ordering-xlep-weight cannot be negative")
    if float(args.ordering_xc_xkd_weight) < 0.0:
        raise ValueError("ordering-xc-xkd-weight cannot be negative")
    if float(args.epsm_weight) < 0.0:
        raise ValueError("epsm-weight cannot be negative")
    if float(args.xstar_weight) < 0.0:
        raise ValueError("xstar-weight cannot be negative")
    params_file = Path(args.params_file).resolve()
    settings_file = Path(args.settings_file).resolve()
    dof_file = Path(args.dof_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"seed{int(args.seed)}"
    output_stem = f"{args.stem}_{seed_tag}"
    csv_path = output_dir / f"{output_stem}.csv"
    metadata_path = output_dir / f"{output_stem}.run.json"
    best_output_path = output_dir / f"{output_stem}_best.json"

    if args.n_mc <= 0 or args.topk < 0 or cma_iterations < 0 or args.cma_population <= 0:
        raise ValueError("n-mc, cma-population must be positive; topk and cma-iterations cannot be negative")
    if not params_file.exists():
        raise FileNotFoundError(f"params file not found: {params_file}")
    if not settings_file.exists():
        raise FileNotFoundError(f"settings file not found: {settings_file}")
    if not dof_file.exists():
        raise FileNotFoundError(f"dof file not found: {dof_file}")

    try:
        import cma
    except ImportError as exc:
        raise RuntimeError("The CMA stage requires the Python package 'cma'. Install it in the active Python environment.") from exc

    base_cards = _make_cards(params_file, settings_file)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    valid_mc: list[dict[str, Any]] = []
    evaluator = ScanEvaluator(
        base_cards,
        dof_file,
        output_dir,
        bool(args.rate_cache),
        params_file,
        settings_file,
    )
    t0 = time.perf_counter()
    best_score = OBJECTIVE_HUGE
    best_row: dict[str, Any] | None = None
    best_out: dict[str, Any] | None = None
    row_outputs: dict[int, dict[str, Any]] = {}
    counts = {"ok": 0, "prefilter": 0, "error": 0}
    cma_csv_rows: list[dict[str, Any]] = []

    def consume_result(
        candidate: dict[str, float],
        out: dict[str, Any] | None,
        base_row: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        nonlocal best_score, best_row, best_out
        if out is None:
            rows.append(base_row)
            counts[str(base_row["status"])] = counts.get(str(base_row["status"]), 0) + 1
            return OBJECTIVE_HUGE, base_row
        score, reason = _objective(
            out,
            OH2_WINDOW[0],
            OH2_WINDOW[1],
            float(args.oh2_log_scale),
            cma_mode=cma_mode,
            ordering_xlep_weight=float(args.ordering_xlep_weight),
            ordering_xc_xkd_weight=float(args.ordering_xc_xkd_weight),
            epsm_weight=float(args.epsm_weight),
            xstar_weight=float(args.xstar_weight),
        )
        row = _row_from_output(candidate, out, base_row, score, reason)
        row_outputs[id(row)] = out
        rows.append(row)
        counts["ok"] += 1
        if math.isfinite(score) and score < best_score:
            best_score = score
            best_row = row
            best_out = out
            if args.save_best_output:
                _save_best_output(best_output_path, out, row)
        return score, row

    print("[Z4 scan] MC + CMA scan", flush=True)
    print(f"[Z4 scan] params = {params_file}", flush=True)
    print(f"[Z4 scan] settings = {settings_file}", flush=True)
    print(
        "[Z4 scan] bounds: "
        + ", ".join(
            f"{key}=[{_format_number(low)}, {_format_number(high)}]" for key, (low, high) in scan_bounds.items()
        ),
        flush=True,
    )
    print("[Z4 scan] log variables = mchi, r, delta, y, lam", flush=True)
    print("[Z4 scan] prefilter = mpsi=r*mchi > mphi=2*mchi/sqrt(1+delta)", flush=True)
    print("[Z4 scan] cBE = complete inelastic FullCel", flush=True)
    print(
        f"[Z4 scan] CMA mode = {cma_mode}; optimized = {', '.join(cma_keys)}; "
        f"base xstar weight={_format_number(args.xstar_weight)} for xstar>xlep; "
        + (
            f"ordering weights: xlep>xc={_format_number(args.ordering_xlep_weight)}, "
            f"xc>xkd={_format_number(args.ordering_xc_xkd_weight)}"
            if cma_mode == ORDERING_MODE
            else (
                f"epsm weight={_format_number(args.epsm_weight)} for abs(eps)<epsm"
                if cma_mode == EPSM_MODE
                else "basic relic/eps objective"
            )
        ),
        flush=True,
    )
    print(
        "[Z4 scan] speed profile: "
        f"xmin={_format_number(base_cards.get('xmin'))}, "
        f"cxmin={_format_number(base_cards.get('cxmin'))}, "
        f"xmax={_format_number(base_cards.get('xmax'))}, "
        f"nBEhinit={_format_number(base_cards.get('nBEhinit'))}, "
        f"cBEhinit={_format_number(base_cards.get('cBEhinit'))}, "
        f"errY={_format_number(base_cards.get('errY'))}, "
        f"FullCel skip xDM={_format_number(base_cards.get('fullCellSkipXdm'))}",
        flush=True,
    )
    print(f"[Z4 scan] fixed eps1 = {_format_number(EPS1_FIXED)}", flush=True)
    objective_description = (
        " + xstar penalty for xstar>xlep + ordering penalty for xlep>xc>xkd"
        if cma_mode == ORDERING_MODE
        else (
            " + xstar penalty for xstar>xlep + epsm penalty for abs(eps)>epsm"
            if cma_mode == EPSM_MODE
            else " + xstar penalty for xstar>xlep"
        )
    )
    print(
        f"[Z4 scan] CMA objective = Oh2_cBE in [{_format_number(OH2_WINDOW[0])}, "
        f"{_format_number(OH2_WINDOW[1])}] + penalty for abs(eps)>1"
        f"{objective_description}; epsm hard filter not applied",
        flush=True,
    )

    mc_candidates = _sample_mc(rng, int(args.n_mc), scan_bounds)
    for index, candidate in enumerate(mc_candidates, start=1):
        out, base_row = evaluator.evaluate(candidate, "MC", mc_index=index)
        score, row = consume_result(candidate, out, base_row)
        if out is not None and math.isfinite(score) and score < OBJECTIVE_HUGE:
            valid_mc.append({"candidate": dict(candidate), "out": out, "row": row, "score": score})
        if index == 1 or index % max(1, int(args.print_every)) == 0 or index == len(mc_candidates):
            print(
                f"[MC] {index}/{len(mc_candidates)} status={row['status']} objective={_format_number(score)} "
                f"ok={_format_number(counts['ok'])} prefilter={_format_number(counts['prefilter'])} "
                f"errors={_format_number(counts['error'])} elapsed={_format_number(time.perf_counter() - t0)} s",
                flush=True,
            )
    valid_mc.sort(key=lambda item: float(item["score"]))
    top_seeds = valid_mc[: int(args.topk)]
    print(
        f"[Z4 scan] MC done: valid={_format_number(len(valid_mc))}, "
        f"top seeds={_format_number(len(top_seeds))}",
        flush=True,
    )

    if args.cma and top_seeds and cma_iterations > 0:
        for seed_index, seed_item in enumerate(top_seeds, start=1):
            seed = dict(seed_item["candidate"])
            x0 = [math.log10(seed[key]) for key in cma_keys]
            cma_seed_value = int(rng.integers(1, 2_000_000_000))
            options = {
                "bounds": [
                    [math.log10(scan_bounds[key][0]) for key in cma_keys],
                    [math.log10(scan_bounds[key][1]) for key in cma_keys],
                ],
                "seed": cma_seed_value,
                "popsize": int(args.cma_population),
                "verb_log": 0,
                "verb_disp": 0,
                "tolfun": float(args.cma_tolfun),
            }
            es = cma.CMAEvolutionStrategy(x0, float(args.cma_sigma), options)
            print(
                f"[CMA] seed={seed_index}/{len(top_seeds)} "
                f"mchi={_format_number(seed['mchi'])} r={_format_number(seed['r'])} "
                f"delta={_format_number(seed['delta'])} "
                f"start_y={_format_number(seed['y'])} start_lam={_format_number(seed['lam'])}",
                flush=True,
            )
            for iteration in range(cma_iterations):
                members = es.ask()
                scores: list[float] = []
                iteration_best = OBJECTIVE_HUGE
                iteration_best_row: dict[str, Any] | None = None
                for member_index, member in enumerate(members, start=1):
                    candidate = _make_cma_candidate(seed, member, cma_keys, scan_bounds)
                    out, base_row = evaluator.evaluate(
                        candidate,
                        "CMA",
                        cma_seed=seed_index,
                        cma_iteration=iteration,
                        cma_member=member_index,
                    )
                    score, row = consume_result(candidate, out, base_row)
                    scores.append(score)
                    if math.isfinite(score) and score < iteration_best:
                        iteration_best = score
                        iteration_best_row = row
                es.tell(members, scores)
                iteration_oh2 = None if iteration_best_row is None else iteration_best_row.get("Oh2_cBE")
                iteration_epsm = None if iteration_best_row is None else iteration_best_row.get("eps_max")
                iteration_eps = None if iteration_best_row is None else iteration_best_row.get("eps")
                iteration_order = (
                    None
                    if iteration_best_row is None
                    else _ordering_penalty(row_outputs.get(id(iteration_best_row), {}))[2]
                )
                iteration_xlep = None if iteration_best_row is None else iteration_best_row.get("xlep")
                iteration_xstar = None if iteration_best_row is None else iteration_best_row.get("xstar")
                iteration_xc = None if iteration_best_row is None else iteration_best_row.get("xc")
                iteration_xkd = None if iteration_best_row is None else iteration_best_row.get("xkd")
                iteration_xstar_order = (
                    None
                    if iteration_best_row is None
                    else _xstar_after_xlep_penalty(row_outputs.get(id(iteration_best_row), {}))[1]
                )
                iteration_epsm_ratio = None
                if iteration_best_row is not None and cma_mode == EPSM_MODE:
                    iteration_epsm_ratio = _epsm_penalty(
                        row_outputs.get(id(iteration_best_row), {})
                    )[3]
                ordering_text = (
                    (
                        f"order={_format_number(1.0 if iteration_order else 0.0)} "
                        f"xlep={_format_number(iteration_xlep)} "
                        f"xc={_format_number(iteration_xc)} "
                        f"xkd={_format_number(iteration_xkd)} "
                    )
                    if cma_mode == ORDERING_MODE
                    else ""
                )
                xstar_text = (
                    f"xstar>xlep={_format_number(1.0 if iteration_xstar_order else 0.0)} "
                    f"xstar={_format_number(iteration_xstar)} "
                    f"xlep={_format_number(iteration_xlep)} "
                )
                epsm_text = (
                    f"eps/epsm={_format_number(iteration_epsm_ratio)} "
                    if cma_mode == EPSM_MODE
                    else ""
                )
                print(
                    f"[CMA] seed={seed_index}/{len(top_seeds)} iter={iteration + 1}/"
                    f"{cma_iterations} best_iter={_format_number(iteration_best)} "
                    f"best_all={_format_number(best_score)} "
                    f"relic={_format_number(iteration_oh2)} "
                    f"epsm={_format_number(iteration_epsm)} "
                    f"eps={_format_number(iteration_eps)} "
                    f"{xstar_text}{ordering_text}{epsm_text}"
                    f"elapsed={_format_number(time.perf_counter() - t0)} s",
                    flush=True,
                )
                if es.stop():
                    break
            # Flush this CMA seed immediately.  Only accepted CMA rows are
            # retained; MC rows and textual rejection diagnostics stay out of
            # the user-facing CSV.
            seed_rows = [
                row for row in rows
                if row.get("stage") == "CMA" and row.get("cma_seed") == seed_index
            ]
            seed_accepted_rows: list[dict[str, Any]] = []
            for row in seed_rows:
                row["post_filter"] = _post_filter_status(row, require_cma=True)
                if row.get("post_filter") == "accepted":
                    seed_accepted_rows.append(row)
            if seed_accepted_rows:
                cma_csv_rows.extend(seed_accepted_rows)
                _write_cma_csv(csv_path, cma_csv_rows)
                print(
                    f"[CMA] seed={seed_index}/{len(top_seeds)} accepted="
                    f"{_format_number(len(seed_accepted_rows))} csv={csv_path}",
                    flush=True,
                )
    elif args.cma and not top_seeds:
        print("[CMA] skipped: no valid MC seed survived the prefilter/solver", flush=True)
    elif not args.cma:
        print("[CMA] disabled by --no-cma", flush=True)

    # The user-facing CSV is intentionally CMA-only.  MC points remain
    # internal seeds and are never written as accepted scan results.
    require_cma = True
    for row in rows:
        row["post_filter"] = _post_filter_status(row, require_cma=require_cma)
    final_rows = [row for row in rows if row.get("post_filter") == "accepted"]
    # Rewrite once at the end so the file is exact even if a later seed updates
    # the in-memory post-filter fields.  A header-only file is retained when no
    # point passes, but it still contains no MC rows or status text.
    _write_cma_csv(csv_path, final_rows)
    final_best_row = min(
        final_rows,
        key=lambda item: float(item.get("objective", OBJECTIVE_HUGE)),
        default=None,
    )
    if final_best_row is not None:
        final_best_out = row_outputs.get(id(final_best_row))
        if final_best_out is not None and args.save_best_output:
            _save_best_output(best_output_path, final_best_out, final_best_row)
        best_row = final_best_row
        best_out = final_best_out
    metadata = {
        "model": "Z4",
        "algorithm": "MC + CMAEvolutionStrategy",
        "solver": "RelicFlow Z4 nBEl + complete inelastic FullCel cBEA",
        "script": str(SCRIPT_PATH),
        "params_file": str(params_file),
        "settings_file": str(settings_file),
        "dof_file": str(dof_file),
        "speed_profile": {
            "name": "settings_Z4_scan",
            "xmin": 1.0,
            "cxmin": 1.0,
            "xmax": 1.0e4,
            "nBEhinit": 1.0e-2,
            "cBEhinit": 1.0e-2,
            "nBEerr": 1.0e-1,
            "cBEerr": 1.0e-1,
            "errY": 1.0,
            "cBEerrNewton": 5.0e-2,
            "fullCellSkipXdm": 100.0,
            "fullCellDim": 32,
            "fullCellNMu": 4,
            "fullCellNU": 8,
            "FullCel": True,
            "warning": "scan-only approximate profile with late-time FullCel freezeout cutoff; compare final candidates with BP/full settings",
        },
        "output_csv": str(csv_path),
        "final_csv": str(csv_path),
        "csv_policy": "CMA-stage rows only; written after CMA post-filter",
        "best_output": str(best_output_path) if args.save_best_output and best_out is not None else None,
        "seed": int(args.seed),
        "n_mc": int(args.n_mc),
        "topk": int(args.topk),
        "cma": bool(args.cma),
        "cma_mode": cma_mode,
        "cma_optimized_keys": list(cma_keys),
        "cma_fixed_keys": [key for key in MC_KEYS if key not in cma_keys],
        "cma_iterations": cma_iterations,
        "cma_population": int(args.cma_population),
        "cma_sigma_log10": float(args.cma_sigma),
        "rate_cache": bool(args.rate_cache),
        "scan_bounds": {key: {"low": low, "high": high, "sampling": "log" if key in LOG_KEYS else "linear"} for key, (low, high) in scan_bounds.items()},
        "prefilter": "reject if mpsi=r*mchi > mphi=2*mchi/sqrt(1+delta)",
        "fixed_inputs": {
            "eps1": EPS1_FIXED,
            "YBL_observed": Y_BL_OBSERVED,
        },
        "objective": {
            "Oh2_window": [OH2_WINDOW[0], OH2_WINDOW[1]],
            "Oh2_log_scale": float(args.oh2_log_scale),
            "eps_limit": EPS_LIMIT,
            "eps_penalty_weight": EPS_PENALTY_WEIGHT,
            "xstar_order_weight": float(args.xstar_weight),
            "xstar_order_condition": "xstar>xlep",
            "ordering_mode": cma_mode == ORDERING_MODE,
            "ordering_xlep_weight": float(args.ordering_xlep_weight),
            "ordering_xc_xkd_weight": float(args.ordering_xc_xkd_weight),
            "epsm_mode": cma_mode == EPSM_MODE,
            "epsm_weight": float(args.epsm_weight),
            "formula": (
                "relic_log_distance^2 + eps_penalty_weight*max(0,abs(eps)-eps_limit)^2"
                " + xstar_order_weight*(max(0,xlep-xstar)/scale)^2"
                + (
                    " + ordering_xlep_weight*(max(0,xc-xlep)/scale)^2"
                    " + ordering_xc_xkd_weight*(max(0,xkd-xc)/scale)^2"
                    if cma_mode == ORDERING_MODE
                    else (
                        " + epsm_weight*max(0,abs(eps)/epsm-1)^2"
                        if cma_mode == EPSM_MODE
                        else ""
                    )
                )
            ),
        },
        "post_filter": {
            "require_CMA_stage": require_cma,
            "condition": "Oh2_cBE in [0.11,0.13] and abs(eps) <= 1",
            "epsm_bound": "not applied; epsm is recorded only",
            "accepted_rows": len(final_rows),
        },
        "counts": counts,
        "rows": len(rows),
        "elapsed_s": time.perf_counter() - t0,
        "best": best_row,
        "temporary_work_dir": str(evaluator.work_dir),
    }
    _write_json(metadata_path, metadata)

    print("[Z4 scan] finished", flush=True)
    if final_rows:
        print(f"[Z4 scan] accepted CMA csv = {csv_path}", flush=True)
    else:
        print(f"[Z4 scan] no accepted CMA point; clean CSV header = {csv_path}", flush=True)
    print(f"[Z4 scan] metadata = {metadata_path}", flush=True)
    print(f"[Z4 scan] final accepted points = {_format_number(len(final_rows))}", flush=True)
    if best_row is not None:
        print(
            "[Z4 scan] best final point" if final_best_row is not None else "[Z4 scan] best computed point",
            flush=True,
        )
        for key in ("objective", "mchi", "r", "delta", "y", "lam", "Oh2_cBE", "YBL_cBE_xstar", "eps", "eps_max", "xstar"):
            print(f"  {key} = {_format_number(best_row.get(key))}", flush=True)
        if args.save_best_output:
            print(f"[Z4 scan] best full output = {best_output_path}", flush=True)
    else:
        print("[Z4 scan] no valid point was produced", flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RelicFlow Z4/VLL logarithmic MC scan followed by basic, ordering, or epsm CMA optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--params-file", default=str(DEFAULT_PARAMS_FILE), help="Base Z4 params JSON; mL and eps1 are inherited.")
    parser.add_argument("--settings-file", default=str(DEFAULT_SETTINGS_FILE), help="Base Z4 settings JSON.")
    parser.add_argument("--dof-file", default=str(DEFAULT_DOF_FILE), help="Drees et al. degrees-of-freedom table.")
    parser.add_argument("--output-dir", default=str(SCRIPT_PATH.parent / "scan_results"), help="Scanner-owned result directory.")
    parser.add_argument("--stem", default="Z4_MC_CMA", help="CSV/metadata/output filename stem.")
    parser.add_argument("--n-mc", type=int, default=500, help="Number of five-dimensional LHS MC points.")
    parser.add_argument("--topk", type=int, default=30, help="Number of best MC points used as CMA seeds.")
    parser.add_argument("--cma-iterations", type=int, default=12, help="CMA iterations per seed.")
    parser.add_argument("--cma-population", type=int, default=6, help="CMA population size per iteration.")
    parser.add_argument("--cma-sigma", type=float, default=0.55, help="Initial CMA sigma in the active log10 CMA coordinates.")
    parser.add_argument("--cma-tolfun", type=float, default=1.0e-5, help="CMA stopping tolerance on objective changes.")
    parser.add_argument(
        "--cma-mode",
        choices=("basic", ORDERING_MODE, EPSM_MODE),
        default="basic",
        help=(
            "CMA mode: all modes score xstar>xlep; basic optimizes y,lam; ordering "
            "also optimizes mchi,r,delta and scores xlep>xc>xkd; epsm optimizes "
            "delta,y,lam and scores abs(eps)<epsm."
        ),
    )
    parser.add_argument(
        "--ordering-xlep-weight",
        type=float,
        default=50.0,
        help="Penalty weight for violating xlep>xc; used only in ordering mode.",
    )
    parser.add_argument(
        "--ordering-xc-xkd-weight",
        type=float,
        default=15.0,
        help="Penalty weight for violating xc>xkd; used only in ordering mode.",
    )
    parser.add_argument(
        "--epsm-weight",
        type=float,
        default=25.0,
        help="Weight of the abs(eps)>epsm soft penalty; used only in epsm mode.",
    )
    parser.add_argument(
        "--xstar-weight",
        type=float,
        default=25,
        help="Base weight of the xstar>xlep soft penalty; inherited by all CMA modes.",
    )
    parser.add_argument("--seed", type=int, default=8, help="NumPy LHS/CMA seed.")
    parser.add_argument("--print-every", type=int, default=1, help="MC progress interval.")
    parser.add_argument("--oh2-log-scale", type=float, default=math.log10(1.10), help="Log10 distance scale outside the [0.11,0.13] relic window.")
    parser.add_argument("--cma", action=argparse.BooleanOptionalAction, default=True, help="Run the CMA refinement after MC.")
    parser.add_argument("--rate-cache", action=argparse.BooleanOptionalAction, default=False, help="Reuse the provenance-checked Z4 thermal rate cache.")
    parser.add_argument("--save-best-output", action=argparse.BooleanOptionalAction, default=True, help="Save the complete trajectory JSON for the best point.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        return run_scan(args)
    except Exception as exc:
        print(f"[Z4 scan] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
