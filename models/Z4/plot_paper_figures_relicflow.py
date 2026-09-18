"""Plot the non-small-scale paper figures from RelicFlow Z4 scan outputs.

Included figures are the five figures used in the paper outside the small-
scale section and the four BP evolution panels:

    rdk.pdf       r--delta, coloured by x_kd
    rd011C.pdf    Omega_cBE/Omega_nBE versus x_kd/x_cd, coloured by C
    rde.pdf       r--delta, coloured by the corrected epsilon_CP
    rd022.pdf     relic-density ratio versus asymmetry ratio, coloured by
                  x_lep/x_cd
    sv.pdf        present-day effective annihilation rate and indirect limits

The script reads the accepted RelicFlow scan CSVs and reuses the active
RelicFlow Z4 ``Z4Physics.sigma_s`` implementation for the indirect-rate plot.
It deliberately does not read or generate the small-scale figures or the
four BP evolution figures.

Typical invocation from the RelicFlow root is:

    python models/Z4/plot_paper_figures_relicflow.py

Use ``--scan-file`` to provide one or more explicit scan CSVs and
``--figure`` to select a subset of the five outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


MODEL_DIR = Path(__file__).resolve().parent
RELICFLOW_ROOT = MODEL_DIR.parents[1]
DEFAULT_SCAN_DIR = MODEL_DIR / "scan_results"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "plots" / "paper_relicflow"
DEFAULT_DATA_DIR = MODEL_DIR / "data" / "indirect_detection"
DEFAULT_DOF_FILE = RELICFLOW_ROOT / "data" / "dof_Drees_etal.dat"

FIGURES = ("rdk", "rd011C", "rde", "rd022", "sv")
OMEGA_LOW = 0.1
OMEGA_HIGH = 0.14
Y_B_OBSERVED = 8.7e-11
T_SPHALERON_GEV = 131.7
GEV2_TO_CM3_S = 0.389379e-27 * 2.99792458e10
# The paper's present-day indirect-detection plot uses the physical
# one-component sigma_s convention.  The 2 x 2 channel multiplicity is used
# by the chemical-decoupling evolution, not applied again to this ID output.
ANNIHILATION_TOTAL_FACTOR = 1.0

POINT_COLOR = "#A73163"
BACKGROUND_COLOR = "lightcoral"
CLASS_COLORS = {
    "Excluded by collider": "#A73163",
    "Excluded by sphaleron": "#FF8C42",
    "Unconstrained": "#00C9A7",
}
VLL_RAINBOW = LinearSegmentedColormap.from_list(
    "vll_rainbow", ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
)


def format_number(value: object) -> str:
    """Format numeric console output with the project-wide four-digit rule."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")
    if abs(number) < 1.0 or abs(number) >= 1.0e5:
        return f"{number:.4e}"
    return f"{number:.4f}"


def numeric(value: object) -> float:
    """Parse a CSV value, accepting the scientific notation used by scans."""

    if value is None or str(value).strip() == "":
        return math.nan
    try:
        return float(str(value).strip())
    except ValueError:
        return math.nan


def discover_scan_files() -> list[Path]:
    """Prefer the current per-seed accepted CMA files, then legacy files."""

    seed_files = sorted(DEFAULT_SCAN_DIR.glob("Z4_MC_CMA_seed*.csv"))
    if seed_files:
        return seed_files
    candidates = []
    for name in ("Z4_MC_CMA.csv", "Z4_MC_CMA_final.csv"):
        path = DEFAULT_SCAN_DIR / name
        if path.exists():
            candidates.append(path)
    return candidates


def row_is_accepted(row: dict[str, str]) -> bool:
    """Keep accepted rows while supporting both scanner CSV schemas."""

    status = row.get("status", "").strip().lower()
    post_filter = row.get("post_filter", "").strip().lower()
    if status and status not in {"ok", "accepted"}:
        return False
    if post_filter and post_filter != "accepted":
        return False
    return True


def load_scan_rows(
    paths: Iterable[Path],
    *,
    omega_low: float = OMEGA_LOW,
    omega_high: float = OMEGA_HIGH,
) -> list[dict[str, float]]:
    """Load finite, viable rows and de-duplicate overlapping scan files."""

    required = ("mchi", "r", "delta", "y", "lam", "Oh2_nBE", "Oh2_cBE")
    rows: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            for raw in reader:
                if not row_is_accepted(raw):
                    continue
                row = {key: numeric(value) for key, value in raw.items() if key}
                if any(not math.isfinite(row.get(key, math.nan)) for key in required):
                    continue
                if not omega_low <= row["Oh2_cBE"] <= omega_high:
                    continue
                key = tuple(round(row[name], 12) for name in ("mchi", "r", "delta", "y", "lam"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    if not rows:
        searched = ", ".join(str(path) for path in paths)
        raise ValueError(f"No accepted scan rows in [{searched}].")
    return rows


def array(rows: list[dict[str, float]], name: str) -> np.ndarray:
    return np.asarray([row.get(name, math.nan) for row in rows], dtype=float)


def finite_mask(*values: np.ndarray, positive: bool = False) -> np.ndarray:
    mask = np.ones(values[0].shape, dtype=bool)
    for value in values:
        mask &= np.isfinite(value)
        if positive:
            mask &= value > 0.0
    return mask


def positive_norm(values: np.ndarray) -> mcolors.LogNorm:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        raise ValueError("A positive colour variable is empty.")
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        low *= 0.9
        high *= 1.1
    return mcolors.LogNorm(vmin=low, vmax=high)


def sym_log_norm(values: np.ndarray) -> mcolors.SymLogNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("A finite colour variable is empty.")
    low = float(np.min(finite))
    high = float(np.max(finite))
    scale = max(abs(low), abs(high), 1.0e-12)
    linthresh = max(1.0e-3, 1.0e-2 * scale)
    if low == high:
        low -= linthresh
        high += linthresh
    return mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0, base=10, vmin=low, vmax=high)


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 20,
            "axes.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "figure.dpi": 100,
            "figure.autolayout": True,
            "savefig.dpi": 300,
            "pdf.compression": 9,
        }
    )


def finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scatter_background(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    ax.scatter(x, y, s=45, c=BACKGROUND_COLOR, alpha=0.18, marker="o", edgecolors="none", zorder=1)


def plot_rdk(rows: list[dict[str, float]], output_dir: Path) -> None:
    r, delta, xkd = array(rows, "r"), array(rows, "delta"), array(rows, "xkd")
    mask = finite_mask(r, delta, xkd, positive=True)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$r=m_\psi/m_{\chi}$")
    ax1.set_ylabel(r"$\delta$", fontsize=20, rotation=0, labelpad=28)
    ax1.tick_params(direction="in", which="both", top=True, right=True)
    ax1.scatter(r[mask], delta[mask], s=45, c="lightcoral", alpha=0.18, marker="o", edgecolors="none", zorder=1)
    sc1 = ax1.scatter(
        r[mask], delta[mask], s=40, c=xkd[mask], cmap="plasma",
        norm=mcolors.Normalize(vmin=np.min(xkd[mask]), vmax=np.max(xkd[mask])),
        alpha=0.95, marker="o", edgecolors="black", linewidths=0.25, zorder=2,
    )
    cb1 = fig.colorbar(sc1, ax=ax1)
    cb1.set_label(r"$x_{\rm kd}$", fontsize=20, rotation=0)
    finish(fig, output_dir / "rdk.pdf")


def plot_rd011c(rows: list[dict[str, float]], output_dir: Path) -> None:
    relic = array(rows, "Oh2_cBE") / array(rows, "Oh2_nBE")
    xkd_xc = array(rows, "xkd") / array(rows, "xc")
    coeff = array(rows, "C")
    mask = finite_mask(relic, xkd_xc, positive=True) & np.isfinite(coeff)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$x_{\rm kd}/x_{\rm cd}$")
    ax1.set_ylabel(r"$\frac{\Omega h^{2}_{\mathrm{cBE}}}{\Omega h^{2}_{\mathrm{nBE}}}$", fontsize=20, rotation=0, labelpad=28)
    ax1.tick_params(direction="in", which="both", top=True, right=True)
    ax1.scatter(xkd_xc[mask], relic[mask], s=45, c="lightcoral", alpha=0.18, marker="o", edgecolors="none", zorder=1)
    normC = mcolors.SymLogNorm(
        linthresh=1.0e-2, linscale=1.0, base=10,
        vmin=np.nanmin(coeff[mask]), vmax=np.nanmax(coeff[mask]),
    )
    sc1 = ax1.scatter(
        xkd_xc[mask], relic[mask], s=40, c=coeff[mask], cmap=VLL_RAINBOW,
        norm=normC, alpha=0.95, marker="o", edgecolors="black", linewidths=0.25, zorder=2,
    )
    cb1 = fig.colorbar(sc1, ax=ax1)
    cb1.set_label(r"$C$", fontsize=20, rotation=0)
    finish(fig, output_dir / "rd011C.pdf")


def plot_rde(rows: list[dict[str, float]], output_dir: Path) -> None:
    r, delta, epsilon = array(rows, "r"), array(rows, "delta"), array(rows, "eps")
    mask = finite_mask(r, delta, epsilon, positive=True)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$r=m_\psi/m_{\chi}$")
    ax1.set_ylabel(r"$\delta$", fontsize=20, rotation=0, labelpad=28)
    ax1.tick_params(direction="in", which="both", top=True, right=True)
    ax1.scatter(r[mask], delta[mask], s=45, c="lightcoral", alpha=0.18, marker="o", edgecolors="none", zorder=1)
    sc1 = ax1.scatter(
        r[mask], delta[mask], s=40, c=epsilon[mask], cmap="plasma",
        norm=mcolors.LogNorm(vmin=np.min(epsilon[mask]), vmax=np.max(epsilon[mask])),
        alpha=0.95, marker="o", edgecolors="black", linewidths=0.25, zorder=2,
    )
    cb1 = fig.colorbar(sc1, ax=ax1)
    cb1.set_label(r"$\epsilon_{\rm cp}$", fontsize=20, rotation=0)
    cb1.ax.yaxis.set_label_coords(5.0, 0.54)
    finish(fig, output_dir / "rde.pdf")


def plot_rd022(rows: list[dict[str, float]], output_dir: Path) -> None:
    relic = array(rows, "Oh2_cBE") / array(rows, "Oh2_nBE")
    asym = array(rows, "cYdl_over_Ydl")
    if not np.any(np.isfinite(asym)):
        asym = array(rows, "Ydl_cBE") / array(rows, "Ydl_nBE")
    xlep_xc = array(rows, "xlep") / array(rows, "xc")
    mask = finite_mask(relic, asym, xlep_xc, positive=True)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\Omega h^{2}_{\mathrm{cBE}}/\Omega h^{2}_{\mathrm{nBE}}$")
    ax1.set_ylabel(r"$\frac{Y_{B-L}^{\mathrm{cBE}}}{Y_{B-L}^{\mathrm{nBE}}}$", fontsize=20, rotation=0, labelpad=28)
    ax1.tick_params(direction="in", which="both", top=True, right=True)
    ax1.scatter(relic[mask], asym[mask], s=45, c="lightcoral", alpha=0.18, marker="o", edgecolors="none", zorder=1)
    sc1 = ax1.scatter(
        relic[mask], asym[mask], s=40, c=xlep_xc[mask], cmap="plasma",
        norm=mcolors.Normalize(vmin=np.min(xlep_xc[mask]), vmax=np.max(xlep_xc[mask])),
        alpha=0.95, marker="o", edgecolors="black", linewidths=0.25, zorder=2,
    )
    cb1 = fig.colorbar(sc1, ax=ax1)
    cb1.set_label(r"$\frac{x_{\rm lep}}{x_{\rm cd}}$", fontsize=20, rotation=0, labelpad=28)
    finish(fig, output_dir / "rd022.pdf")


def present_day_s_from_vrel(mchi: float, vrel: float) -> float:
    if not 0.0 < vrel < 1.0:
        raise ValueError("--vrel must lie strictly between 0 and 1.")
    return 2.0 * mchi**2 * (1.0 + 1.0 / math.sqrt(1.0 - vrel**2))


def build_physics(row: dict[str, float], dof: object):
    # These defaults are the active Z4 scan-card defaults when older CSVs do
    # not carry mL or eps1 explicitly.
    params = {
        "mDM": row["mchi"],
        "r": row["r"],
        "delta": row["delta"],
        "y": row["y"],
        "lam": row["lam"],
        "mL": row.get("mL", 0.1),
        "eps1": row.get("eps1", 1.0e-5),
    }
    from nbe.models.z4 import Z4Model, Z4Physics

    return Z4Physics(Z4Model().prepare_params(params), {}, dof)


def compute_id_rate(rows: list[dict[str, float]], dof: object, vrel: float, visible_fraction: float) -> np.ndarray:
    values = []
    for row in rows:
        physics = build_physics(row, dof)
        s_value = present_day_s_from_vrel(float(row["mchi"]), vrel)
        one_component_gev2 = physics.sigma_s(s_value)
        values.append(one_component_gev2 * ANNIHILATION_TOTAL_FACTOR * visible_fraction * GEV2_TO_CM3_S)
    return np.asarray(values, dtype=float)


def sphaleron_label(row: dict[str, float]) -> str:
    """Use the RelicFlow x_star diagnostic for the paper's x_lep<x_star gate."""

    mpsi = row.get("mPsi_GeV", row["mchi"] * row["r"])
    if math.isfinite(mpsi) and mpsi <= 210.0:
        return "Excluded by collider"
    xlep = row.get("xlep", math.nan)
    xstar = row.get("xstar", math.nan)
    if math.isfinite(xlep) and math.isfinite(xstar) and xlep >= xstar:
        return "Excluded by sphaleron"
    # For legacy rows without x_star, retain the original temperature test.
    if math.isfinite(xlep) and xlep > 0.0 and row["mchi"] / xlep < T_SPHALERON_GEV:
        return "Excluded by sphaleron"
    return "Unconstrained"


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=float)
    data = np.atleast_2d(data)
    mask = np.all(np.isfinite(data[:, :2]), axis=1) & np.all(data[:, :2] > 0.0, axis=1)
    return data[mask, 0], data[mask, 1]


def plot_sv(rows: list[dict[str, float]], output_dir: Path, data_dir: Path, dof_file: Path, vrel: float, visible_fraction: float) -> None:
    from nbe.dof import DofTable

    mchi = array(rows, "mchi")
    sv_id = compute_id_rate(rows, DofTable.from_file(dof_file), vrel, visible_fraction)
    labels = [sphaleron_label(row) for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(mchi, sv_id, s=45, c=BACKGROUND_COLOR, alpha=0.18, marker="o", edgecolors="none", zorder=0)
    for label, zorder in (("Excluded by collider", 3), ("Excluded by sphaleron", 4), ("Unconstrained", 5)):
        mask = np.asarray([item == label for item in labels], dtype=bool)
        if np.any(mask):
            ax.scatter(mchi[mask], sv_id[mask], s=40, c=CLASS_COLORS[label], alpha=0.95, marker="o", edgecolors="black", linewidths=0.3, label=label, zorder=zorder)

    curve_specs = (
        ("fermi_bb.txt", "#4E8397", r"$Fermi\text{-}LAT\ b\bar{b}$"),
        ("HESS.txt", "#845EC2", "H.E.S.S"),
        ("CTA.txt", "#D5CABD", r"CTA $W^{+}W^{-}$"),
    )
    handles = []
    for name, color, label in curve_specs:
        x_values, y_values = load_curve(data_dir / name)
        ax.plot(x_values, y_values, color=color, linewidth=2, linestyle="-.", label=label)
        handles.append(Line2D([], [], color=color, linewidth=2, linestyle="-.", label=label))
    class_handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=CLASS_COLORS[label], markeredgecolor="black", label=label) for label in CLASS_COLORS]
    ax.legend(handles=class_handles + handles, loc="lower right", fontsize=9, ncol=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10.0, 1.0e4)
    ax.set_ylim(1.0e-27, 1.0e-24)
    ax.set_xlabel(r"$m_{\chi}\,[\mathrm{GeV}]$")
    ax.set_ylabel(r"$\langle\sigma v\rangle_{\rm ID}\,[\mathrm{cm^3\,s^{-1}}]$")
    finish(fig, output_dir / "sv.pdf")

    counts = {label: labels.count(label) for label in CLASS_COLORS}
    print("[sv] vrel=" + format_number(vrel) + " visible_fraction=" + format_number(visible_fraction))
    print("[sv] counts=" + ", ".join(f"{key}:{format_number(value)}" for key, value in counts.items()))
    print("[sv] rate_min=" + format_number(np.min(sv_id)) + " rate_max=" + format_number(np.max(sv_id)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-file", action="append", type=Path, help="Explicit scan CSV; repeat for multiple files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dof-file", type=Path, default=DEFAULT_DOF_FILE)
    parser.add_argument("--figure", choices=("all",) + FIGURES, default=[], action="append")
    parser.add_argument("--omega-low", type=float, default=OMEGA_LOW)
    parser.add_argument("--omega-high", type=float, default=OMEGA_HIGH)
    parser.add_argument("--vrel", type=float, default=1.0e-3, help="Present-day relative velocity used for sv.pdf.")
    parser.add_argument("--visible-fraction", type=float, default=1.0, help="Visible-energy fraction multiplying the total annihilation rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.omega_low >= args.omega_high:
        raise ValueError("--omega-low must be smaller than --omega-high.")
    if not 0.0 < args.visible_fraction <= 1.0:
        raise ValueError("--visible-fraction must lie in (0, 1].")
    scan_files = [path.resolve() for path in args.scan_file] if args.scan_file else discover_scan_files()
    if not scan_files:
        raise FileNotFoundError(f"No scan CSV found under {DEFAULT_SCAN_DIR}.")
    rows = load_scan_rows(scan_files, omega_low=args.omega_low, omega_high=args.omega_high)
    paper_style()
    selected = set(FIGURES if not args.figure or "all" in args.figure else args.figure)
    print("[Z4 paper plots] rows=" + format_number(len(rows)))
    print("[Z4 paper plots] scan_files=" + format_number(len(scan_files)))
    if "rdk" in selected:
        plot_rdk(rows, args.output_dir)
    if "rd011C" in selected:
        plot_rd011c(rows, args.output_dir)
    if "rde" in selected:
        plot_rde(rows, args.output_dir)
    if "rd022" in selected:
        plot_rd022(rows, args.output_dir)
    if "sv" in selected:
        if not args.dof_file.exists():
            raise FileNotFoundError(f"Degree-of-freedom table not found: {args.dof_file}")
        plot_sv(rows, args.output_dir, args.data_dir, args.dof_file, args.vrel, args.visible_fraction)
    print("[Z4 paper plots] output_dir=" + str(args.output_dir.resolve()))


if __name__ == "__main__":
    _root = str(RELICFLOW_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    main()
