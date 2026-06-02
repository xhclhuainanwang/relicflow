import json
import re
import time
from pathlib import Path

from .config import load_native_config
from .dof import DofTable
from .models import get_model_adapter
from .solver import solve_nbe, yeq
from .thermal import build_total_svx
from .wl_cards import load_wl_cards


def _parse_reference_oh2(path: Path) -> float | None:
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Oh2_nBE\s*=\s*([^\r\n]+)", txt)
    if not m:
        return None
    raw = m.group(1).strip().replace("*^", "e")
    try:
        return float(raw)
    except Exception:
        return None


def run_nbe_workflow_from_cards(
    model_name: str,
    cards: dict,
    dof_file: str | Path,
    out_json: str | Path,
    param_overrides: dict | None = None,
    inputs_meta: dict | None = None,
) -> dict:
    t_total0 = time.perf_counter()
    timing_s: dict[str, float | dict[str, float]] = {}

    t0 = time.perf_counter()
    _ = cards
    timing_s["load_cards"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    adapter = get_model_adapter(model_name)
    params = adapter.prepare_params(cards)
    if param_overrides:
        params.update(param_overrides)
    settings = dict(cards)
    timing_s["prepare_model"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dof = DofTable.from_file(dof_file)
    timing_s["load_dof"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    channels = adapter.build_channels(params)
    timing_s["build_channels"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    svx, sv_meta = build_total_svx(channels, settings, float(params["mDM"]))
    timing_s["build_svx"] = time.perf_counter() - t0
    if "timing_s" in sv_meta:
        timing_s["build_svx_channels"] = dict(sv_meta["timing_s"]["channels"])

    t0 = time.perf_counter()
    solved = solve_nbe(params, settings, svx, dof)
    timing_s["solve_nbe"] = time.perf_counter() - t0
    runtime_s = float(timing_s["build_svx"]) + float(timing_s["solve_nbe"])

    t0 = time.perf_counter()
    sample_x = [1.0, 3.0, 10.0, 30.0, 100.0]
    sample_svx = {f"{x:g}": float(svx(x)) for x in sample_x}
    sample_yeq = {f"{x:g}": float(yeq(float(params["mDM"]), x, float(params["gDM"]), dof)) for x in sample_x}
    channel_svx_funcs = sv_meta.get("channel_svx_funcs", {})
    sample_channel_svx: dict[str, dict[str, float]] = {}
    sample_channel_frac: dict[str, dict[str, float]] = {}
    for x in sample_x:
        kx = f"{x:g}"
        sample_channel_svx[kx] = {}
        sample_channel_frac[kx] = {}
        vals = {name: float(fn(x)) for name, fn in channel_svx_funcs.items()}
        tot = sum(vals.values())
        for name, v in vals.items():
            sample_channel_svx[kx][name] = v
            sample_channel_frac[kx][name] = 0.0 if tot <= 0 else v / tot
    timing_s["sample_eval"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_file = None
    ref_oh2 = None
    if inputs_meta and all(k in inputs_meta for k in ("model_dir", "param", "settings")):
        ref_file = Path(inputs_meta["model_dir"]) / f"results_{model_name}_{Path(inputs_meta['param']).stem}_{Path(inputs_meta['settings']).stem}.txt"
        ref_oh2 = _parse_reference_oh2(ref_file)
    rel_err = None
    if ref_oh2 is not None and ref_oh2 != 0:
        rel_err = abs(solved.oh2_nbe - ref_oh2) / abs(ref_oh2)
    timing_s["reference_eval"] = time.perf_counter() - t0

    channel_rows = {name: int(len(tab)) for name, tab in sv_meta["tables"].items()}
    channel_timing = timing_s.get("build_svx_channels", {})
    channel_timing_frac = {}
    if isinstance(channel_timing, dict):
        t_sum = sum(float(v) for v in channel_timing.values())
        if t_sum > 0:
            channel_timing_frac = {k: float(v) / t_sum for k, v in channel_timing.items()}
        else:
            channel_timing_frac = {k: 0.0 for k in channel_timing.keys()}

    out = {
        "inputs": {
            "model": model_name,
            **(inputs_meta or {"dof_file": str(dof_file), "source": "native_cards"}),
            "param_overrides": param_overrides or {},
        },
        "result": {
            "oh2_nbe": solved.oh2_nbe,
            "runtime_s": runtime_s,
            "n_points": int(len(solved.xy)),
            "svx_xmax": float(svx(float(settings["xmax"]))),
        },
        "samples": {
            "svx": sample_svx,
            "yeq": sample_yeq,
            "channel_svx": sample_channel_svx,
            "channel_frac": sample_channel_frac,
        },
        "reference": {
            "file": str(ref_file) if ref_file is not None else None,
            "oh2_nbe_ref": ref_oh2,
            "rel_err": rel_err,
        },
        "xy": solved.xy.tolist(),
        "sv_tables": channel_rows,
        "timing_channel_frac": channel_timing_frac,
        "units": {
            "time": "s",
            "svx": "GeV^-2",
            "yeq": "dimensionless",
            "oh2_nbe": "dimensionless",
        },
        "timing_s": timing_s,
    }

    t0 = time.perf_counter()
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    timing_s["write_output"] = time.perf_counter() - t0
    timing_s["total"] = time.perf_counter() - t_total0
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_nbe_workflow(
    model_name: str,
    model_dir: str | Path,
    param_card: str,
    settings_card: str,
    dof_file: str | Path,
    out_json: str | Path,
    param_overrides: dict | None = None,
) -> dict:
    model_dir = Path(model_dir)
    cards = load_wl_cards(model_dir / param_card, model_dir / settings_card)
    return run_nbe_workflow_from_cards(
        model_name=model_name,
        cards=cards,
        dof_file=dof_file,
        out_json=out_json,
        param_overrides=param_overrides,
        inputs_meta={
            "model_dir": str(model_dir),
            "param": param_card,
            "settings": settings_card,
            "dof_file": str(dof_file),
            "source": "wl_cards",
        },
    )


def run_nbe_workflow_from_config(
    config_path: str | Path,
    dof_file: str | Path,
    out_json: str | Path,
    model_name: str | None = None,
    param_overrides: dict | None = None,
) -> dict:
    model_in_file, cards = load_native_config(config_path)
    model = model_name or model_in_file
    if not model:
        raise ValueError("Model name is required: set --model or include 'model' in config JSON.")
    return run_nbe_workflow_from_cards(
        model_name=model,
        cards=cards,
        dof_file=dof_file,
        out_json=out_json,
        param_overrides=param_overrides,
        inputs_meta={
            "config": str(config_path),
            "dof_file": str(dof_file),
            "source": "native_config",
        },
    )

