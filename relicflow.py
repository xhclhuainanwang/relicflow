#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

from nbe import run_nbe_workflow, run_nbe_workflow_from_cards, run_nbe_workflow_from_config
from nbe.config import load_native_config, load_native_split
#python relicflow/relicflow.py A4 parameters2
#python relicflow.py V4 BP00
def main() -> None:
    ap = argparse.ArgumentParser(description="RelicFlow: modular nBE mini-framework with pluggable models.")
    ap.add_argument("pos_model", nargs="?", default=None, help="Short form model name, e.g. A4.")
    ap.add_argument("pos_case", nargs="?", default=None, help="Short form case name, e.g. parameters2.")
    ap.add_argument("--model", type=str, default=None, help="Model adapter name, e.g. A4. Optional if config contains model.")
    ap.add_argument("--config", type=str, default=None, help="Native project config JSON (single-file mode).")
    ap.add_argument("--params-file", type=str, default=None, help="Native params JSON file (split-file mode).")
    ap.add_argument("--settings-file", type=str, default=None, help="Native settings JSON file (split-file mode).")
    ap.add_argument("--model-dir", type=str, default=None, help="Legacy: directory containing .wl parameter/settings cards")
    ap.add_argument("--param", type=str, default=None, help="Legacy: parameter card filename (.wl)")
    ap.add_argument("--settings", type=str, default=None, help="Legacy: settings card filename (.wl)")
    ap.add_argument("--dof-file", type=str, default="data/dof_Drees_etal.dat")
    ap.add_argument("--out", type=str, default="nbe_out.json")
    ap.add_argument("--compare-mma-txt", type=str, default=None, help="Optional MMA results txt file for direct comparison.")
    ap.add_argument(
        "--with-wwzz",
        action="store_true",
        help="A4 mode: enable WW/ZZ channels (default is NN-only).",
    )
    ap.add_argument(
        "--fast-nn-only",
        action="store_true",
        help="Alias of default A4 behavior (NN-only); kept for compatibility.",
    )
    ap.add_argument(
        "--with-bau",
        action="store_true",
        help="A4 mode: solve the coupled DM-RH-neutrino flavored BAU diagnostic.",
    )
    ap.add_argument(
        "--bau-fast",
        action="store_true",
        help="BAU mode: use the fast implicit logarithmic-grid approximation.",
    )
    ap.add_argument(
        "--bau-exact",
        action="store_true",
        help="BAU mode: override the default fast approximation and use BDF.",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    shorthand_used = bool(args.pos_model or args.pos_case)
    if shorthand_used:
        if not (args.pos_model and args.pos_case):
            raise ValueError("Short form requires both positional args: relicflow.py <model> <case>.")
        if any([args.config, args.params_file, args.settings_file, args.model, args.model_dir, args.param, args.settings]):
            raise ValueError("Use short form alone, or the explicit flag form, not both.")
        args.model = args.pos_model
        case = args.pos_case
        model_dir_name = "Z4" if args.model.strip().lower() == "vll" else args.model
        search_roots = [script_dir / "models" / model_dir_name, script_dir / "configs" / model_dir_name]
        params_guess = next((root / f"{case}_params.json" for root in search_roots if (root / f"{case}_params.json").exists()), None)
        single_config_guess = next((root / f"{case}.json" for root in search_roots if (root / f"{case}.json").exists()), None)
        settings_guess = next((root / f"{case}_settings.json" for root in search_roots if (root / f"{case}_settings.json").exists()), None)
        settings_model_name = "Z4" if args.model.strip().lower() == "vll" else args.model
        default_settings = next((root / f"settings_{settings_model_name}_default.json" for root in search_roots if (root / f"settings_{settings_model_name}_default.json").exists()), None)
        if default_settings is None and args.model.strip().lower() in {"disp", "dispersion"}:
            default_settings = next((root / "benchmark_settings.json" for root in search_roots if (root / "benchmark_settings.json").exists()), None)
        if params_guess is not None:
            args.params_file = str(params_guess)
            if settings_guess is not None:
                args.settings_file = str(settings_guess)
            elif default_settings is not None:
                args.settings_file = str(default_settings)
            else:
                raise FileNotFoundError(f"Cannot find settings file for short form case {case} under {search_roots}")
        elif single_config_guess is not None:
            args.params_file = str(single_config_guess)
            if settings_guess is not None:
                args.settings_file = str(settings_guess)
            elif default_settings is not None:
                args.settings_file = str(default_settings)
            else:
                raise FileNotFoundError(f"Cannot find settings file for short form case {case} under {search_roots}")
        else:
            raise FileNotFoundError(f"Cannot find params file for short form: {case}_params.json or {case}.json under {search_roots}")
        if args.out == "nbe_out.json":
            output_name = f"{case}_out.json" if args.model.strip().lower() == "v4" else f"{args.model}_{case}_out.json"
            args.out = str(script_dir / "models" / model_dir_name / output_name)
        if args.dof_file == "data/dof_Drees_etal.dat":
            args.dof_file = str(script_dir / "data" / "dof_Drees_etal.dat")

    if args.config and (args.params_file or args.settings_file):
        raise ValueError("Use either --config OR --params-file/--settings-file, not both.")
    if bool(args.params_file) ^ bool(args.settings_file):
        raise ValueError("Provide both --params-file and --settings-file for split-file mode.")

    effective_model = args.model
    if args.config and not effective_model:
        cfg_model, _ = load_native_config(args.config)
        effective_model = cfg_model
    if args.params_file and args.settings_file and not effective_model:
        cfg_model, _ = load_native_split(args.params_file, args.settings_file)
        effective_model = cfg_model

    param_overrides = {}
    if args.bau_fast and args.bau_exact:
        raise ValueError("Use only one of --bau-fast and --bau-exact.")
    if args.bau_fast or (args.with_bau and not args.bau_exact):
        param_overrides["bau_fast"] = True
    elif args.bau_exact:
        param_overrides["bau_fast"] = False
    model_key = (effective_model or "").strip().lower()
    if model_key == "a4":
        if args.fast_nn_only:
            param_overrides.update({"chanWW": 0, "chanZZ": 0})
    elif args.fast_nn_only:
        raise ValueError("--fast-nn-only is currently implemented for model A4 only.")

    if model_key in {"z4", "vll"}:
        from models.Z4.z4_runner import print_z4_report, run_z4_from_cards

        if args.config:
            file_model, cards = load_native_config(args.config)
            model_name = args.model or file_model or "Z4"
            inputs_meta = {"config": args.config, "source": "native_config"}
        elif args.params_file and args.settings_file:
            file_model, cards = load_native_split(args.params_file, args.settings_file)
            model_name = args.model or file_model or "Z4"
            inputs_meta = {
                "params_file": args.params_file,
                "settings_file": args.settings_file,
                "source": "native_split",
            }
        else:
            if not (args.model_dir and args.param and args.settings):
                raise ValueError("Z4/VLL requires native params/settings files or legacy --model-dir/--param/--settings.")
            from nbe.wl_cards import load_wl_cards

            model_dir = Path(args.model_dir)
            param_path = Path(args.param)
            settings_path = Path(args.settings)
            if param_path.suffix.lower() != ".wl":
                param_path = param_path.with_suffix(".wl")
            if settings_path.suffix.lower() != ".wl":
                settings_path = settings_path.with_suffix(".wl")
            if not param_path.is_absolute():
                param_path = model_dir / param_path
            if not settings_path.is_absolute():
                settings_path = model_dir / settings_path
            cards = load_wl_cards(param_path, settings_path)
            model_name = args.model
            inputs_meta = {
                "model_dir": str(model_dir),
                "param_card": str(param_path),
                "settings_card": str(settings_path),
                "source": "legacy_wl_cards",
            }
        out = run_z4_from_cards(
            model_name=model_name,
            cards=cards,
            dof_file=args.dof_file,
            out_path=args.out,
            compare_mma_txt=args.compare_mma_txt,
            inputs_meta=inputs_meta,
        )
        print_z4_report(out, args.out)
        return

    if model_key in {"disp", "dispersion"}:
        if args.config:
            raise ValueError("DISP cBE short runner currently uses split params/settings files, not --config.")
        if not (args.params_file and args.settings_file):
            raise ValueError("DISP cBE runner requires params/settings files; use short form `relicflow.py DISP benchmark` or split-file flags.")
        from models.DISP.dispersion_cbe import print_dispersion_report, run_dispersion

        out = run_dispersion(
            params_file=Path(args.params_file),
            settings_file=Path(args.settings_file),
            dof_file=Path(args.dof_file),
            out_path=Path(args.out),
        )
        print_dispersion_report(out, args.out)
        return

    if args.config:
        out = run_nbe_workflow_from_config(
            config_path=args.config,
            dof_file=args.dof_file,
            out_json=args.out,
            model_name=args.model,
            param_overrides=param_overrides,
            compare_mma_txt=args.compare_mma_txt,
            include_bau=args.with_bau,
        )
        used_model = out.get("inputs", {}).get("model", args.model)
    elif args.params_file and args.settings_file:
        file_model, cards = load_native_split(args.params_file, args.settings_file)
        model_name = args.model or file_model
        if not model_name:
            raise ValueError("Model name is required: set --model or include 'model' in params/settings JSON.")
        out = run_nbe_workflow_from_cards(
            model_name=model_name,
            cards=cards,
            dof_file=args.dof_file,
            out_json=args.out,
            param_overrides=param_overrides,
            compare_mma_txt=args.compare_mma_txt,
            include_bau=args.with_bau,
            inputs_meta={
                "params_file": args.params_file,
                "settings_file": args.settings_file,
                "dof_file": args.dof_file,
                "source": "native_split",
            },
        )
        used_model = model_name
    else:
        if not (args.model and args.model_dir and args.param and args.settings):
            raise ValueError("Use --config, or --params-file/--settings-file, or legacy --model --model-dir --param --settings.")
        out = run_nbe_workflow(
            model_name=args.model,
            model_dir=args.model_dir,
            param_card=args.param,
            settings_card=args.settings,
            dof_file=args.dof_file,
            out_json=args.out,
            param_overrides=param_overrides,
            compare_mma_txt=args.compare_mma_txt,
            include_bau=args.with_bau,
        )
        used_model = args.model

    timing = out.get("timing_s", {})
    def _print_table(title: str, rows: list[list[str]], widths: list[int]) -> None:
        print(title)
        for row in rows:
            print("  " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)))

    def _abbr_channel(name: str) -> str:
        mapping = {
            "chiChi_N1": "N1",
            "chiChi_N2": "N2",
            "chiChi_N3": "N3",
            "chiChi_WW": "WW",
            "chiChi_ZZ": "ZZ",
            "chiEta": "CE",
            "etaEta_N1": "E1",
            "etaEta_N2": "E2",
            "etaEta_N3": "E3",
            "etaEta_WW": "EW",
            "etaEta_ZZ": "EZ",
        }
        return mapping.get(name, name)

    print("Timing [s]:")
    for k in ["load_cards", "load_model_adapter", "prepare_model", "load_dof", "build_relic_functions", "build_channels", "build_coann_channels", "build_svx", "solve_nbe", "postprocess_solution", "sample_eval", "reference_eval", "write_output"]:
        if k in timing:
            label = "build_channel_formulas" if k == "build_channels" else k
            print(f"  {label:>22}: {float(timing[k]):.6f} s")
    if "total" in timing:
        print(f"  {'total':>14}: {float(timing['total']):.6f} s")
    samples = out.get("samples", {})
    relic_meta = out.get("relicflow", {})
    coann_meta = relic_meta.get("coann") if isinstance(relic_meta, dict) else None
    diag_meta = relic_meta.get("diagnostics") if isinstance(relic_meta, dict) else None

    def _print_row(values: list[str], widths: list[int]) -> None:
        print("  " + " | ".join(str(v).ljust(w) for v, w in zip(values, widths)))

    def _print_fraction_table(title: str, x_keys: list[str], frac_tbl: dict, total_tbl: dict) -> None:
        if not isinstance(frac_tbl, dict) or not isinstance(total_tbl, dict):
            return
        channel_names: list[str] = []
        for x_key in x_keys:
            if x_key in frac_tbl and isinstance(frac_tbl[x_key], dict):
                channel_names = list(frac_tbl[x_key].keys())
                break
        if not channel_names:
            return

        print(title)
        headers = ["x", "total"]
        widths = [8, 14]
        for name in channel_names:
            headers.append(f"{name}%")
            widths.append(12)
        _print_row(headers, widths)
        _print_row(["-" * w for w in widths], widths)
        for x_key in x_keys:
            if x_key not in frac_tbl or x_key not in total_tbl:
                continue
            row_frac = frac_tbl[x_key]
            row = [f"x={x_key}", f"{float(total_tbl[x_key]):.3e}"]
            for name in channel_names:
                row.append(f"{100.0 * float(row_frac.get(name, 0.0)):8.2f}")
            _print_row(row, widths)

    def _v4_format_value(value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if not math.isfinite(number):
                return "nan" if math.isnan(number) else ("inf" if number > 0.0 else "-inf")
            if number == 0.0:
                return "0.0000"
            if abs(number) < 1.0 or abs(number) >= 1.0e5:
                return f"{number:.4e}"
            return f"{number:.4f}"
        if isinstance(value, (list, tuple)):
            if all(not isinstance(item, (dict, list)) for item in value):
                return "[" + ", ".join(_v4_format_value(item) for item in value) + "]"
            return f"<{len(value)} entries>"
        return str(value)

    def _v4_unit(key: str) -> str:
        if key.endswith("_GeV4"):
            return "GeV^4"
        if key.endswith("_GeV2"):
            return "GeV^2"
        if key.endswith("_GeV_inv"):
            return "GeV^-1"
        if key.endswith("_GeV_minus2"):
            return "GeV^-2"
        if key.endswith("_GeV"):
            return "GeV"
        if key in {"svx", "svx_xmax", "rate", "rate_max", "inverse_sigma_v_GeV_minus2"}:
            return "GeV^-2"
        if key in {"yeq", "Y", "Y_initial", "Y_final", "oh2_nbe", "omega_h2"}:
            return "dimensionless"
        return ""

    def _print_v4_mapping(
        mapping: dict,
        indent: str = "  ",
        skip: set[str] | None = None,
    ) -> None:
        skip = skip or set()
        for key, value in mapping.items():
            if key in skip:
                continue
            if isinstance(value, dict):
                print(f"{indent}{key}:")
                _print_v4_mapping(value, indent + "  ", skip)
                continue
            shown = _v4_format_value(value)
            unit = _v4_unit(str(key)) if isinstance(value, (int, float)) and not isinstance(value, bool) else ""
            print(f"{indent}{key}: {shown}" + (f" {unit}" if unit else ""))

    def _load_json_section(path: str | None, section: str) -> dict:
        if not path:
            return {}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        value = data.get(section)
        return dict(value) if isinstance(value, dict) else {}

    def _print_v4_report() -> None:
        params_card = _load_json_section(args.params_file, "params")
        settings_card = _load_json_section(args.settings_file, "settings")
        if args.config and not params_card:
            params_card = _load_json_section(args.config, "params")
            settings_card = _load_json_section(args.config, "settings")

        result = out.get("result", {})
        diagnostics = diag_meta if isinstance(diag_meta, dict) else {}
        print("V4 input parameters:")
        parameter_order = [
            "mDM_GeV",
            "LambdaF_GeV",
            "v_GeV",
            "xi",
            "f_GeV",
            "lambda_sigma",
            "gate_name",
            "dispersionMode",
            "phase_tracer",
            "transition_temperature_GeV",
            "transition_temperature_source",
        ]
        for key in parameter_order:
            if key not in params_card:
                continue
            value = params_card[key]
            unit = _v4_unit(key) if isinstance(value, (int, float)) and not isinstance(value, bool) else ""
            print(f"  {key}: {_v4_format_value(value)}" + (f" {unit}" if unit else ""))
        print("V4 relic result:")
        result_order = [
            "oh2_nbe",
            "runtime_s",
            "n_points",
            "svx_xmax",
            "Tp_GeV",
            "Tp_source",
            "x_p",
            "x_p_definition",
            "x_cd",
            "T_cd_GeV",
            "Gamma_chem_over_H_at_xcd",
            "x_cd_status",
            "x_cd_definition",
        ]
        for key in result_order:
            if key in result:
                unit = _v4_unit(key) if isinstance(result[key], (int, float)) and not isinstance(result[key], bool) else ""
                print(f"  {key}: {_v4_format_value(result[key])}" + (f" {unit}" if unit else ""))
        sigma_fields = [
            ("phase_mass_F_at_Tn_GeV", "m_sigma_F_at_Tn_GeV"),
            ("phase_mass_T_at_Tn_GeV", "m_sigma_T_at_Tn_GeV"),
        ]
        for source_key, shown_key in sigma_fields:
            if source_key not in diagnostics:
                continue
            value = diagnostics[source_key]
            print(f"  {shown_key}: {_v4_format_value(value)} GeV")
        xy = out.get("xy")
        if isinstance(xy, list) and xy and all(isinstance(row, list) and len(row) >= 2 for row in xy):
            print(f"  x_initial: {_v4_format_value(xy[0][0])}")
            print(f"  Y_initial: {_v4_format_value(xy[0][1])} dimensionless")
            print(f"  x_final: {_v4_format_value(xy[-1][0])}")
            print(f"  Y_final: {_v4_format_value(xy[-1][1])} dimensionless")

        comparison = out.get("comparison")
        if isinstance(comparison, dict):
            control = comparison.get("no_dispersion")
            control_result = control.get("result", {}) if isinstance(control, dict) else {}
            print("V4 no-dispersion control:")
            print(f"  status: {_v4_format_value(comparison.get('status'))}")
            print(f"  relative_difference_omega_h2: {_v4_format_value(comparison.get('relative_difference_omega_h2'))}")
            for key in ["oh2_nbe", "runtime_s", "n_points", "Tp_GeV", "x_p", "x_cd", "T_cd_GeV", "x_cd_status"]:
                if key not in control_result:
                    continue
                unit = _v4_unit(key)
                print(f"  {key}: {_v4_format_value(control_result[key])}" + (f" {unit}" if unit else ""))
            control_relicflow = control.get("relicflow", {}) if isinstance(control, dict) else {}
            control_diagnostics = control_relicflow.get("diagnostics", {}) if isinstance(control_relicflow, dict) else {}
            for source_key, shown_key in sigma_fields:
                if source_key not in control_diagnostics:
                    continue
                print(f"  {shown_key}: {_v4_format_value(control_diagnostics[source_key])} GeV")

    if isinstance(coann_meta, dict):
        print("Coann metadata:")
        print(f"  enabled: {bool(coann_meta.get('enabled', False))}")
        print(f"  deltaSame: {coann_meta.get('deltaSame')}")
        print(f"  deltaSameMax: {coann_meta.get('deltaSameMax')}")
        print(f"  gChi: {coann_meta.get('gChi')}")
        print(f"  gEta: {coann_meta.get('gEta')}")
        print(f"  mEta: {coann_meta.get('mEta')}")
        if "build_coann_channels_n" in relic_meta:
            print(f"  build_coann_channels_n: {relic_meta.get('build_coann_channels_n')}")
        if "build_coann_channels" in timing:
            print(f"  build_coann_channels_t[s]: {float(timing['build_coann_channels']):.3e}")
    if isinstance(diag_meta, dict) and str(used_model).strip().lower() not in {"v4", "dispersion_v4"}:
        print(f"{used_model} diagnostics:")
        diag_units = {
            "mDM": "GeV",
            "mEta": "GeV",
            "vphi": "GeV",
            "mZp": "GeV",
            "mh2": "GeV",
            "MN1": "GeV",
            "MN2": "GeV",
            "MN3": "GeV",
            "Gammah2": "GeV",
            "g1ChiChi": "GeV",
            "g2ChiChi": "GeV",
            "g1EtaEta": "GeV",
            "g2EtaEta": "GeV",
            "g1ZpZp_norm": "GeV",
            "g2ZpZp_norm": "GeV",
        }
        diag_order = [
            "mDM",
            "mEta",
            "vphi",
            "mZp",
            "mh2",
            "MN1",
            "MN2",
            "MN3",
            "Gammah2",
            "alphaMix",
            "A4lambdaChi",
            "A4lambdaEta",
            "g1ChiChi",
            "g2ChiChi",
            "g1EtaEta",
            "g2EtaEta",
            "g1ZpZp_norm",
            "g2ZpZp_norm",
            "scalarNorm_thr_re",
            "scalarNorm_thr_im",
            "abs_2_minus_scalarNorm_thr_sq",
        ]
        for key in diag_order:
            if key not in diag_meta:
                continue
            val = diag_meta.get(key)
            unit = diag_units.get(key)
            if isinstance(val, (int, float)):
                shown = f"{float(val):.6g}"
            else:
                shown = str(val)
            print(f"  {key}: {shown}" + (f" {unit}" if unit else ""))
    frac_tbl = samples.get("channel_frac")
    sv_tbl = samples.get("channel_svx")
    eff_comp_tbl = samples.get("sigma_eff_components")
    eff_comp_frac_tbl = samples.get("sigma_eff_component_frac")
    if isinstance(eff_comp_tbl, dict) and eff_comp_tbl and isinstance(eff_comp_frac_tbl, dict) and eff_comp_frac_tbl:
        print("Effective sigma components [coann]:")
        headers = ["x", "total", "wEta"]
        widths = [8, 14, 14]
        for comp_name in ["chiChi", "chiEta", "etaEta"]:
            headers.extend([f"{comp_name}%", comp_name])
            widths.extend([12, 14])
        _print_row(headers, widths)
        _print_row(["-" * w for w in widths], widths)
        for x_key in ["1", "3", "10", "30", "100"]:
            if x_key not in eff_comp_tbl or x_key not in eff_comp_frac_tbl:
                continue
            row_comp = eff_comp_tbl[x_key]
            row_frac = eff_comp_frac_tbl[x_key]
            row = [
                f"x={x_key}",
                f"{float(row_comp['total']):.3e}",
                f"{float(row_comp['wEta']):.3e}",
            ]
            for comp_name in ["chiChi", "chiEta", "etaEta"]:
                row.append(f"{100.0 * float(row_frac[comp_name]):10.2f}")
                row.append(f"{float(row_comp[comp_name]):.3e}")
            _print_row(row, widths)

    relic_contrib = out.get("relic_contrib")
    if isinstance(relic_contrib, dict):
        x_f = relic_contrib.get("x_f", 25.0)
        group_tbl = relic_contrib.get("group", {})
        if isinstance(group_tbl, dict) and group_tbl:
            print(f"Relic contributions [x_f={float(x_f):g}] to (Omega h^2)^-1:")
            headers = ["group", "J", "pct"]
            widths = [12, 16, 12]
            _print_row(headers, widths)
            _print_row(["-" * w for w in widths], widths)
            for name in ["chiChi", "chiEta", "etaEta"]:
                if name not in group_tbl:
                    continue
                row = [
                    name,
                    f"{float(group_tbl[name].get('J', 0.0)):.3e}",
                    f"{float(group_tbl[name].get('pct', 0.0)):10.2f}",
                ]
                _print_row(row, widths)

        nn_tbl = relic_contrib.get("nn_total", {})
        if isinstance(nn_tbl, dict) and nn_tbl:
            print(f"NN final-state contributions [x_f={float(x_f):g}] to total effective rate:")
            headers = ["final_state", "J", "pct_total", "pct_within_NN"]
            widths = [12, 16, 12, 14]
            _print_row(headers, widths)
            _print_row(["-" * w for w in widths], widths)
            for name in ["N1", "N2", "N3"]:
                if name not in nn_tbl:
                    continue
                row = [
                    name,
                    f"{float(nn_tbl[name].get('J', 0.0)):.3e}",
                    f"{float(nn_tbl[name].get('pct_total', 0.0)):10.2f}",
                    f"{float(nn_tbl[name].get('pct_within_NN', 0.0)):12.2f}",
                    ]
                _print_row(row, widths)
        else:
            # Compatibility with relicflow outputs generated before the
            # all-initial-state NN table was introduced.
            nn_tbl = relic_contrib.get("nn", {})
            if isinstance(nn_tbl, dict) and nn_tbl:
                print(f"NN chiChi-only contributions [x_f={float(x_f):g}]:")
                headers = ["channel", "J", "pct_total", "pct_in_chiChi"]
                widths = [12, 16, 12, 14]
                _print_row(headers, widths)
                _print_row(["-" * w for w in widths], widths)
                for name in ["chiChi_N1", "chiChi_N2", "chiChi_N3"]:
                    if name not in nn_tbl:
                        continue
                    row = [
                        name.replace("chiChi_", ""),
                        f"{float(nn_tbl[name].get('J', 0.0)):.3e}",
                        f"{float(nn_tbl[name].get('pct_total', 0.0)):10.2f}",
                        f"{float(nn_tbl[name].get('pct_within_chiChi', 0.0)):12.2f}",
                    ]
                    _print_row(row, widths)
        zp_tbl = relic_contrib.get("zp", {})
        if isinstance(zp_tbl, dict) and zp_tbl:
            print(f"ZpZp contribution [x_f={float(x_f):g}] to total and inside chiChi:")
            headers = ["channel", "J", "pct_total", "pct_in_chiChi"]
            widths = [12, 16, 12, 14]
            _print_row(headers, widths)
            _print_row(["-" * w for w in widths], widths)
            row = [
                "ZpZp",
                f"{float(zp_tbl.get('J', 0.0)):.3e}",
                f"{float(zp_tbl.get('pct_total', 0.0)):10.2f}",
                f"{float(zp_tbl.get('pct_within_chiChi', 0.0)):12.2f}",
            ]
            _print_row(row, widths)

    if str(used_model).strip().lower() in {"v4", "dispersion_v4"}:
        _print_v4_report()
    print(f"Saved: {args.out}")
    if str(used_model).strip().lower() == "a4":
        coann_samples = samples.get("sigma_eff_components", {})
        coann_on = False
        if isinstance(coann_samples, dict) and "1" in coann_samples:
            try:
                coann_on = float(coann_samples["1"].get("wEta", 0.0)) > 0.0
            except Exception:
                coann_on = False
        if args.with_wwzz:
            print("Mode: A4 full channels (WW/ZZ enabled)" + (" + same-family coann" if coann_on else ""))
        else:
            print("Mode: A4 default NN-only (chanWW=0, chanZZ=0)" + (" + same-family coann" if coann_on else ""))
    print(f"*** Oh2_nBE(py)={out['result']['oh2_nbe']:.12g} (dimensionless) ***")
    bau_out = out.get("bau")
    if isinstance(bau_out, dict):
        print(f"BAU status: {bau_out.get('status', 'unknown')}")
        print(f"BAU approximation: {bau_out.get('approximation', 'n/a')}")
        print(f"*** YB={float(bau_out.get('YB', 0.0)):.4e} (dimensionless) ***")
        eps_i = bau_out.get("epsilon_i")
        if isinstance(eps_i, list):
            print("CP asymmetry epsilon_i:")
            for idx, value in enumerate(eps_i, start=1):
                print(f"  N{idx}: {float(value):.4e}")
    ref = out["reference"]["oh2_nbe_ref"]
    if ref is not None:
        print(f"*** Oh2_nBE(ref)={ref:.12g} (dimensionless) ***")
    if out["reference"]["rel_err"] is not None:
        print(f"rel_err={out['reference']['rel_err']:.6%}")
    cmp_out = out.get("comparison")
    if isinstance(cmp_out, dict):
        if "oh2_rel_err" in cmp_out:
            print(f"compare_oh2_rel_err={float(cmp_out['oh2_rel_err']):.6%}")
        if "nbe_curve" in cmp_out:
            print("MMA comparison (nBE curve):")
            for k, v in cmp_out["nbe_curve"].items():
                print(f"  {k}: {v}")
        for block in ["seff_component_fractions", "seff_sigma_ratio", "nn_fractions_over_seff", "nn_sigma_ratio"]:
            if block in cmp_out:
                print(f"MMA comparison ({block}):")
                for metric, stats in cmp_out[block].items():
                    if stats.get("max_rel_err") is None:
                        print(f"  {metric}: n=0")
                    else:
                        print(
                            f"  {metric}: max={stats['max_rel_err']:.3e}, "
                            f"mean={stats['mean_rel_err']:.3e}, median={stats['median_rel_err']:.3e}"
                        )


if __name__ == "__main__":
    main()
