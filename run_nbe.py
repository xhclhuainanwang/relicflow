#!/usr/bin/env python
import argparse

from nbe import run_nbe_workflow, run_nbe_workflow_from_cards, run_nbe_workflow_from_config
from nbe.config import load_native_config, load_native_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic nBE mini-framework with pluggable models.")
    ap.add_argument("--model", type=str, default=None, help="Model adapter name, e.g. A4. Optional if config contains model.")
    ap.add_argument("--config", type=str, default=None, help="Native project config JSON (single-file mode).")
    ap.add_argument("--params-file", type=str, default=None, help="Native params JSON file (split-file mode).")
    ap.add_argument("--settings-file", type=str, default=None, help="Native settings JSON file (split-file mode).")
    ap.add_argument("--model-dir", type=str, default=None, help="Legacy: directory containing .wl parameter/settings cards")
    ap.add_argument("--param", type=str, default=None, help="Legacy: parameter card filename (.wl)")
    ap.add_argument("--settings", type=str, default=None, help="Legacy: settings card filename (.wl)")
    ap.add_argument("--dof-file", type=str, default="data/dof_Drees_etal.dat")
    ap.add_argument("--out", type=str, default="nbe_out.json")
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
    args = ap.parse_args()

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
    model_key = (effective_model or "").strip().lower()
    if model_key == "a4":
        if args.with_wwzz:
            pass
        else:
            param_overrides.update({"chanWW": 0, "chanZZ": 0})
    elif args.fast_nn_only:
        raise ValueError("--fast-nn-only is currently implemented for model A4 only.")

    if args.config:
        out = run_nbe_workflow_from_config(
            config_path=args.config,
            dof_file=args.dof_file,
            out_json=args.out,
            model_name=args.model,
            param_overrides=param_overrides,
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
        )
        used_model = args.model

    timing = out.get("timing_s", {})
    print("Timing [s]:")
    for k in ["load_cards", "prepare_model", "load_dof", "build_channels", "build_svx", "solve_nbe", "sample_eval", "reference_eval", "write_output"]:
        if k in timing:
            print(f"  {k:>14}: {float(timing[k]):.6f} s")
    ch_t = timing.get("build_svx_channels")
    if isinstance(ch_t, dict):
        t_sum = sum(float(v) for v in ch_t.values())
        print("  build_svx_channels:")
        for ch_name, sec in ch_t.items():
            frac = 0.0 if t_sum <= 0 else float(sec) / t_sum
            print(f"    {ch_name:>10}: {float(sec):.6f} s ({100.0*frac:.2f}%)")
    if "total" in timing:
        print(f"  {'total':>14}: {float(timing['total']):.6f} s")

    samples = out.get("samples", {})
    frac_tbl = samples.get("channel_frac")
    sv_tbl = samples.get("channel_svx")
    if isinstance(frac_tbl, dict) and isinstance(sv_tbl, dict):
        print("Channel fractions in <sigma v> [GeV^-2]:")
        for x_key in ["1", "3", "10", "30", "100"]:
            if x_key not in frac_tbl or x_key not in sv_tbl:
                continue
            row_frac = frac_tbl[x_key]
            row_svx = sv_tbl[x_key]
            tot = out.get("samples", {}).get("svx", {}).get(x_key)
            if tot is None:
                continue
            print(f"  x={x_key}: total={float(tot):.6e} GeV^-2")
            for ch_name in sorted(row_frac.keys()):
                print(f"    {ch_name:>10}: {100.0*float(row_frac[ch_name]):6.2f}% ({float(row_svx[ch_name]):.6e} GeV^-2)")

    print(f"Saved: {args.out}")
    if str(used_model).strip().lower() == "a4":
        if args.with_wwzz:
            print("Mode: A4 full channels (WW/ZZ enabled)")
        else:
            print("Mode: A4 default NN-only (chanWW=0, chanZZ=0)")
    print(f"Oh2_nBE(py)={out['result']['oh2_nbe']:.12g} (dimensionless)")
    ref = out["reference"]["oh2_nbe_ref"]
    if ref is not None:
        print(f"Oh2_nBE(ref)={ref:.12g} (dimensionless)")
    if out["reference"]["rel_err"] is not None:
        print(f"rel_err={out['reference']['rel_err']:.6%}")


if __name__ == "__main__":
    main()
