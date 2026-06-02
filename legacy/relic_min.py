#!/usr/bin/env python
"""
Backward-compatible entrypoint for the previous single-file tool.
Internally it now uses the modular nBE mini-framework.
"""

import argparse

from nbe import run_nbe_workflow


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal nBE CLI (compat mode), now backed by modular framework.")
    ap.add_argument("--model-dir", type=str, default="models/A4")
    ap.add_argument("--param", type=str, default="parameters1.wl")
    ap.add_argument("--settings", type=str, default="settings.wl")
    ap.add_argument("--out", type=str, default="out.json")
    ap.add_argument("--dof-file", type=str, default="data/dof_Drees_etal.dat")
    ap.add_argument("--model", type=str, default="A4")
    ap.add_argument("--with-wwzz", action="store_true", help="A4 mode: enable WW/ZZ channels (default NN-only).")
    ap.add_argument("--fast-nn-only", action="store_true", help="A4 fast mode: chanWW=0, chanZZ=0")
    args = ap.parse_args()

    param_overrides = {}
    if args.model.strip().lower() == "a4":
        if args.with_wwzz:
            pass
        else:
            param_overrides.update({"chanWW": 0, "chanZZ": 0})
    elif args.fast_nn_only:
        raise ValueError("--fast-nn-only is currently implemented for model A4 only.")

    out = run_nbe_workflow(
        model_name=args.model,
        model_dir=args.model_dir,
        param_card=args.param,
        settings_card=args.settings,
        dof_file=args.dof_file,
        out_json=args.out,
        param_overrides=param_overrides,
    )

    print(f"Saved: {args.out}")
    if args.model.strip().lower() == "a4":
        if args.with_wwzz:
            print("Mode: A4 full channels (WW/ZZ enabled)")
        else:
            print("Mode: A4 default NN-only (chanWW=0, chanZZ=0)")
    print(f"Oh2_nBE(py)={out['result']['oh2_nbe']:.12g}")
    ref = out["reference"]["oh2_nbe_ref"]
    if ref is not None:
        print(f"Oh2_nBE(ref)={ref:.12g}")
    if out["reference"]["rel_err"] is not None:
        print(f"rel_err={out['reference']['rel_err']:.6%}")


if __name__ == "__main__":
    main()
