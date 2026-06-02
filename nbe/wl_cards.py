import math
import re
from pathlib import Path

import numpy as np


def strip_mma_comments(text: str) -> str:
    return re.sub(r"\(\*.*?\*\)", "", text, flags=re.DOTALL)


def mma_to_python_expr(expr: str) -> str:
    s = expr.strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("*^", "e")
    s = s.replace("^", "**")
    s = re.sub(r"\bI\b", "1j", s)
    s = re.sub(r"\bPi\b", "np.pi", s)

    func_map = {
        "Abs": "abs",
        "Sqrt": "np.sqrt",
        "Sin": "np.sin",
        "Cos": "np.cos",
        "Tan": "np.tan",
        "Exp": "np.exp",
        "Log": "np.log",
        "Re": "np.real",
        "Im": "np.imag",
    }
    for mma_name, py_name in func_map.items():
        patt = re.compile(rf"\b{mma_name}\[(.*?)\]")
        while True:
            s_new, n = patt.subn(lambda m: f"{py_name}({m.group(1)})", s)
            s = s_new
            if n == 0:
                break

    s = s.replace("{", "[").replace("}", "]")
    s = re.sub(r"(?<=[0-9A-Za-z_\)\]])\s+(?=[A-Za-z_(\[])", "*", s)
    s = re.sub(r"(?<=[A-Za-z_\)\]])\s+(?=[0-9])", "*", s)
    return s


def safe_eval_expr(expr: str, env: dict) -> object:
    eval_env = {
        "np": np,
        "math": math,
        "abs": abs,
    }
    eval_env.update(env)
    return eval(expr, {"__builtins__": {}}, eval_env)


def load_wl_cards(param_path: str | Path, settings_path: str | Path) -> dict:
    out: dict = {}
    for p in [Path(param_path), Path(settings_path)]:
        text = strip_mma_comments(p.read_text(encoding="utf-8"))
        statements = [x.strip() for x in text.split(";") if x.strip()]
        for st in statements:
            m = re.match(r"^([A-Za-z]\w*)\s*=\s*(.+)$", st, flags=re.DOTALL)
            if not m:
                continue
            key = m.group(1)
            raw = m.group(2).strip()
            py_expr = mma_to_python_expr(raw)
            try:
                val = safe_eval_expr(py_expr, out)
            except Exception:
                continue
            out[key] = val
    return out

