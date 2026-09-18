"""Lazy ctypes bridge for the C++ Z4/VLL hot paths.

The DLL is a generated build artifact.  This module keeps the model usable on
machines without MSVC by returning ``None`` from :func:`get_backend`, in which
case the existing SciPy implementation remains active.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "z4_native.cpp"
_DLL = _ROOT / "z4_native.dll"
_VERSION = 1

_backend: "Z4NativeBackend | None" = None
_attempted = False
_last_error: str | None = None


def _find_vsdevcmd() -> Path | None:
    env_path = os.environ.get("Z4_NATIVE_VSDEVCMD", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"),
        ]
    )
    for root in (
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ):
        if root.is_dir():
            candidates.extend(root.glob(r"*\BuildTools\Common7\Tools\VsDevCmd.bat"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _compile_dll() -> bool:
    global _last_error
    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        _last_error = "Visual Studio VsDevCmd.bat was not found"
        return False
    compile_line = (
        f'"{vsdevcmd}" -arch=x64 -host_arch=x64 && '
        f'cl /nologo /O2 /EHsc /std:c++17 /openmp /LD /Fe:"{_DLL}" "{_SOURCE}"'
    )
    try:
        result = subprocess.run(
            compile_line,
            cwd=str(_ROOT),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - platform/toolchain dependent
        _last_error = f"native compile failed to start: {exc}"
        return False
    if result.returncode != 0 or not _DLL.is_file():
        diagnostic = (result.stdout + "\n" + result.stderr).strip()
        _last_error = f"native compile returned {result.returncode}: {diagnostic[-2000:]}"
        return False
    return True


def _ensure_dll() -> Path | None:
    global _last_error
    if os.environ.get("Z4_NATIVE_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        _last_error = "native backend disabled by Z4_NATIVE_DISABLE"
        return None
    try:
        needs_build = not _DLL.is_file() or _SOURCE.stat().st_mtime_ns > _DLL.stat().st_mtime_ns
    except OSError as exc:
        _last_error = f"native artifact check failed: {exc}"
        return None
    if needs_build and not _compile_dll():
        return None
    return _DLL if _DLL.is_file() else None


class Z4NativeBackend:
    """Small, typed wrapper around ``z4_native.dll``."""

    name = "cpp"

    def __init__(self, dll_path: Path):
        self.path = dll_path
        self._lib = ctypes.CDLL(str(dll_path))
        self._lib.z4_native_version.argtypes = []
        self._lib.z4_native_version.restype = ctypes.c_int
        if int(self._lib.z4_native_version()) != _VERSION:
            raise RuntimeError("unsupported z4_native.dll ABI version")

        double_ptr = ctypes.POINTER(ctypes.c_double)
        self._lib.z4_rates.argtypes = [ctypes.c_int, double_ptr, double_ptr, ctypes.c_int, double_ptr]
        self._lib.z4_rates.restype = None
        self._lib.z4_full_cell.argtypes = [
            double_ptr,
            double_ptr,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            double_ptr,
        ]
        self._lib.z4_full_cell.restype = None
        self._lib.z4_rel_temp.argtypes = [ctypes.c_int, double_ptr, double_ptr]
        self._lib.z4_rel_temp.restype = None

    def thermal_rates(self, params: np.ndarray, xs: np.ndarray, pg: int) -> np.ndarray:
        params_arr = np.ascontiguousarray(params, dtype=np.float64).reshape(6)
        x_arr = np.ascontiguousarray(xs, dtype=np.float64).reshape(-1)
        out = np.empty((x_arr.size, 6), dtype=np.float64)
        self._lib.z4_rates(
            ctypes.c_int(int(x_arr.size)),
            x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            params_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(int(pg)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out

    def full_cell(
        self,
        params: np.ndarray,
        config: np.ndarray,
        xi: float,
        yeq: float,
        y: float,
        exp_dbk2: float,
        der_dbk2: float,
    ) -> tuple[float, float]:
        params_arr = np.ascontiguousarray(params, dtype=np.float64).reshape(6)
        config_arr = np.ascontiguousarray(config, dtype=np.float64).reshape(6)
        out = np.empty(2, dtype=np.float64)
        self._lib.z4_full_cell(
            params_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            config_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(float(xi)),
            ctypes.c_double(float(yeq)),
            ctypes.c_double(float(y)),
            ctypes.c_double(float(exp_dbk2)),
            ctypes.c_double(float(der_dbk2)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return float(out[0]), float(out[1])

    def rel_temp(self, xs: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(xs, dtype=np.float64).reshape(-1)
        out = np.empty((x_arr.size, 2), dtype=np.float64)
        self._lib.z4_rel_temp(
            ctypes.c_int(int(x_arr.size)),
            x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out


def get_backend() -> Z4NativeBackend | None:
    """Return the cached native backend, compiling it on first use if needed."""

    global _backend, _attempted, _last_error
    if _attempted:
        return _backend
    _attempted = True
    dll_path = _ensure_dll()
    if dll_path is None:
        return None
    try:
        _backend = Z4NativeBackend(dll_path)
    except Exception as exc:  # pragma: no cover - platform/ABI dependent
        _last_error = f"native DLL load failed: {exc}"
        _backend = None
    return _backend


def backend_status() -> dict[str, Any]:
    """Return diagnostics without forcing callers to inspect ctypes state."""

    backend = get_backend()
    return {
        "name": backend.name if backend is not None else "python",
        "available": backend is not None,
        "dll": str(_DLL),
        "error": _last_error,
    }
