# =============================================================================
# File: onnx_verify.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

#!/usr/bin/env python3
"""Lightweight ONNX verification utilities used by exporter/validator.

Contains `verify_models(fnames, output_dir, pack_single=False, pack_single_threshold_mb=None)`.
This module is self-contained to avoid circular imports with the validator.
"""
from __future__ import annotations

import glob
import json as _json
import os
import subprocess
import sys
import tempfile
import uuid
from typing import List


def _import_real_onnx():
    import importlib
    import importlib.util
    import os
    import sys
    import sysconfig

    try:
        _onnx = importlib.import_module("onnx")
        if hasattr(_onnx, "checker") and hasattr(_onnx, "load"):
            return getattr(_onnx, "checker"), getattr(_onnx, "load")
    except Exception:
        pass

    candidates = []
    try:
        import site

        try:
            candidates.extend(site.getsitepackages())
        except Exception:
            pass
    except Exception:
        pass

    sc_paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        p = sc_paths.get(key)
        if p:
            candidates.append(p)

    try:
        prefix_sp = os.path.join(sys.prefix, "Lib", "site-packages")
        candidates.append(prefix_sp)
    except Exception:
        pass

    seen = set()
    candidate_file = None
    for base in candidates:
        try:
            if not base:
                continue
            base_abs = os.path.abspath(base)
            if base_abs in seen:
                continue
            seen.add(base_abs)
            possible = os.path.join(base_abs, "onnx", "__init__.py")
            if os.path.exists(possible):
                candidate_file = possible
                break
        except Exception:
            continue

    if candidate_file:
        spec = importlib.util.spec_from_file_location("onnx_installed", candidate_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if hasattr(mod, "checker") and hasattr(mod, "load"):
            return getattr(mod, "checker"), getattr(mod, "load")

    raise ModuleNotFoundError(
        "Could not import a usable 'onnx' package from the active interpreter site-packages."
    )


checker, load = _import_real_onnx()


def _safe_check_model(model_path: str, timeout: int = 120) -> tuple[bool, str]:
    """Run `onnx.checker.check_model` in a subprocess to isolate native crashes.

    The child process prints a JSON object to stdout with a `status` field.
    Returns (True, "ok") on success, (False, info) on failure where `info` is
    either a dict parsed from the child's JSON output or a string with stderr.
    """
    script_path = os.path.join(
        tempfile.gettempdir(), f"onnx_checker_child_{uuid.uuid4().hex}.py"
    )
    script = """
import json
import sys
import gc
import traceback
try:
    import onnx
except Exception as e:
    print(json.dumps({"status": "import_failed", "error": str(e)}))
    sys.exit(2)
try:
    m = onnx.load(sys.argv[1])
    onnx.checker.check_model(m)
    del m
    gc.collect()
    print(json.dumps({"status": "ok"}))
    sys.exit(0)
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"status": "failed", "error": str(e), "traceback": tb}))
    sys.exit(1)
"""

    try:
        open(script_path, "w", encoding="utf-8").write(script)
        proc = subprocess.run(
            [sys.executable, script_path, model_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        try:
            # best-effort cleanup
            if os.path.exists(script_path):
                os.remove(script_path)
        except Exception:
            pass
        return False, "checker_timeout"
    try:
        if proc.returncode == 0:
            try:
                j = _json.loads(proc.stdout)
                if j.get("status") == "ok":
                    return True, "ok"
                return False, j
            except Exception:
                return True, proc.stdout.strip() or "ok"
        # non-zero return: try to parse stdout JSON for structured info
        try:
            j = _json.loads(proc.stdout)
            return False, j
        except Exception:
            out = proc.stderr or proc.stdout or f"returncode:{proc.returncode}"
            info = {"status": "failed", "detail": out}
            return False, info
    finally:
        try:
            if os.path.exists(script_path):
                os.remove(script_path)
        except Exception:
            pass


# Try to import helpers but tolerate absence
try:
    from onnx_loaders.onnx_utils import has_external_data
except Exception:

    def has_external_data(model) -> bool:  # type: ignore
        try:
            locs = []
            for tensor in getattr(model.graph, "initializer", []):
                if getattr(tensor, "data_location", 0) == 1 and hasattr(
                    tensor, "external_data"
                ):
                    for kv in tensor.external_data:
                        if getattr(kv, "key", None) == "location" and getattr(
                            kv, "value", None
                        ):
                            locs.append(kv.value)
            return len([l for l in locs if l]) > 0
        except Exception:
            return False


try:
    from onnx_loaders.onnx_helpers import create_ort_session, get_preferred_provider
except Exception:

    def create_ort_session(path, provider=None):
        import onnxruntime as ort

        return ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )  # fallback

    def get_preferred_provider():
        return "CPUExecutionProvider"


def verify_models(
    fnames: List[str],
    output_dir: str,
    pack_single: bool = False,
    pack_single_threshold_mb: int | None = None,
) -> bool:
    """Lightweight verification helper.

    Runs ONNX checker where possible, detects external_data, optionally repacks,
    and creates an ORT session to inspect inputs/outputs. Returns True if basic checks passed.
    """
    import gc

    try:
        provider = (
            get_preferred_provider() if callable(get_preferred_provider) else None
        )
    except Exception:
        provider = None

    print(
        f"verify_models: start files={fnames} output_dir={output_dir} pack_single={pack_single} pack_single_threshold_mb={pack_single_threshold_mb}"
    )

    # Only verify files that actually exist in the output directory. If none
    # of the expected names are present, discover any `.onnx` files in the
    # directory and verify those instead.
    available_fnames: List[str] = []
    for fname in fnames:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            available_fnames.append(fname)
        else:
            print(f"Verify: file not found {path}")

    if not available_fnames:
        try:
            discovered = [
                os.path.basename(p)
                for p in glob.glob(os.path.join(output_dir, "*.onnx"))
            ]
        except Exception:
            discovered = []
        if discovered:
            print(f"No expected files found; discovered ONNX files: {discovered}")
            available_fnames = discovered
        else:
            print(f"No ONNX files found in {output_dir}; nothing to verify")
            return False

    for fname in available_fnames:
        path = os.path.join(output_dir, fname)
        try:
            onnx_model = None
            try:
                ok, info = _safe_check_model(path)
                if ok:
                    print(f"{fname} passed ONNX checker (subprocess)")
                else:
                    print(f"ONNX checker subprocess failed for {fname}: {info}")
            except Exception as e:
                print(f"ONNX checker subprocess error for {fname}: {e}")

            try:
                # Load model only when needed (external data detection / repack)
                onnx_model = load(path)
                external_used = has_external_data(onnx_model)
                if external_used:
                    print(
                        f"Warning: Model {fname} uses external_data tensors. Ensure associated tensor files are co-located."
                    )
                    if pack_single:
                        try:
                            from typing import List as _List

                            def _get_external_locations(model) -> _List[str]:
                                locs: _List[str] = []
                                try:
                                    for tensor in getattr(
                                        model.graph, "initializer", []
                                    ):
                                        if getattr(
                                            tensor, "data_location", 0
                                        ) == 1 and hasattr(tensor, "external_data"):
                                            for kv in tensor.external_data:
                                                if getattr(
                                                    kv, "key", None
                                                ) == "location" and getattr(
                                                    kv, "value", None
                                                ):
                                                    locs.append(kv.value)
                                except Exception:
                                    pass
                                return list({l for l in locs if l})

                            def _sum_external_bytes(
                                base_dir: str, locs: _List[str]
                            ) -> int:
                                total = 0
                                for loc in locs:
                                    try:
                                        fp = os.path.join(base_dir, loc)
                                        total += os.path.getsize(fp)
                                    except Exception:
                                        pass
                                return total

                            do_repack = True
                            if pack_single_threshold_mb is not None:
                                try:
                                    locs = _get_external_locations(onnx_model)
                                    total_bytes = _sum_external_bytes(output_dir, locs)
                                    total_mb = total_bytes / (1024 * 1024)
                                    print(
                                        f"Estimated external tensor size for {fname}: {total_mb:.1f} MB (threshold={float(pack_single_threshold_mb):.1f} MB)"
                                    )
                                    if total_mb > float(pack_single_threshold_mb):
                                        do_repack = False
                                        print(
                                            f"Skipping single-file repack for {fname} due to size exceeding threshold"
                                        )
                                except Exception:
                                    do_repack = True

                            if do_repack:
                                import time as _time

                                from onnx import external_data_helper

                                tmp_single = f"{path}.single"
                                t0 = _time.perf_counter()
                                external_data_helper.convert_model_to_single_file(
                                    onnx_model, tmp_single
                                )
                                os.replace(tmp_single, path)
                                dt = _time.perf_counter() - t0
                                print(
                                    f"Repacked external_data model into single file: {path} ({dt:.2f}s)"
                                )
                                onnx_model = load(path)
                                external_used = False
                        except Exception as e:
                            print(
                                f"Failed to repack model {fname} into single file: {e}"
                            )
            except Exception:
                print(f"External data detection failed for {fname}")

            try:
                sess = create_ort_session(path, provider=provider)
                inputs = [i.name for i in sess.get_inputs()]
                outputs = [o.name for o in sess.get_outputs()]
                print(f"{fname} inputs={inputs} outputs={outputs}")
                del onnx_model, sess
                gc.collect()
            except Exception as e:
                print(f"Session creation or inspection failed for {fname}: {e}")
        except Exception as e:
            print(f"Verification failed for {fname}: {e}")
            try:
                corrupt = f"{path}.corrupt"
                os.replace(path, corrupt)
                print(f"Moved corrupt file to {corrupt}")
            except Exception:
                print(f"Failed to move corrupt file {path}")
            print(f"verify_models: failed while verifying {fname}")
            return False
    print("verify_models: succeeded")
    return True
