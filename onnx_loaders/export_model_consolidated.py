# =============================================================================
# File: export_model_consolidated.py
# Date: 2026-01-06
# Consolidated exporter that unifies behaviors from v1 and v2.
# Defaults to v2-style export but preserves a v1 compatibility mode via `use_v1=True`.
# =============================================================================

import gc
import os
import pathlib
import traceback
from pathlib import Path
from typing import Any, List, cast


def _import_real_onnx():
    """Import the installed `onnx` package while avoiding a local `onnx/` folder
    in the repository that may shadow the real package. This temporarily removes
    the repository root from sys.path during the import and then restores it.
    Returns the `checker` and `load` callables from the imported package.
    """
    # Robust import that prefers the installed package under the active
    # interpreter's site-packages (useful when a local `onnx/` folder exists
    # in the repository and may shadow the installed package). Strategy:
    # 1) try a normal import and verify it exposes expected symbols
    # 2) inspect standard site-packages locations for an `onnx/__init__.py`
    #    belonging to the active interpreter and load it directly
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

    # Candidate site-packages locations to search
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

    # include the venv prefix site-packages on Windows
    try:
        prefix_sp = os.path.join(sys.prefix, "Lib", "site-packages")
        candidates.append(prefix_sp)
    except Exception:
        pass

    # De-duplicate and check existence
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

from .fine_tune_t5 import fine_tune_t5_embeddings
from .onnx_helpers import (
    create_ort_session,
    get_default_opset,
    get_logger,
    get_preferred_provider,
)
from .onnx_utils import has_external_data

logger = get_logger(__name__)


def _configure_protobuf():
    """Try a few strategies to increase protobuf message size limits for large models."""
    try:
        import google.protobuf.message

        msg_cls = cast(Any, google.protobuf.message.Message)
        if hasattr(msg_cls, "_SetGlobalDefaultMaxMessageSize"):
            msg_cls._SetGlobalDefaultMaxMessageSize(2**31 - 1)  # type: ignore[attr-defined]
            logger.info(
                "Protobuf message size limit set using _SetGlobalDefaultMaxMessageSize"
            )
        elif hasattr(msg_cls, "SetGlobalDefaultMaxMessageSize"):
            msg_cls.SetGlobalDefaultMaxMessageSize(2**31 - 1)  # type: ignore[attr-defined]
            logger.info(
                "Protobuf message size limit set using SetGlobalDefaultMaxMessageSize"
            )
        else:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
            logger.warning(
                "Using environment variable fallback for protobuf configuration"
            )
    except Exception as e:
        logger.warning("Could not configure protobuf message size: %s", e)
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"


def export_and_optimize_onnx_unified(
    model_name: str,
    model_for: str = "fe",
    optimize: bool = False,
    optimization_level: int = 1,
    task: str | None = None,
    model_folder: str = None,
    onnx_path: str = "onnx",
    opset_version: int | None = None,
    use_t5_encoder: bool = False,
    use_cache: bool = False,
    finetine: bool = False,
    force: bool = False,
    use_v1: bool = False,
    pack_single_file: bool = False,
    pack_single_threshold_mb: int | None = 1536,
    framework: str | None = None,
):
    """Unified exporter.

    - Defaults to v2 behavior (main_export path + robust fallbacks).
    - If `use_v1=True`, uses the legacy ORTModel.from_pretrained(export=True) path.
    """
    # Configure protobuf and environment
    _configure_protobuf()

    if opset_version is None:
        opset_version = get_default_opset()

    if not model_name or not model_name.strip():
        raise ValueError("model_name cannot be empty")

    _model_for = model_for.lower()
    if _model_for not in [
        "fe",
        "s2s",
        "sc",
        "llm",
        "feature-extraction",
        "seq2seq-lm",
        "sequence-classification",
    ]:
        raise ValueError(f"Invalid model_for: {model_for}")

    logger.info(
        "Unified exporter: model=%s for=%s use_v1=%s", model_name, model_for, use_v1
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.normpath(onnx_path)
    if ".." in onnx_path:
        raise ValueError("Path traversal detected in onnx_path")

    _output_base = os.path.join(onnx_path, "models", _model_for)
    if not model_folder:
        model_folder = model_name.split("/")[-1] if "/" in model_name else model_name
    model_folder = os.path.basename(model_folder)
    _output_dir = os.path.join(BASE_DIR, _output_base, model_folder)
    Path(_output_dir).mkdir(parents=True, exist_ok=True)

    # If user requested fine-tuning and the model appears to be T5-based, run fine-tune first.
    # The fine-tuned model will be written to a subfolder under the output dir and used as
    # the export source for subsequent steps.
    if finetine and ("t5" in (model_name or "").lower()):
        finetune_dir = os.path.join(_output_dir, f"{model_folder}_finetuned")
        Path(finetune_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Finetuning requested and model looks like T5; fine-tuning to %s",
            finetune_dir,
        )
        try:
            fine_tune_t5_embeddings(model_name, finetune_dir)
            # Use the fine-tuned local folder as the export source
            model_name = finetune_dir
            logger.info("Using fine-tuned model at %s for export", finetune_dir)
        except Exception as e:
            logger.exception("Fine-tuning failed: %s", e)
            raise

    # Prevent concurrent exports for the same model using a lock file.
    lock_path = os.path.join(_output_dir, ".export.lock")
    created_lock = False
    try:
        import time

        def _is_pid_running(pid: int) -> bool:
            try:
                import psutil

                return psutil.pid_exists(pid)
            except Exception:
                try:
                    # portable fallback: sending signal 0 to check for existence
                    os.kill(pid, 0)
                except Exception:
                    return False
                else:
                    return True

        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r", encoding="utf-8") as fh:
                    lines = [l.strip() for l in fh.readlines() if l.strip()]
                pid = int(lines[0]) if lines else None
                ts = float(lines[1]) if len(lines) > 1 else None
            except Exception:
                pid = None
                ts = None

            now = time.time()
            stale_threshold = 24 * 60 * 60  # 24 hours
            if pid is not None and _is_pid_running(pid):
                logger.info(
                    "An export for this model is already running (pid=%s). Exiting.",
                    pid,
                )
                raise SystemExit(0)
            else:
                # If lock is stale (old) or recent but owner not running, remove it and continue
                age = (now - ts) if ts else None
                if age is None or age > stale_threshold:
                    try:
                        os.remove(lock_path)
                        logger.info(
                            "Removed stale lock file (age=%.0fs): %s",
                            age if age else 0,
                            lock_path,
                        )
                    except Exception:
                        logger.warning(
                            "Could not remove stale lock file: %s", lock_path
                        )
                else:
                    # Recent lock but owner not running; remove it and continue (retry-friendly)
                    try:
                        os.remove(lock_path)
                        logger.info(
                            "Found recent lock file but owner not running; removing and continuing: %s",
                            lock_path,
                        )
                    except Exception:
                        logger.warning(
                            "Could not remove recent lock file; exiting to avoid race: %s",
                            lock_path,
                        )
                        raise SystemExit(0)

        # Create lock file exclusively
        try:
            # 'x' mode will fail if file exists
            with open(lock_path, "x", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()}\n{time.time()}\n{model_name}\n")
            created_lock = True
            logger.info("Acquired export lock: %s", lock_path)
        except FileExistsError:
            logger.info("Lock file created by concurrent process; exiting")
            raise SystemExit(0)
    except SystemExit:
        # re-raise to allow graceful exit
        raise
    except Exception:
        # If lock management fails, log and continue (best-effort)
        logger.warning(
            "Could not create or check export lock; proceeding without exclusive lock"
        )

    # Pre-export memory cleanup: try to free Python and GPU memory to maximize
    # available resources for large model exports/optimization.
    try:
        gc.collect()
        try:
            import torch

            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache before export to free GPU memory")
        except Exception:
            # torch not installed or CUDA unavailable — continue silently
            logger.debug("No torch/CUDA available for pre-export cache clear")

        try:
            import psutil

            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            logger.info("Available memory before export: %.1f MB", available_mb)
        except Exception:
            logger.debug("psutil not available for memory check before export")
    except Exception:
        logger.debug("Pre-export cleanup failed", exc_info=True)

    _onnx_name = "model.onnx"
    _encoder_onnx_name = "encoder_model.onnx"
    _decoder_onnx_name = "decoder_model.onnx"
    _decoder_with_past_name = "decoder_with_past_model.onnx"

    # Setup per-run file logging so large logs can be inspected per-export.
    file_handler = None
    _logfile_fd = None
    _stdout_orig = None
    _stderr_orig = None
    try:
        import logging
        import logging.handlers
        import time

        # Place logs under the repository-level `logs/onnx_exports` folder
        logs_dir = Path(BASE_DIR).parent / "logs" / "onnx_exports"
        logs_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_model = model_folder.replace("/", "_").replace("\\", "_")

        # Try to detect a model revision if specified (e.g., 'repo@rev' or 'repo:rev'),
        # otherwise attempt a lightweight repo query via huggingface_hub if available.
        model_rev = None
        try:
            if "@" in model_name:
                model_rev = model_name.split("@", 1)[1]
            elif ":" in model_name:
                model_rev = model_name.split(":", 1)[1]
            else:
                from huggingface_hub import HfApi

                if "/" in model_name:
                    try:
                        info = HfApi().repo_info(model_name)
                        model_rev = getattr(info, "sha", None) or getattr(
                            info, "revision", None
                        )
                    except Exception:
                        model_rev = None
        except Exception:
            model_rev = None

        rev_tag = model_rev or "local"
        logfile = logs_dir / f"{safe_model}_{rev_tag}_{ts}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            str(logfile), maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        # Avoid duplicate propagation to root handlers for cleaner output
        logger.propagate = False
        logger.info("Logging to file: %s", logfile)
        # Also capture anything written to stdout/stderr (prints, third-party libs)
        try:
            import sys

            class Tee:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, data):
                    for s in self.streams:
                        try:
                            s.write(data)
                        except Exception:
                            pass

                def flush(self):
                    for s in self.streams:
                        try:
                            s.flush()
                        except Exception:
                            pass

                def isatty(self):
                    for s in self.streams:
                        try:
                            if s and getattr(s, "isatty", lambda: False)():
                                return True
                        except Exception:
                            pass
                    return False

            _logfile_fd = open(logfile, "a", encoding="utf-8")
            _stdout_orig = sys.stdout
            _stderr_orig = sys.stderr
            sys.stdout = Tee(sys.stdout, _logfile_fd)
            sys.stderr = Tee(sys.stderr, _logfile_fd)
        except Exception:
            logger.warning(
                "Failed to tee stdout/stderr to logfile; continuing without stream capture"
            )
    except Exception:
        # If file logging setup fails, continue without file handler
        logger.warning(
            "Failed to initialize per-run file logging; continuing without it"
        )

    # Pre-compute expected outputs to support early-skip / force re-export behavior
    if _model_for in ["s2s", "seq2seq-lm"]:
        _expected_pre = [_encoder_onnx_name, _decoder_onnx_name]
        if use_cache:
            _expected_pre.append(_decoder_with_past_name)
    else:
        _expected_pre = [_onnx_name]

    # If all expected outputs already exist and user didn't request force, skip export.
    try:
        all_exist = all(
            os.path.exists(os.path.join(_output_dir, fname)) for fname in _expected_pre
        )
    except Exception:
        all_exist = False

    if all_exist and not force:
        logger.info(
            "All expected ONNX files already exist in %s — skipping export (use --force to re-export)",
            _output_dir,
        )
        logger.info("Process log: export step skipped because outputs already present")
        return _output_dir

    # If force requested, remove existing output dir to ensure a clean export
    if force and os.path.exists(_output_dir):
        try:
            import shutil

            shutil.rmtree(_output_dir)
            Path(_output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(
                "Removed existing output directory because --force was given: %s",
                _output_dir,
            )
            logger.info("Process log: cleaned output directory as part of --force")
        except Exception as e:
            logger.warning("Failed to remove existing output dir with --force: %s", e)
            # Best-effort: clear files inside to avoid stale artifacts
            try:
                import pathlib

                for p in pathlib.Path(_output_dir).glob("**/*"):
                    try:
                        if p.is_file() or p.is_symlink():
                            p.unlink(missing_ok=True)  # type: ignore[arg-type]
                        elif p.is_dir():
                            p.rmdir()
                    except Exception:
                        pass
                logger.info("Best-effort cleanup completed despite rmtree failure")
            except Exception:
                logger.warning(
                    "Best-effort cleanup could not complete; export will attempt to overwrite files"
                )

    def _verify_models(
        fnames: List[str],
        output_dir: str,
        pack_single: bool = False,
        pack_single_threshold_mb: int | None = None,
    ) -> bool:
        provider = get_preferred_provider()
        missing = False
        for fname in fnames:
            path = os.path.join(output_dir, fname)
            if not os.path.exists(path):
                logger.warning("Verify: file not found %s", path)
                missing = True
                continue
            try:
                onnx_model = load(path)
                try:
                    checker.check_model(onnx_model)
                    logger.info("%s passed ONNX checker", fname)
                except MemoryError:
                    logger.warning("Skipping checker for %s due to MemoryError", fname)

                # Detect external data usage and emit actionable guidance
                try:
                    external_used = has_external_data(onnx_model)
                    if external_used:
                        logger.warning(
                            "Model %s uses external_data tensors. Ensure associated tensor files are co-located with the model and retained during deployment.",
                            fname,
                        )
                        logger.info(
                            "If a single-file model is required, consider onnx.external_data_helper.convert_model_to_single_file and re-export or repack."
                        )
                        # If repack requested, convert to single-file and replace original path
                        if pack_single:
                            try:
                                # If a threshold is provided, estimate packed size and skip if too large
                                def _get_external_locations(model) -> List[str]:
                                    locs: List[str] = []
                                    try:
                                        # TensorProto.DataLocation.EXTERNAL == 1
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
                                    # de-duplicate
                                    return list({l for l in locs if l})

                                def _sum_external_bytes(
                                    base_dir: str, locs: List[str]
                                ) -> int:
                                    total = 0
                                    for loc in locs:
                                        try:
                                            fp = os.path.join(base_dir, loc)
                                            total += os.path.getsize(fp)
                                        except Exception:
                                            # ignore missing or inaccessible files
                                            pass
                                    return total

                                do_repack = True
                                if pack_single_threshold_mb is not None:
                                    try:
                                        locs = _get_external_locations(onnx_model)
                                        total_bytes = _sum_external_bytes(
                                            output_dir, locs
                                        )
                                        total_mb = total_bytes / (1024 * 1024)
                                        logger.info(
                                            "Estimated external tensor size for %s: %.1f MB (threshold=%.1f MB)",
                                            fname,
                                            total_mb,
                                            float(pack_single_threshold_mb),
                                        )
                                        if total_mb > float(pack_single_threshold_mb):
                                            do_repack = False
                                            logger.info(
                                                "Skipping single-file repack for %s due to size exceeding threshold",
                                                fname,
                                            )
                                    except Exception:
                                        # If estimation fails, proceed with repack attempt
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
                                    logger.info(
                                        "Repacked external_data model into single file: %s (%.2fs)",
                                        path,
                                        dt,
                                    )
                                    # Reload the repacked model for session/metadata below
                                    onnx_model = load(path)
                                    external_used = False
                            except Exception as e:
                                logger.warning(
                                    "Failed to repack model %s into single file: %s",
                                    fname,
                                    e,
                                )
                except Exception:
                    logger.debug("External data detection failed for %s", fname)

                sess = create_ort_session(path, provider=provider)
                logger.info(
                    "%s inputs=%s outputs=%s",
                    fname,
                    [i.name for i in sess.get_inputs()],
                    [o.name for o in sess.get_outputs()],
                )
                del onnx_model, sess
                gc.collect()
            except Exception as e:
                logger.error("Verification failed for %s: %s", fname, e)
                try:
                    corrupt = f"{path}.corrupt"
                    os.replace(path, corrupt)
                    logger.warning("Moved corrupt file to %s", corrupt)
                except Exception:
                    logger.exception("Failed to move corrupt file %s", path)
                return False
        return not missing

    # Export phase: either legacy v1 or v2 strategy
    export_succeeded = False
    try:
        if use_v1:
            # Legacy: use ORTModelFor* .from_pretrained(..., export=True)
            from optimum.onnxruntime import (
                ORTModelForFeatureExtraction,
                ORTModelForSeq2SeqLM,
                ORTModelForSequenceClassification,
            )

            logger.info("Using legacy ORTModel export path (v1)")
            if _model_for in ["s2s", "seq2seq-lm"]:
                # Pass use_cache through to ORTModel export where supported
                model = ORTModelForSeq2SeqLM.from_pretrained(
                    model_name, export=True, use_cache=use_cache
                )
            elif _model_for in ["sc", "sequence-classification"]:
                model = ORTModelForSequenceClassification.from_pretrained(
                    model_name, export=True
                )
            else:
                model = ORTModelForFeatureExtraction.from_pretrained(
                    model_name, export=True
                )

            # Save exported ONNX artifacts
            model.save_pretrained(_output_dir)
            logger.info("Legacy export saved to %s", _output_dir)
            logger.info(
                "Process log: export phase completed using legacy ORTModel path"
            )
            # Determine expected files
            if _model_for in ["s2s", "seq2seq-lm"]:
                expected = [_encoder_onnx_name, _decoder_onnx_name]
                if use_cache:
                    expected.append(_decoder_with_past_name)
            else:
                expected = [_onnx_name]

        else:
            # v2: primary path uses optimum.exporters.onnx.main_export with fallback
            from optimum.exporters.onnx import main_export

            logger.info(
                "Using v2 export path (main_export) with opset=%s", opset_version
            )
            # Determine export task. Allow explicit `task`, fallback to seq2seq for s2s models.
            export_task = task or (
                "seq2seq-lm" if _model_for in ["s2s", "seq2seq-lm"] else None
            )
            if export_task is None and _model_for in ["fe", "feature-extraction"]:
                export_task = "feature-extraction"
            # KV-cache export: for seq2seq models request the "-with-past" task used by exporters
            if use_cache and _model_for in ["s2s", "seq2seq-lm"] and not task:
                export_task = "text2text-generation-with-past"
            # If user requested T5 encoder-only export, override task to feature-extraction
            if use_t5_encoder:
                export_task = "feature-extraction"

            # Some libraries (e.g., sentence-transformers) need explicit library hint
            export_library = None
            try:
                if "sentence-transformers" in (model_name or ""):
                    export_library = "sentence_transformers"
            except Exception:
                export_library = None

            try:
                me_kwargs = {
                    "model_name_or_path": model_name,
                    "output": _output_dir,
                    "task": export_task,
                    "opset": opset_version,
                    "device": "cpu",
                    "use_external_data_format": True,
                }
                if framework:
                    me_kwargs["framework"] = framework
                if export_library:
                    me_kwargs["library"] = export_library
                main_export(**me_kwargs)
                # If no exception was raised, main_export completed
                logger.info("v2 main_export saved artifacts to %s", _output_dir)
                logger.info("Process log: export phase completed using v2 main_export")
                export_succeeded = True
            except Exception as e:
                logger.warning("Primary main_export failed: %s", e)
                # Retry with lower opset if protobuf serialization issues
                err = str(e).lower()
                if "failed to serialize proto" in err or "encodeerror" in err:
                    try:
                        logger.info("Retrying main_export with opset=11 as fallback")
                        main_export(
                            model_name_or_path=model_name,
                            output=_output_dir,
                            task=export_task,
                            opset=11,
                            device="cpu",
                            use_external_data_format=True,
                        )
                    except Exception as e2:
                        logger.error("Fallback export with opset 11 failed: %s", e2)
                        # Last resort: try ORTModel-based export (v1 style)
                        from optimum.onnxruntime import (
                            ORTModelForFeatureExtraction,
                            ORTModelForSeq2SeqLM,
                            ORTModelForSequenceClassification,
                        )

                        logger.info("Attempting ORTModel fallback export")
                        if _model_for in ["s2s", "seq2seq-lm"]:
                            ort_model = ORTModelForSeq2SeqLM.from_pretrained(
                                model_name, export=True
                            )
                        elif _model_for in ["sc", "sequence-classification"]:
                            ort_model = (
                                ORTModelForSequenceClassification.from_pretrained(
                                    model_name, export=True
                                )
                            )
                        else:
                            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                                model_name, export=True
                            )
                        ort_model.save_pretrained(_output_dir)
                        del ort_model
                        logger.info("Fallback ORTModel export saved to %s", _output_dir)
                        logger.info(
                            "Process log: export phase completed using ORTModel fallback"
                        )

                else:
                    # For other failures, attempt ORTModel fallback to avoid empty outputs
                    try:
                        from optimum.onnxruntime import (
                            ORTModelForFeatureExtraction,
                            ORTModelForSeq2SeqLM,
                            ORTModelForSequenceClassification,
                        )

                        logger.info(
                            "Attempting ORTModel fallback export after main_export failure"
                        )
                        if _model_for in ["s2s", "seq2seq-lm"]:
                            ort_model = ORTModelForSeq2SeqLM.from_pretrained(
                                model_name, export=True
                            )
                        elif _model_for in ["sc", "sequence-classification"]:
                            ort_model = (
                                ORTModelForSequenceClassification.from_pretrained(
                                    model_name, export=True
                                )
                            )
                        else:
                            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                                model_name, export=True
                            )
                        ort_model.save_pretrained(_output_dir)
                        del ort_model
                        logger.info("Fallback ORTModel export saved to %s", _output_dir)
                        logger.info(
                            "Process log: export phase completed using ORTModel fallback"
                        )
                        export_succeeded = True
                    except Exception as e3:
                        logger.error("ORTModel fallback export failed: %s", e3)

            # Determine expected outputs
            if _model_for in ["s2s", "seq2seq-lm"]:
                expected = [_encoder_onnx_name, _decoder_onnx_name]
                if use_cache:
                    expected.append(_decoder_with_past_name)
            else:
                expected = [_onnx_name]

        # Post-export verification
        ok = _verify_models(
            expected,
            _output_dir,
            pack_single=pack_single_file,
            pack_single_threshold_mb=pack_single_threshold_mb,
        )
        if not ok:
            logger.error("Post-export verification failed for %s", model_name)
            logger.info("Process log: verification step completed with failures")
        else:
            logger.info("Post-export verification succeeded")
            logger.info("Process log: verification step completed successfully")

        # Optimization phase
        if optimize:
            try:
                from optimum.onnxruntime import ORTOptimizer
                from optimum.onnxruntime.configuration import OptimizationConfig

                # Basic memory check
                try:
                    import psutil

                    available_mb = psutil.virtual_memory().available / (1024 * 1024)
                except Exception:
                    available_mb = None

                if available_mb is not None and available_mb < 500:
                    logger.warning(
                        "Insufficient memory (%.1f MB) for optimization. Skipping.",
                        available_mb,
                    )
                    logger.info(
                        "Process log: optimization skipped due to insufficient memory"
                    )
                else:
                    config = OptimizationConfig(optimization_level=optimization_level)
                    # Load model for optimization
                    from optimum.onnxruntime import (
                        ORTModelForFeatureExtraction,
                        ORTModelForSeq2SeqLM,
                        ORTModelForSequenceClassification,
                    )

                    if _model_for in ["s2s", "seq2seq-lm"]:
                        # Preserve use_cache when loading exported model for optimization
                        opt_model = ORTModelForSeq2SeqLM.from_pretrained(
                            _output_dir, use_cache=use_cache
                        )
                    elif _model_for in ["sc", "sequence-classification"]:
                        opt_model = ORTModelForSequenceClassification.from_pretrained(
                            _output_dir
                        )
                    else:
                        opt_model = ORTModelForFeatureExtraction.from_pretrained(
                            _output_dir
                        )

                    optimizer = ORTOptimizer.from_pretrained(opt_model)
                    optimizer.optimize(
                        save_dir=Path(_output_dir), optimization_config=config
                    )
                    logger.info("Optimization completed for %s", model_name)
                    logger.info("Process log: optimization step completed successfully")
                    del opt_model, optimizer

                    # Post-optimization re-verify: load optimized artifacts and log inputs/outputs
                    try:
                        logger.info(
                            "Post-optimization verification: checking optimized artifacts..."
                        )
                        for fname in expected:
                            opt_path = os.path.join(_output_dir, fname)
                            if os.path.exists(opt_path):
                                try:
                                    opt_sess = create_ort_session(
                                        opt_path, provider=get_preferred_provider()
                                    )
                                    opt_inputs = [i.name for i in opt_sess.get_inputs()]
                                    opt_outputs = [
                                        o.name for o in opt_sess.get_outputs()
                                    ]
                                    logger.info(
                                        "Optimized %s: inputs=%s outputs=%s",
                                        fname,
                                        opt_inputs,
                                        opt_outputs,
                                    )
                                    del opt_sess
                                    gc.collect()
                                except Exception as e:
                                    logger.warning(
                                        "Could not verify optimized %s: %s", fname, e
                                    )
                    except Exception as e:
                        logger.warning("Post-optimization re-verify failed: %s", e)
            except Exception as e:
                logger.warning("Optimization failed: %s", e)
                logger.info("Process log: optimization step failed")

    except Exception as e:
        logger.exception("Export failed: %s", e)
        # Clean up lock file on error to avoid blocking retries
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
                logger.info("Removed lock file after export failure: %s", lock_path)
        except Exception:
            logger.debug("Could not remove lock file on error cleanup")
        raise

    gc.collect()
    # Clear CUDA cache if torch is available and CUDA is initialized to free GPU memory
    try:
        import torch

        try:
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache via torch.cuda.empty_cache()")
        except Exception:
            # Don't fail the exporter if cuda management fails
            logger.debug("torch.cuda.empty_cache() call failed or CUDA not available")
    except Exception:
        # torch not installed; nothing to do
        pass
    # detach file handler if we attached one earlier to avoid duplicate logs
    try:
        # restore stdout/stderr if we replaced them
        try:
            import sys

            if _stdout_orig is not None:
                try:
                    sys.stdout = _stdout_orig
                except Exception:
                    pass
            if _stderr_orig is not None:
                try:
                    sys.stderr = _stderr_orig
                except Exception:
                    pass
        except Exception:
            pass

        if file_handler is not None:
            try:
                logger.removeHandler(file_handler)
            except Exception:
                pass
            try:
                file_handler.close()
            except Exception:
                pass
        if _logfile_fd is not None:
            try:
                _logfile_fd.close()
            except Exception:
                pass
    except Exception:
        pass

    return _output_dir
