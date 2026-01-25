# =============================================================================
# File: export_model_consolidated.py
# Date: 2026-01-10
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

"""
This file was regenerated from the older consolidated exporter and refactored to
delegate heavy operations to helper modules. It preserves the original
behaviour while implementing the previously refactored features:
- protobuf configuration via `export_helpers.configure_protobuf`
- per-export logging setup/teardown via `export_logging`
- lockfile semantics, deterministic finetune handling, device/opset propagation
- v2 `main_export` with opset retry and ORTModel fallback, plus clone-wrapper
- quick structural verification, numeric validator invocation and optimization
"""

import gc
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from .export_helpers import (
    cleanup_validator_logging_handlers,
    configure_protobuf,
    is_pid_running,
)
from .export_logging import setup_export_logging, teardown_export_logging
from .export_validator import invoke_validator
from .onnx_helpers import get_default_opset, get_logger
from .onnx_verify import verify_models

logger = get_logger(__name__)


def _build_expected_list(
    model_for: str, use_cache: bool, task: str | None = None
) -> List[str]:
    mf = (model_for or "").lower()
    t = (task or "").lower()
    # For seq2seq models we normally expect encoder+decoder files. If KV-cache
    # is requested via `use_cache` or the `task` indicates a "with-past" export,
    # include the `decoder_with_past_model.onnx` artifact in the expected list.
    if mf in ["s2s", "seq2seq-lm"]:
        names = ["encoder_model.onnx", "decoder_model.onnx"]
        if use_cache or (t == "text2text-generation-with-past"):
            names.append("decoder_with_past_model.onnx")
        return names
    names = ["model.onnx"]
    if use_cache or (t == "text-generation-with-past"):
        names.append("model_with_past.onnx")
    return names


def _auto_enable_use_cache(
    base_dir: str,
    model_name: str,
    model_folder: str,
    model_for: str,
    current_flag: bool,
) -> bool:
    """Decide whether to enable `use_cache` automatically.

    Strategy:
    - If caller already set `current_flag`, respect it.
    - Try to read `app/config/onnx_config.json` (project-level) and look up
      the model by `model_folder` name. If the config indicates LLM tasks,
      `use_seq2seqlm` or a decoder model entry, enable cache.
    - Otherwise, attempt to inspect the HF model config via
      `transformers.AutoConfig` and use `cfg.use_cache` / `cfg.is_encoder_decoder`
      to decide.
    - Fallback: keep `current_flag` (False by default).
    """
    if current_flag:
        return True

    # Skip project-level config file lookup; rely on transformers AutoConfig
    # and the caller-provided flag. This avoids reading repository files here.

    # As a secondary check, consult HF model config if available
    try:
        from transformers import AutoConfig

        try:
            cfg = AutoConfig.from_pretrained(str(model_name))
            if getattr(cfg, "use_cache", False):
                logger.info(
                    "Transformer config indicates use_cache for %s; enabling use_cache",
                    model_name,
                )
                return True
            # Decoder-only models that support caching
            if not getattr(cfg, "is_encoder_decoder", False) and getattr(
                cfg, "use_cache", False
            ):
                logger.info(
                    "Transformer config indicates decoder caching for %s; enabling use_cache",
                    model_name,
                )
                return True
        except Exception:
            # If model not available or network is disabled, ignore
            pass
    except Exception:
        pass

    return False


def _handle_finetune(
    model_name: str, finetune_flag: bool, model_folder: str, output_dir: str, logger
) -> str:
    """If requested, run deterministic T5 finetune into an output subfolder.

    Returns possibly-updated `model_name` (local path to finetuned folder).
    """
    if not finetune_flag or "t5" not in (model_name or "").lower():
        return model_name

    finetune_dir = os.path.join(output_dir, f"{model_folder}_finetuned")
    Path(finetune_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Finetuning requested and model looks like T5; fine-tuning to %s", finetune_dir
    )
    try:
        from .fine_tune_t5 import fine_tune_t5_embeddings

        fine_tune_t5_embeddings(model_name, finetune_dir)
        logger.info("Using fine-tuned model at %s for export", finetune_dir)
        return finetune_dir
    except Exception as e:
        logger.exception("Fine-tuning failed: %s", e)
        raise


def _create_export_lock(output_dir: str, model_name: str, logger) -> tuple[str, bool]:
    """Create a lock file for `output_dir`. Returns (lock_path, created_lock)."""
    lock_path = os.path.join(output_dir, ".export.lock")
    created_lock = False
    try:
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
            stale_threshold = 24 * 60 * 60
            if pid is not None and is_pid_running(pid):
                logger.info(
                    "An export for this model is already running (pid=%s). Exiting.",
                    pid,
                )
                raise SystemExit(0)
            else:
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

        try:
            with open(lock_path, "x", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()}\n{time.time()}\n{model_name}\n")
            created_lock = True
            logger.info("Acquired export lock: %s", lock_path)
        except FileExistsError:
            logger.info("Lock file created by concurrent process; exiting")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        logger.warning(
            "Could not create or check export lock; proceeding without exclusive lock"
        )

    return lock_path, created_lock


@contextmanager
def _with_export_lock(output_dir: str, model_name: str, logger):
    """Context manager that creates an export lock and removes it on exit

    Yields `(lock_path, created_lock)`.
    """
    lock_path = None
    created_lock = False
    try:
        lock_path, created_lock = _create_export_lock(output_dir, model_name, logger)
        yield lock_path, created_lock
    finally:
        try:
            if created_lock and lock_path and os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                    logger.info("Removed lock file: %s", lock_path)
                except Exception:
                    logger.debug("Could not remove lock file: %s", lock_path)
        except Exception:
            # Ensure contextmanager cleanup never raises
            pass


def _run_export_with_fallback(
    export_source: str,
    output_dir: str,
    model_for: str,
    opset_version: int | None,
    device: str,
    task: str | None,
    framework: str | None,
    library: str | None,
    logger,
    trust_remote_code: bool,
    use_v1: bool,
    use_external_data_format: bool = True,
    no_post_process: bool = False,
    merge: bool = False,
) -> tuple[bool, bool]:
    """Try v2 export, fallback to ORTModel/v1, and finally clone-based retry.

    Returns a tuple (export_succeeded, used_trust_remote).
    """
    export_succeeded = False
    used_trust_remote = False

    try:
        logger.info(
            "_run_export_with_fallback start: source=%s output_dir=%s model_for=%s opset=%s device=%s task=%s framework=%s library=%s trust_remote_code=%s use_v1=%s",
            export_source,
            output_dir,
            model_for,
            opset_version,
            device,
            task,
            framework,
            library,
            trust_remote_code,
            use_v1,
        )
    except Exception:
        try:
            logger.info("_run_export_with_fallback start")
        except Exception:
            pass

    # Legacy ORTModel export (when explicitly requested).
    # Use the `export_v1` helper first to centralize ORTModel export behavior
    # and avoid duplicating calls; if that helper is not available, fall back
    # to an inline optimum-based attempt.
    attempted_v1 = False
    if use_v1:
        try:
            from .export_v1 import export_v1_ortmodel

            attempted_v1 = True
            logger.info("Attempting legacy v1 ORTModel export via export_v1_ortmodel")
            try:
                v1_ok, v1_used_trust = export_v1_ortmodel(
                    export_source,
                    output_dir,
                    model_for,
                    trust_remote_code,
                    logger,
                    opset_version=opset_version,
                    device=device,
                    task=task,
                    framework=framework,
                    library=library,
                    use_external_data_format=use_external_data_format,
                    merge=merge,
                )
                if v1_ok:
                    export_succeeded = True
                    logger.info("Legacy v1 ORTModel export succeeded")
                if v1_used_trust:
                    used_trust_remote = True
            except Exception as ort_err:
                logger.warning(
                    "Legacy ORTModel export (export_v1 helper) failed: %s", ort_err
                )
        except Exception as e:
            logger.warning(
                "Required export_v1 helper `export_v1_ortmodel` not available; will attempt clone-based fallback if allowed: %s",
                e,
            )

            # Clone-based fallback moved into `export_v1_ortmodel` helper.
            # The helper will perform a clone-based retry when `trust_remote_code`
            # is True so that all legacy v1 behaviour is centralized there.

    # Attempt v2 main_export via helper (now returns (success, used_trust_remote)).
    # If caller explicitly requested legacy v1 (`use_v1=True`) we must NOT
    # invoke the v2 exporter at all (even as a fallback). Honor that contract
    # by skipping the v2 path when `use_v1` is true.
    if not use_v1:
        try:
            from .export_v2 import export_v2_main_export

            logger.info("Attempting v2 main_export via export_v2_main_export")
            v2_ok, v2_used_trust = export_v2_main_export(
                export_source,
                output_dir,
                model_for,
                opset_version,
                device,
                task,
                framework,
                library,
                trust_remote_code,
                logger,
                use_external_data_format=use_external_data_format,
                no_post_process=no_post_process,
                merge=merge,
            )
            if v2_ok:
                export_succeeded = True
                logger.info("v2 main_export succeeded")
            if v2_used_trust:
                used_trust_remote = True
            if not v2_ok:
                logger.info(
                    "v2 main_export did not succeed; will attempt ORTModel fallback"
                )
        except Exception as import_err:
            logger.warning(
                "v2 exporter wrapper failed to import or execute: %s", import_err
            )
    else:
        logger.info("use_v1=True: skipping v2 main_export as requested by caller")
    # All legacy v1 fallback logic (ORTModel and clone-based retry) is
    # handled inside the `if use_v1:` branch above. We do not perform v1
    # fallback or clone-based retries for v2 exports.
    # After successful export, run ONNX sanitizer to deduplicate tied initializers
    try:
        if export_succeeded:
            try:
                from .export_helpers import sanitize_onnx_initializers
            except Exception:
                sanitize_onnx_initializers = None
            if sanitize_onnx_initializers is not None:
                try:
                    modified = sanitize_onnx_initializers(output_dir, logger)
                    if modified:
                        logger.info("Sanitized %d ONNX files after export", modified)
                except Exception:
                    logger.debug("sanitize_onnx_initializers failed", exc_info=True)
    except Exception:
        logger.debug("Post-export sanitizer handling failed", exc_info=True)

    try:
        logger.info(
            "_run_export_with_fallback result: export_succeeded=%s used_trust_remote=%s",
            export_succeeded,
            used_trust_remote,
        )
    except Exception:
        pass

    return export_succeeded, used_trust_remote


def export_and_optimize_onnx_unified(
    model_name: str,
    model_for: str = "fe",
    optimize: bool = False,
    merge: bool = False,
    optimization_level: int = 99,
    portable: bool = False,
    model_folder: str | None = None,
    onnx_path: str | None = None,
    task: str | None = None,
    finetune: bool = False,
    force: bool = False,
    opset_version: int | None = None,
    use_v1: bool = False,
    pack_single_file: bool = False,
    framework: str | None = None,
    pack_single_threshold_mb: int | None = 1536,
    require_validator: bool = False,
    trust_remote_code: bool = False,
    normalize_embeddings: bool = False,
    skip_validator: bool = False,
    device: str = "cpu",
    library: str | None = None,
    use_external_data_format: bool = True,
    no_local_prep: bool = False,
    **kwargs,
) -> str:
    """Orchestrate export, verification, numeric validation and optimization.

    This implementation mirrors the previously refactored orchestrator: it is
    lightweight on import and defers heavy operations to helper modules.
    """

    # Configure protobuf limits early
    configure_protobuf()

    # Normalize Hugging Face token handling: accept token via kwargs or environment
    # and attempt to login for the session. If login fails (invalid token),
    # remove any env vars we set and continue anonymously.
    try:
        token = kwargs.pop("hf_token", None) or kwargs.pop(
            "huggingface_hub_token", None
        )
    except Exception:
        token = None
    if not token:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

    prev_hf = os.environ.get("HF_TOKEN")
    prev_hub = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    set_hf = False
    set_hub = False
    if token:
        try:
            if prev_hf is None:
                os.environ["HF_TOKEN"] = token
                set_hf = True
            if prev_hub is None:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                set_hub = True
        except Exception:
            logger.debug("Could not set HF token environment variables")

    # If a token is available, try to login to the Hugging Face hub for the session.
    # If login fails, remove any env vars we set and continue without a token.
    if token:
        login_ok = False
        try:
            try:
                from huggingface_hub import login as _hf_login

                _hf_login(token=token, add_to_git_credential=False)
                login_ok = True
            except Exception:
                # older versions expose login differently; try high-level import
                import huggingface_hub as _hfh

                try:
                    _hfh.login(token=token, add_to_git_credential=False)
                    login_ok = True
                except Exception as e:
                    logger.debug("huggingface_hub.login failed: %s", e)

            if login_ok:
                logger.info(
                    "Logged into Hugging Face hub for this session via provided token"
                )
            else:
                raise RuntimeError("huggingface_hub.login did not succeed")
        except Exception as e:
            logger.warning(
                "Hugging Face login failed (%s); will continue without token", e
            )
            # remove env vars we set so downstream libs use anonymous access
            try:
                if set_hf and "HF_TOKEN" in os.environ:
                    del os.environ["HF_TOKEN"]
                if set_hub and "HUGGINGFACE_HUB_TOKEN" in os.environ:
                    del os.environ["HUGGINGFACE_HUB_TOKEN"]
            except Exception:
                logger.debug("Failed to clear HF env vars after failed login")

    opset_version = opset_version or get_default_opset()

    if not model_name or not str(model_name).strip():
        raise ValueError("model_name cannot be empty")

    _model_for = (model_for or "").lower()
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

    # an explicit `--task` (e.g., `feature-extraction`) when exporting T5 encoders.

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    onnx_path = onnx_path or "onnx"
    onnx_path = os.path.normpath(onnx_path)
    if ".." in onnx_path:
        raise ValueError("Path traversal detected in onnx_path")

    # Allow callers to pass `use_cache` in kwargs for backward compatibility;
    # primary source of truth will be auto-detection via `_auto_enable_use_cache`.
    # Do not accept `use_cache` as a positional parameter anymore; compute it
    # internally via auto-detection and task hints.
    use_cache = False

    if not model_folder:
        model_folder = (
            model_name.split("/")[-1] if "/" in str(model_name) else str(model_name)
        )
    model_folder = os.path.basename(model_folder)
    _output_dir = os.path.join(BASE_DIR, onnx_path, "models", _model_for, model_folder)
    Path(_output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect whether to enable KV-cache exports when user didn't explicitly set it
    try:
        auto_flag = _auto_enable_use_cache(
            BASE_DIR, model_name, model_folder, _model_for, use_cache
        )
        if auto_flag and not use_cache:
            logger.info("Automatically enabling use_cache for model %s", model_name)
        use_cache = bool(auto_flag)
    except Exception:
        logger.debug(
            "Auto-detection for use_cache failed; using provided value: %s", use_cache
        )

    # Respect an explicit task that requests KV-cache exports (e.g. text2text-generation-with-past)
    try:
        if task and "with-past" in str(task).lower():
            if not use_cache:
                logger.info(
                    "Enabling use_cache because explicit task requests KV-cache: %s",
                    task,
                )
            use_cache = True
    except Exception:
        pass

    # Respect an explicit seq2seq task which requests no KV-cache (encoder-decoder
    # export). If the caller asked for `seq2seq-lm` / `seq2seq` and this is an
    # s2s export, ensure `use_cache` is disabled even if auto-detection enabled it.
    try:
        if (
            task
            and "seq2seq" in str(task).lower()
            and _model_for in ["s2s", "seq2seq-lm"]
        ):
            if use_cache:
                logger.info(
                    "Disabling use_cache because explicit task requests seq2seq (no KV-cache): %s",
                    task,
                )
            use_cache = False
    except Exception:
        pass

    # Finetune early and deterministically into an output subfolder if requested
    model_name = _handle_finetune(
        model_name, finetune, model_folder, _output_dir, logger
    )

    # Create export lock to avoid concurrent exports (kept for backward compatibility)
    with _with_export_lock(_output_dir, model_name, logger) as (
        lock_path,
        created_lock,
    ):
        # Pre-export cleanup
        try:
            gc.collect()
            try:
                import torch

                if (
                    getattr(torch, "cuda", None) is not None
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache before export to free GPU memory")
            except Exception:
                logger.debug("No torch/CUDA available for pre-export cache clear")

            try:
                import psutil

                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                logger.info("Available memory before export: %.1f MB", available_mb)
            except Exception:
                logger.debug("psutil not available for memory check before export")
        except Exception:
            logger.debug("Pre-export cleanup failed", exc_info=True)

    # Setup per-run logging
    file_handler = None
    logfile_fd = None
    old_stdout = None
    old_stderr = None
    logfile_path = None
    safe_model = model_folder.replace("/", "_").replace("\\", "_")
    rev_tag = "local"
    try:
        from huggingface_hub import HfApi

        if "/" in str(model_name):
            try:
                info = HfApi().repo_info(str(model_name))
                rev_tag = (
                    getattr(info, "sha", None)
                    or getattr(info, "revision", None)
                    or "local"
                )
            except Exception:
                rev_tag = "local"
    except Exception:
        rev_tag = "local"

    try:
        file_handler, logfile_fd, old_stdout, old_stderr, logfile_path = (
            setup_export_logging(BASE_DIR, safe_model, rev_tag, logger)
        )
    except Exception:
        logger.warning(
            "Failed to initialize per-run logging; continuing without file capture"
        )

    expected = _build_expected_list(_model_for, use_cache, task)

    try:
        # Skip export if outputs exist and no force requested
        all_exist = all(
            os.path.exists(os.path.join(_output_dir, fname)) for fname in expected
        )
        if all_exist and not force:
            logger.info(
                "All expected ONNX files already exist in %s — skipping export (use --force to re-export)",
                _output_dir,
            )
            logger.info(
                "Process log: export step skipped because outputs already present"
            )
            return _output_dir

        # If force requested, remove existing output dir to ensure a clean export
        if force and os.path.exists(_output_dir):
            try:
                import pathlib

                shutil.rmtree(_output_dir)
                Path(_output_dir).mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Removed existing output directory because --force was given: %s",
                    _output_dir,
                )
                logger.info("Process log: cleaned output directory as part of --force")
            except Exception as e:
                logger.warning(
                    "Failed to remove existing output dir with --force: %s", e
                )
                # Best-effort: clear files inside to avoid stale artifacts
                try:
                    for p in pathlib.Path(_output_dir).glob("**/*"):
                        try:
                            if p.is_file():
                                p.unlink()
                            elif p.is_dir():
                                try:
                                    shutil.rmtree(p)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    logger.debug("Best-effort cleanup of output dir failed")

        # Export phase: prepare local copy for certain HF models (LLMs) to
        # canonicalize tied weights and avoid duplicate initializer issues,
        # then delegate to helper that handles v2/v1/clone fallbacks.
        prep_tmp_p = None
        try:
            export_source = model_name
            # If source looks like a HF hub id (not a local path) and this is an
            # LLM export, prepare a local tied-weights copy to stabilize ONNX
            # initializer naming. Place it under onnx/tmp_export/<model_folder>-local.
            try:
                if (
                    _model_for == "llm"
                    and not os.path.exists(str(model_name))
                    and not bool(no_local_prep)
                ):
                    from .export_helpers import prepare_local_model_dir

                    # Prepare the transient local copy inside the model folder
                    # so it's colocated with the eventual ONNX output. Use a
                    # `temp_local` subfolder to avoid colliding with the final
                    # output artifacts. This folder will be removed after export
                    # (success or failure).
                    tmp_p = os.path.join(
                        BASE_DIR,
                        onnx_path,
                        "models",
                        _model_for,
                        model_folder,
                        "temp_local",
                    )
                    # Ensure parent exists
                    try:
                        os.makedirs(os.path.dirname(tmp_p), exist_ok=True)
                    except Exception:
                        pass
                    # Remove stale temp_local before creating
                    try:
                        if os.path.exists(tmp_p):
                            shutil.rmtree(tmp_p)
                    except Exception:
                        pass
                    os.makedirs(tmp_p, exist_ok=True)
                    prep_ok = prepare_local_model_dir(
                        model_name, tmp_p, trust_remote_code, logger
                    )
                    if prep_ok:
                        export_source = tmp_p
                        prep_tmp_p = tmp_p
                        logger.info("Using prepared local model for export: %s", tmp_p)
                elif (
                    _model_for == "llm"
                    and not os.path.exists(str(model_name))
                    and bool(no_local_prep)
                ):
                    logger.info(
                        "Skipping local prep for LLM as requested by --no-local-prep"
                    )
            except Exception:
                logger.debug(
                    "Local model prep skipped or failed; continuing with original source",
                    exc_info=True,
                )
            # Always attempt export after any preparation step. Ensure
            # `export_succeeded` is assigned regardless of whether local
            # preparation succeeded or raised an exception.
            export_succeeded, used_trust_remote = _run_export_with_fallback(
                export_source,
                _output_dir,
                _model_for,
                opset_version,
                device,
                task,
                framework,
                library,
                logger,
                trust_remote_code,
                use_v1,
                use_external_data_format=use_external_data_format,
                no_post_process=bool(kwargs.get("no_post_process", False)),
                merge=bool(merge),
            )
            if not export_succeeded:
                raise RuntimeError(
                    "All export attempts failed for model %s" % model_name
                )
        except Exception as e:
            # On export failure, attempt to remove the output model folder to
            # free disk space for retries or subsequent operations.
            try:
                logger.warning(
                    "Export failed; attempting to remove output dir to free space: %s",
                    _output_dir,
                )
                if os.path.exists(_output_dir):
                    try:
                        shutil.rmtree(_output_dir)
                        logger.info(
                            "Removed output directory after failed export: %s",
                            _output_dir,
                        )
                    except Exception as rm_e:
                        logger.warning(
                            "Failed to remove output directory %s: %s",
                            _output_dir,
                            rm_e,
                        )
            except Exception:
                logger.debug(
                    "Output-dir cleanup raised during export failure handling",
                    exc_info=True,
                )
            logger.exception("Export failed: %s", e)
            raise
        finally:
            # Clean up any prepared transient local copy used for export
            try:
                if prep_tmp_p and os.path.exists(prep_tmp_p):
                    try:
                        shutil.rmtree(prep_tmp_p)
                        logger.info(
                            "Removed temporary local model folder: %s", prep_tmp_p
                        )
                    except Exception as rm_e:
                        logger.debug(
                            "Failed to remove temporary local model folder %s: %s",
                            prep_tmp_p,
                            rm_e,
                        )
            except Exception:
                logger.debug("Cleanup of prep_tmp_p failed", exc_info=True)

        # Quick structural verification
        try:
            quick_ok = bool(
                verify_models(
                    expected,
                    _output_dir,
                    pack_single=pack_single_file,
                    pack_single_threshold_mb=pack_single_threshold_mb,
                )
            )
        except Exception as v_err:
            logger.warning("Quick verification raised: %s", v_err)
            quick_ok = False

        # Seq2seq (encoder+decoder) merging is disallowed by policy.
        # Merge is only applicable to decoder-only causal LLMs that support
        # `text-generation-with-past` (KV-cache) and must run before
        # optimization. No encoder+decoder or post-optimization merging is
        # performed here.

        # Invoke numeric validator unless explicitly skipped.
        # Note: the centralized numeric validator expects a single `model.onnx` file.
        # For multi-file seq2seq exports (encoder/decoder[/decoder_with_past]) skip
        # numeric validation unless the user asked for `--pack_single_file`.
        validator_rc = 0
        validator_quick_ok = quick_ok
        if not skip_validator:
            # If this is a seq2seq multi-file export and not packed into a single file,
            # don't run the numeric validator which requires `model.onnx`.
            if _model_for in ["s2s", "seq2seq-lm"] and not pack_single_file:
                logger.info(
                    "Skipping numeric validator for multi-file seq2seq export; use --pack_single_file to create model.onnx and enable numeric validation"
                )
            # If user requested `--pack_single_file` but the export produced the
            # usual multi-file seq2seq artifacts (encoder/decoder...), there is
            # nothing for `pack_single_file` to combine into `model.onnx` — the
            # repack operation only handles external_data, it does not merge
            # multiple ONNX artifacts. Avoid invoking the numeric validator in
            # this situation to prevent misleading `.validation_failed` markers.
            elif (
                pack_single_file
                and _model_for in ["s2s", "seq2seq-lm"]
                and "model.onnx" not in expected
            ):
                logger.info(
                    "--pack_single_file was requested but export produced multi-file seq2seq artifacts; skipping numeric validator since model.onnx is not present"
                )
            else:
                try:
                    try:
                        cleanup_validator_logging_handlers()
                    except Exception:
                        pass

                    # Pre-validator tokenizer fallback: some tokenizers lack a pad_token
                    # which causes the numeric validator to fail. Attempt to load the
                    # tokenizer saved in the output dir and set a pad_token fallback
                    # before invoking the validator. Prefer `eos_token` when available
                    # to avoid changing token indices; otherwise add a new '[PAD]' token.
                    try:
                        try:
                            from transformers import AutoTokenizer
                        except Exception:
                            AutoTokenizer = None
                        if AutoTokenizer is not None:
                            try:
                                tok = AutoTokenizer.from_pretrained(
                                    _output_dir,
                                    use_fast=False,
                                    trust_remote_code=trust_remote_code,
                                )
                                pad_ok = getattr(tok, "pad_token", None)
                                if pad_ok is None:
                                    eos = getattr(tok, "eos_token", None)
                                    if eos:
                                        try:
                                            tok.pad_token = eos
                                            tok.save_pretrained(_output_dir)
                                            logger.info(
                                                "Tokenizer had no pad_token; set pad_token=eos_token (%s) and saved tokenizer",
                                                eos,
                                            )
                                        except Exception:
                                            logger.debug(
                                                "Failed to set/save pad_token from eos_token for tokenizer in %s",
                                                _output_dir,
                                                exc_info=True,
                                            )
                                    else:
                                        try:
                                            tok.add_special_tokens(
                                                {"pad_token": "[PAD]"}
                                            )
                                            tok.save_pretrained(_output_dir)
                                            logger.info(
                                                "Tokenizer had no pad_token or eos_token; added '[PAD]' and saved tokenizer"
                                            )
                                        except Exception:
                                            logger.debug(
                                                "Failed to add/save pad_token '[PAD]' for tokenizer in %s",
                                                _output_dir,
                                                exc_info=True,
                                            )
                            except Exception:
                                # If tokenizer can't be loaded from output dir, ignore
                                logger.debug(
                                    "Could not load tokenizer from %s for pad_token fallback",
                                    _output_dir,
                                    exc_info=True,
                                )
                    except Exception:
                        logger.debug(
                            "Tokenizer pad_token fallback check failed", exc_info=True
                        )

                    logger.info(
                        "Validator flags: request_trust=%s used_trust_remote=%s",
                        trust_remote_code,
                        ("used_trust_remote" in locals() and used_trust_remote),
                    )

                    validator_rc, validator_quick_ok = invoke_validator(
                        output_dir=_output_dir,
                        expected=expected,
                        model_name=model_name,
                        pack_single_file=pack_single_file,
                        pack_single_threshold_mb=pack_single_threshold_mb,
                        trust_remote_code=(
                            trust_remote_code
                            or ("used_trust_remote" in locals() and used_trust_remote)
                        ),
                        normalize_embeddings=normalize_embeddings,
                        logger=logger,
                    )
                    if validator_rc != 0:
                        logger.warning(
                            "Numeric validator returned non-zero code: %s", validator_rc
                        )
                        try:
                            marker = os.path.join(_output_dir, ".validation_failed")
                            with open(marker, "w", encoding="utf-8") as mf:
                                mf.write(
                                    f"validation_failed: rc={validator_rc} ts={int(time.time())} model={model_name}\n"
                                )
                            logger.info("Wrote validation failure marker: %s", marker)
                        except Exception:
                            logger.debug("Could not write .validation_failed marker")
                    else:
                        # Validator passed — remove any stale marker file if present
                        try:
                            marker = os.path.join(_output_dir, ".validation_failed")
                            if os.path.exists(marker):
                                os.remove(marker)
                                logger.info(
                                    "Removed stale validation failure marker: %s",
                                    marker,
                                )
                        except Exception:
                            logger.debug("Could not remove .validation_failed marker")
                        # Remove any diagnostic dumps produced earlier since validation passed
                        try:
                            dumps_dir = os.path.join(_output_dir, "validation_dumps")
                            if os.path.exists(dumps_dir):
                                shutil.rmtree(dumps_dir)
                                logger.info(
                                    "Removed validation_dumps directory: %s", dumps_dir
                                )
                        except Exception:
                            logger.debug("Could not remove validation_dumps directory")
                        if require_validator:
                            raise SystemExit(validator_rc)
                except Exception as e:
                    logger.exception("Validator invocation failed: %s", e)
                    if require_validator:
                        raise

        # Optimization step (optional)
        if optimize:
            try:
                # Optimization level selection is handled inside the
                # `run_optimization` helper. Pass the requested
                # `optimization_level` through and let the optimizer
                # decide conservative defaults for decoder/LLM artifacts.
                opt_level_for_run = optimization_level

                # Disk-space guard: avoid running optimizer when free space is low
                try:
                    try:
                        usage = shutil.disk_usage(_output_dir)
                    except Exception:
                        # Fallback to current working dir if output_dir not mounted yet
                        usage = shutil.disk_usage(os.getcwd())
                    free_bytes = int(getattr(usage, "free", 0))
                except Exception:
                    free_bytes = 2 << 30  # assume plenty if check fails

                MIN_FREE_BYTES_FOR_OPT = 1 << 30  # 1 GiB
                if free_bytes < MIN_FREE_BYTES_FOR_OPT:
                    logger.warning(
                        "Insufficient disk space (%.1f MB) to safely run optimizer; skipping optimization",
                        free_bytes / (1024.0 * 1024.0),
                    )
                    logger.info(
                        "Process log: optimization skipped due to low disk space"
                    )
                    rc_post = 0
                else:
                    try:
                        from .export_optimizer import optimize_if_encoder
                    except Exception:
                        optimize_if_encoder = None

                    if optimize_if_encoder is None:
                        logger.warning(
                            "optimize_if_encoder helper not available; skipping optimization"
                        )
                        rc_post = 0
                    else:
                        rc_post = optimize_if_encoder(
                            _output_dir,
                            _model_for,
                            logger,
                            optimization_level,
                            portable=portable,
                        )
                    if rc_post != 0:
                        logger.warning(
                            "Post-optimization validator returned %s", rc_post
                        )
                    # If optimization succeeded, detect whether the optimizer
                    # produced any optimized artifacts and, if so, run the
                    # numeric validator again to verify the optimized model.
                    if rc_post == 0:
                        try:
                            optimized_found = False
                            # First, check for an explicit marker written by the optimizer
                            marker_path = os.path.join(
                                _output_dir, ".optimizations_applied"
                            )
                            if os.path.exists(marker_path):
                                optimized_found = True
                            # Next, look for ort_config.json which the optimizer writes
                            if not optimized_found:
                                for _root, _dirs, files in os.walk(_output_dir):
                                    for fname in files:
                                        if fname == "ort_config.json":
                                            optimized_found = True
                                            break
                                        if (
                                            fname.endswith("_optimized.onnx")
                                            or fname.endswith("optimized.onnx")
                                            or fname == "model_optimized.onnx"
                                        ):
                                            optimized_found = True
                                            break
                                    if optimized_found:
                                        break
                            if optimized_found:
                                logger.info(
                                    "Optimized ONNX artifact detected; running quick structural verification and numeric validator post-optimization"
                                )
                                # Run quick structural verification (same as pre-export quick check)
                                try:
                                    post_quick_ok = bool(
                                        verify_models(
                                            expected,
                                            _output_dir,
                                            pack_single=pack_single_file,
                                            pack_single_threshold_mb=pack_single_threshold_mb,
                                        )
                                    )
                                    if post_quick_ok:
                                        logger.info(
                                            "Post-optimization quick verification passed for %s",
                                            _output_dir,
                                        )
                                    else:
                                        logger.warning(
                                            "Post-optimization quick verification failed for %s",
                                            _output_dir,
                                        )
                                except Exception as v_err:
                                    post_quick_ok = False
                                    logger.warning(
                                        "Post-optimization quick verification raised: %s",
                                        v_err,
                                    )
                                if post_quick_ok and not skip_validator:
                                    try:
                                        post_rc, post_quick_ok = invoke_validator(
                                            output_dir=_output_dir,
                                            expected=expected,
                                            model_name=model_name,
                                            pack_single_file=pack_single_file,
                                            pack_single_threshold_mb=pack_single_threshold_mb,
                                            trust_remote_code=(
                                                trust_remote_code
                                                or (
                                                    "used_trust_remote" in locals()
                                                    and used_trust_remote
                                                )
                                            ),
                                            normalize_embeddings=normalize_embeddings,
                                            logger=logger,
                                        )
                                        if post_rc != 0:
                                            logger.warning(
                                                "Post-optimization numeric validator returned non-zero code: %s",
                                                post_rc,
                                            )
                                            try:
                                                marker = os.path.join(
                                                    _output_dir,
                                                    ".validation_failed",
                                                )
                                                with open(
                                                    marker, "w", encoding="utf-8"
                                                ) as mf:
                                                    mf.write(
                                                        f"post_optimization_validation_failed: rc={post_rc} ts={int(time.time())} model={model_name}\n"
                                                    )
                                                logger.info(
                                                    "Wrote post-optimization validation failure marker: %s",
                                                    marker,
                                                )
                                            except Exception:
                                                logger.debug(
                                                    "Could not write post-optimization .validation_failed marker"
                                                )
                                        else:
                                            try:
                                                marker = os.path.join(
                                                    _output_dir,
                                                    ".validation_failed",
                                                )
                                                if os.path.exists(marker):
                                                    os.remove(marker)
                                                    logger.info(
                                                        "Removed stale validation failure marker after post-optimization: %s",
                                                        marker,
                                                    )
                                            except Exception:
                                                logger.debug(
                                                    "Could not remove post-optimization .validation_failed marker"
                                                )
                                    except Exception as post_e:
                                        logger.exception(
                                            "Post-optimization validator invocation failed: %s",
                                            post_e,
                                        )
                        except Exception:
                            logger.debug(
                                "Post-optimization verification check failed",
                                exc_info=True,
                            )
            except SystemExit:
                raise
            except Exception as e:
                logger.exception("Optimization failed: %s", e)

        # invoke clean up here
        try:
            try:
                from .export_helpers import cleanup_extraneous_onnx_files
            except Exception:
                cleanup_extraneous_onnx_files = None

            if cleanup_extraneous_onnx_files is not None:
                try:
                    cleanup_extraneous_onnx_files(
                        _output_dir,
                        logger,
                        bool(kwargs.get("cleanup", False)),
                        bool(kwargs.get("prune_canonical", False)),
                    )
                except Exception:
                    logger.debug("cleanup_extraneous_onnx_files failed", exc_info=True)
            else:
                logger.debug("cleanup_extraneous_onnx_files helper not available")
        except Exception:
            pass
        return _output_dir
    finally:
        # Cleanup any prepared local model snapshot created for this export
        try:
            if "prep_tmp_p" in locals() and prep_tmp_p:
                try:
                    if os.path.exists(prep_tmp_p):
                        shutil.rmtree(prep_tmp_p)
                        logger.info(
                            "Removed prepared local model folder: %s", prep_tmp_p
                        )
                    # Do not remove parent directories; only remove the prepared
                    # `temp_local` folder to avoid accidentally deleting output
                    # directories that may contain export artifacts.
                except Exception:
                    logger.debug(
                        "Failed to cleanup prepared local model folder: %s",
                        prep_tmp_p,
                        exc_info=True,
                    )
        except Exception:
            pass
        # Best-effort: clean other temporary export artifacts (system temp and any
        # lingering temp_local folders). This helps when clone/copy attempts
        # failed due to disk pressure and left behind partially-copied folders.
        try:
            try:
                from .export_helpers import cleanup_temporary_export_artifacts

                try:
                    cleanup_temporary_export_artifacts(logger=logger, base_dir=BASE_DIR)
                except Exception:
                    logger.debug(
                        "cleanup_temporary_export_artifacts raised", exc_info=True
                    )
            except Exception:
                # helper not available or import error; ignore
                pass
        except Exception:
            pass
        try:
            if file_handler is not None:
                try:
                    teardown_export_logging(
                        file_handler, logfile_fd, old_stdout, old_stderr, logger
                    )
                except Exception:
                    pass
        except Exception:
            pass


if __name__ == "__main__":
    print(
        "export_model_consolidated.py is an importable orchestrator. Use export_model.py for CLI."
    )
