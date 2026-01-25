# =============================================================================
# File: export_helpers.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

#!/usr/bin/env python3
"""Helper utilities extracted from `export_model_consolidated.py`.

Contains:
- configure_protobuf(): increase protobuf message size limits where possible
- cleanup_validator_logging_handlers(): remove likely validator-added logging handlers
- is_pid_running(pid): portable check whether a pid is active

These are kept lightweight and safe to import from other modules.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Any, cast


def ensure_empty_dir(path: str, logger=None) -> bool:
    """Ensure `path` is an empty directory.

    If `path` exists and contains files or subdirectories, remove it and
    recreate it. Returns True on success, False otherwise.
    """
    try:
        if os.path.exists(path):
            # If it's a file, remove it and create directory
            if os.path.isfile(path) or os.path.islink(path):
                try:
                    os.remove(path)
                except Exception:
                    if logger:
                        logger.debug(
                            "Failed to remove file at path before mkdir: %s",
                            path,
                            exc_info=True,
                        )
                    return False
            elif os.path.isdir(path):
                # If directory exists and not empty, remove it entirely
                try:
                    # On Windows, rmtree may fail if files are in use; catch exceptions
                    shutil.rmtree(path)
                except Exception:
                    # Attempt conservative cleanup of contents
                    try:
                        for entry in os.listdir(path):
                            p = os.path.join(path, entry)
                            try:
                                if os.path.isdir(p):
                                    shutil.rmtree(p)
                                else:
                                    os.remove(p)
                            except Exception:
                                if logger:
                                    logger.debug(
                                        "Failed to remove path during cleanup: %s",
                                        p,
                                        exc_info=True,
                                    )
                        # Finally remove the directory itself
                        try:
                            os.rmdir(path)
                        except Exception:
                            pass
                    except Exception:
                        if logger:
                            logger.debug(
                                "Failed to cleanup directory contents: %s",
                                path,
                                exc_info=True,
                            )
                        return False
        # Recreate the directory
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception:
            if logger:
                logger.debug("Failed to create directory: %s", path, exc_info=True)
            return False
    except Exception:
        if logger:
            logger.debug(
                "ensure_empty_dir encountered an error for %s", path, exc_info=True
            )
        return False


def configure_protobuf() -> None:
    """Try a few strategies to increase protobuf message size limits for large models.

    This function attempts multiple approaches for different protobuf versions
    to increase global message size limits; falls back to setting an env var.
    """
    try:
        import google
        import google.protobuf
        import google.protobuf.message

        msg_cls = cast(Any, google.protobuf.message.Message)
        if hasattr(msg_cls, "_SetGlobalDefaultMaxMessageSize"):
            msg_cls._SetGlobalDefaultMaxMessageSize(2**31 - 1)  # type: ignore[attr-defined]
            logging.getLogger(__name__).info(
                "Protobuf message size limit set using _SetGlobalDefaultMaxMessageSize"
            )
        elif hasattr(msg_cls, "SetGlobalDefaultMaxMessageSize"):
            msg_cls.SetGlobalDefaultMaxMessageSize(2**31 - 1)  # type: ignore[attr-defined]
            logging.getLogger(__name__).info(
                "Protobuf message size limit set using SetGlobalDefaultMaxMessageSize"
            )
        else:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
            logging.getLogger(__name__).warning(
                "Using environment variable fallback for protobuf configuration"
            )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Could not configure protobuf message size: %s", e
        )
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"


def cleanup_validator_logging_handlers() -> None:
    """Remove likely validator-added logging handlers to avoid duplicate handlers.

    Heuristic: remove `FileHandler` instances and handlers with names that
    contain 'onnx' or 'validator'. This is defensive — it avoids touching
    non-file handlers that may be important to the caller.
    """
    try:
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                name = getattr(h, "name", "") or ""
                if (
                    isinstance(h, logging.FileHandler)
                    or "onnx" in name.lower()
                    or "validator" in name.lower()
                ):
                    try:
                        root.removeHandler(h)
                    except Exception:
                        pass
                    try:
                        h.close()
                    except Exception:
                        pass
            except Exception:
                # Ignore handler-cleanup errors — we don't want to fail export
                pass
    except Exception:
        # If logging isn't available for some reason, ignore
        pass


def is_pid_running(pid: int) -> bool:
    """Portable check whether a PID is active on this system.

    Tries `psutil.pid_exists` when `psutil` is available; otherwise falls
    back to sending signal 0 on POSIX or using `os.kill` and other heuristics
    compatible with Windows.
    """
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


def sanitize_onnx_initializers(output_dir: str, logger) -> int:
    """Sanitize ONNX model files in `output_dir` by deduplicating identical
    initializers (tied weights).

    For each .onnx file in the directory:
    - Load the model proto.
    - Compute a hash of each initializer tensor bytes.
    - If multiple initializer names share identical content, pick a canonical
      name and update all node inputs to reference the canonical name.
    - Remove duplicate initializer entries and save a backup of the original
      as `<name>.pre_sanitize.bak` before writing the sanitized model.

    Returns the number of models modified.
    """
    modified_count = 0
    try:
        import hashlib

        import numpy as _np

        try:
            import onnx
            from onnx import numpy_helper
        except Exception as e:
            try:
                logger.debug("ONNX package not available; skipping sanitization: %s", e)
            except Exception:
                pass
            return 0

        for fname in os.listdir(output_dir):
            if not fname.lower().endswith(".onnx"):
                continue
            path = os.path.join(output_dir, fname)
            try:
                m = onnx.load(path)
            except Exception as e:
                logger.debug(
                    "Failed to load ONNX model for sanitization: %s (%s)", path, e
                )
                continue

            init_map: dict[bytes, list[str]] = {}
            name_to_tensor = {}
            try:
                for init in list(getattr(m, "initializer", [])):
                    try:
                        arr = numpy_helper.to_array(init)
                        # stable hash over bytes
                        h = hashlib.sha256(arr.tobytes()).digest()
                        init_map.setdefault(h, []).append(init.name)
                        name_to_tensor[init.name] = arr
                    except Exception:
                        # fallback: hash name only
                        h = hashlib.sha256(init.name.encode("utf-8")).digest()
                        init_map.setdefault(h, []).append(init.name)
                        name_to_tensor[init.name] = None
            except Exception:
                logger.debug(
                    "Failed enumerating initializers for %s", path, exc_info=True
                )

            # Build replacement map for duplicate names
            replace_map: dict[str, str] = {}
            for h, names in init_map.items():
                if len(names) <= 1:
                    continue
                # choose canonical name: prefer names containing 'embed' or 'lm_head'
                canon = None
                for cand in names:
                    if "embed" in cand.lower() or "lm_head" in cand.lower():
                        canon = cand
                        break
                if canon is None:
                    canon = names[0]
                for n in names:
                    if n != canon:
                        replace_map[n] = canon

            if not replace_map:
                continue

            # Apply replacements in node inputs
            changed = False
            try:
                for node in m.graph.node:
                    for i, inp in enumerate(list(node.input)):
                        if inp in replace_map:
                            node.input[i] = replace_map[inp]
                            changed = True
                # Also update graph input names if they reference duplicates
                for gi in getattr(m.graph, "input", []):
                    if gi.name in replace_map:
                        gi.name = replace_map[gi.name]
                        changed = True
            except Exception:
                logger.debug(
                    "Failed to rewrite node inputs for %s", path, exc_info=True
                )

            if not changed:
                continue

            # Remove duplicate initializers
            try:
                keep_inits = []
                for init in list(getattr(m, "initializer", [])):
                    if init.name in replace_map:
                        # skip duplicate
                        continue
                    keep_inits.append(init)
                # replace initializer list
                del m.graph.initializer[:]
                m.graph.initializer.extend(keep_inits)
            except Exception:
                logger.debug(
                    "Failed to prune duplicate initializers for %s", path, exc_info=True
                )

            # Backup and write sanitized model
            try:
                bak = path + ".pre_sanitize.bak"
                try:
                    if os.path.exists(bak):
                        os.remove(bak)
                    shutil.copy2(path, bak)
                except Exception:
                    logger.debug("Could not write backup for %s", path, exc_info=True)
                onnx.save(m, path)
                logger.info(
                    "Sanitized ONNX initializers for %s (replaced %d names)",
                    path,
                    len(replace_map),
                )
                modified_count += 1
            except Exception:
                logger.warning(
                    "Failed to save sanitized ONNX model for %s", path, exc_info=True
                )
    except Exception:
        logger.debug("sanitize_onnx_initializers failed", exc_info=True)

    return modified_count


def cleanup_extraneous_onnx_files(
    output_dir_arg: str, logger_arg, cleanup_flag: bool, prune_canonical: bool = False
) -> None:
    """Remove extraneous .onnx files from `output_dir_arg`.

    Behavior mirrors the previous `_do_cleanup` helper that lived inside
    `export_optimizer.run_optimization`: it preserves prioritized artifacts
    (merged_optimized -> merged -> optimized) and always keeps canonical
    filenames like `model.onnx`, `encoder_model.onnx`, `decoder_model.onnx`.
    If `cleanup_flag` is False this function is a no-op.
    """
    try:
        files = [f for f in os.listdir(output_dir_arg) if f.lower().endswith(".onnx")]
        try:
            logger_arg.info("Found ONNX files for cleanup: %s", sorted(files))
        except Exception:
            pass

        def _matches_any(sub: str, lst: list[str]) -> bool:
            sub = sub.lower()
            return any(sub in f.lower() for f in lst)

        # If cleanup is not requested, skip cleanup entirely.
        if not cleanup_flag:
            try:
                logger_arg.info("Cleanup skipped because --cleanup is false")
            except Exception:
                pass
            return

        # Prioritized cleanup sequence: prefer `merged_optimized` -> `merged` -> `optimized`.
        if _matches_any("merged_optimized", files):
            keep_set = {f for f in files if "merged_optimized" in f.lower()}
            try:
                logger_arg.info(
                    "Found merged_optimized artifacts; keeping only those files"
                )
            except Exception:
                pass
        elif _matches_any("merged", files):
            keep_set = {f for f in files if "merged" in f.lower()}
            try:
                logger_arg.info("Found merged artifacts; keeping only those files")
            except Exception:
                pass
        elif _matches_any("optimized", files):
            keep_set = {f for f in files if "optimized" in f.lower()}
            try:
                logger_arg.info("Found optimized artifacts; keeping only those files")
            except Exception:
                pass
        else:
            # Default conservative behavior: delete only `_with_past` artifacts
            keep_set = set(files) - {f for f in files if "_with_past" in f.lower()}
            try:
                logger_arg.info(
                    "No merged/optimized artifacts found; deleting only _with_past files"
                )
            except Exception:
                pass

        # Always preserve common canonical filenames if present
        canonical_names = {"model.onnx", "encoder_model.onnx", "decoder_model.onnx"}
        for cn in canonical_names:
            if cn in files:
                keep_set.add(cn)

        # Optionally prune canonical files when merged artifacts exist.
        # If `prune_canonical` is True and a merged variant is present for a
        # canonical artifact (e.g., `decoder_model_merged.onnx`), allow the
        # canonical file to be removed by discarding it from `keep_set`.
        if prune_canonical:
            try:
                # Helper to detect merged variants for a base name
                def _has_merged_variant(base: str) -> bool:
                    base_l = base.lower()
                    for f in files:
                        fl = f.lower()
                        # require the base to appear and 'merged' keyword present
                        if base_l in fl and "merged" in fl:
                            return True
                        # prefer merged_optimized variants too
                        if base_l in fl and "merged_optimized" in fl:
                            return True
                    return False

                # Prune decoder canonical
                if (
                    _has_merged_variant("decoder_model")
                    and "decoder_model.onnx" in keep_set
                ):
                    keep_set.discard("decoder_model.onnx")

                # Prune encoder canonical
                if (
                    _has_merged_variant("encoder_model")
                    and "encoder_model.onnx" in keep_set
                ):
                    keep_set.discard("encoder_model.onnx")

                # Prune single-file model canonical only if a non-encoder/decoder
                # merged `model` artifact exists (avoid matching encoder/decoder)
                has_model_merged = any(
                    (
                        "model" in f.lower()
                        and "merged" in f.lower()
                        and "encoder" not in f.lower()
                        and "decoder" not in f.lower()
                    )
                    for f in files
                )
                if has_model_merged and "model.onnx" in keep_set:
                    keep_set.discard("model.onnx")
                # Additionally, if pruning is requested and only optimized
                # variants exist (e.g., `model_optimized.onnx`), allow removing
                # the canonical file as well to prefer the optimized artifact.
                try:

                    def _has_optimized_variant(base: str) -> bool:
                        base_l = base.lower()
                        for f in files:
                            fl = f.lower()
                            if base_l in fl and "optimized" in fl:
                                return True
                        return False

                    if (
                        _has_optimized_variant("decoder_model")
                        and "decoder_model.onnx" in keep_set
                    ):
                        keep_set.discard("decoder_model.onnx")
                    if (
                        _has_optimized_variant("encoder_model")
                        and "encoder_model.onnx" in keep_set
                    ):
                        keep_set.discard("encoder_model.onnx")
                    # For single-file model, ensure we don't accidentally match encoder/decoder
                    if (
                        _has_optimized_variant("model")
                        and "model.onnx" in keep_set
                        and not any(
                            k
                            for k in files
                            if "encoder" in k.lower() and "optimized" in k.lower()
                        )
                        and not any(
                            k
                            for k in files
                            if "decoder" in k.lower() and "optimized" in k.lower()
                        )
                    ):
                        keep_set.discard("model.onnx")
                except Exception:
                    try:
                        logger_arg.debug(
                            "Optimized-variant prune check failed", exc_info=True
                        )
                    except Exception:
                        pass
            except Exception:
                try:
                    logger_arg.debug("Prune canonical check failed", exc_info=True)
                except Exception:
                    pass

        try:
            logger_arg.info(
                "Cleaning up unneeded .onnx files in %s (keeping: %s)",
                output_dir_arg,
                sorted(keep_set),
            )
        except Exception:
            pass

        # Build a reason map for files we keep so we can log per-file rationale
        reasons: dict[str, str] = {}
        try:
            for f in files:
                fl = f.lower()
                if f in keep_set:
                    if "merged_optimized" in fl:
                        reasons[f] = "keep (merged_optimized)"
                    elif "merged" in fl:
                        reasons[f] = "keep (merged)"
                    elif "optimized" in fl:
                        reasons[f] = "keep (optimized)"
                    elif "_with_past" in fl:
                        reasons[f] = "keep (with_past)"
                    elif f in {
                        "model.onnx",
                        "encoder_model.onnx",
                        "decoder_model.onnx",
                    }:
                        reasons[f] = "keep (canonical)"
                    else:
                        reasons[f] = "keep (other)"
                else:
                    # not in keep_set -> marked for removal
                    reasons[f] = "remove (not in keep set)"
        except Exception:
            pass

        removed: list[str] = []
        failed: list[str] = []
        for entry in files:
            try:
                target = os.path.join(output_dir_arg, entry)
                reason = reasons.get(entry, "no-reason")
                if entry in keep_set:
                    try:
                        logger_arg.info("Keeping %s: %s", entry, reason)
                    except Exception:
                        pass
                    continue

                try:
                    os.remove(target)
                    removed.append(entry)
                    try:
                        logger_arg.info("Removed %s: %s", entry, reason)
                    except Exception:
                        pass
                except Exception:
                    failed.append(entry)
                    try:
                        logger_arg.warning("Failed to remove %s: %s", entry, reason)
                    except Exception:
                        pass
            except Exception:
                try:
                    logger_arg.debug(
                        "Error while processing ONNX cleanup for %s",
                        entry,
                        exc_info=True,
                    )
                except Exception:
                    pass
    except Exception:
        try:
            logger_arg.debug("Cleanup of extraneous ONNX files failed", exc_info=True)
        except Exception:
            pass

    # Also remove any leftover backup files created during merges
    bak_removed: list[str] = []
    bak_failed: list[str] = []
    try:
        for entry in os.listdir(output_dir_arg):
            if not entry.lower().endswith(".pre_with_past.bak"):
                continue
            bp = os.path.join(output_dir_arg, entry)
            try:
                os.remove(bp)
                bak_removed.append(entry)
                try:
                    logger_arg.info("Removed merge backup file: %s", entry)
                except Exception:
                    pass
            except Exception:
                bak_failed.append(entry)
                try:
                    logger_arg.warning("Failed to remove merge backup file: %s", entry)
                except Exception:
                    pass
    except Exception:
        try:
            logger_arg.debug(
                "Error while scanning for merge backup files", exc_info=True
            )
        except Exception:
            pass

    try:
        logger_arg.info(
            "Cleanup summary for %s: removed=%d failed=%d backups_removed=%d backups_failed=%d kept=%d",
            output_dir_arg,
            len(removed),
            len(failed),
            len(bak_removed),
            len(bak_failed),
            len(keep_set),
        )
    except Exception:
        pass


# The helper `prepare_local_model_dir` implementation below is the canonical
# version used by the exporter. The earlier, duplicate implementation was
# removed to avoid unreachable/nested try/except blocks and to ensure a single
# clear code path for preparing local model snapshots.


def prepare_local_model_dir(
    model_name: str, dest_dir: str, trust_remote_code: bool, logger
) -> bool:
    """Prepare a local copy of `model_name` under `dest_dir`.

    Loads the model with reduced memory flags where supported, calls the
    model's `tie_weights()`/`tie_word_embeddings()` helper to canonicalize
    shared tensors, and saves the model and tokenizer to `dest_dir`.

    Returns True on success, False otherwise. This is safe to call from the
    exporter orchestrator to avoid duplicate initializer issues in ONNX.
    """
    try:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            logger.warning("Could not import transformers/torch for local prep: %s", e)
            return False

        def cleanup_temporary_export_artifacts(
            logger=None, base_dir: str | None = None
        ) -> None:
            """Attempt best-effort removal of temporary export artifacts.

            - Remove directories in the system temp dir that start with common prefixes
              used by this tooling (`onnx_export_`, `onnx_opt_clone_`).
            - Remove any lingering `temp_local` folders under the repository's
              `onnx/models/*/*/temp_local` path when `base_dir` is provided.

            This is conservative and only removes folders that match expected
            patterns to avoid accidental deletions.
            """
            try:
                import glob
                import tempfile

                tdir = tempfile.gettempdir()
                prefixes = ("onnx_export_", "onnx_opt_clone_")
                for name in os.listdir(tdir):
                    try:
                        for p in prefixes:
                            if name.startswith(p):
                                path = os.path.join(tdir, name)
                                try:
                                    if os.path.isdir(path):
                                        shutil.rmtree(path)
                                        if logger:
                                            logger.info(
                                                "Removed temp export dir: %s", path
                                            )
                                except Exception:
                                    if logger:
                                        logger.debug(
                                            "Failed to remove temp dir: %s",
                                            path,
                                            exc_info=True,
                                        )
                                break
                    except Exception:
                        continue

                # Remove any lingering temp_local under repository onnx/models if base_dir provided
                if base_dir:
                    try:
                        pattern = os.path.join(
                            base_dir, "onnx", "models", "**", "temp_local"
                        )
                        for p in glob.glob(pattern, recursive=True):
                            try:
                                if os.path.isdir(p):
                                    shutil.rmtree(p)
                                    if logger:
                                        logger.info(
                                            "Removed lingering temp_local folder: %s", p
                                        )
                            except Exception:
                                if logger:
                                    logger.debug(
                                        "Failed to remove temp_local folder: %s",
                                        p,
                                        exc_info=True,
                                    )
                    except Exception:
                        if logger:
                            logger.debug(
                                "Failed scanning for temp_local folders under %s",
                                base_dir,
                                exc_info=True,
                            )
            except Exception:
                if logger:
                    logger.debug(
                        "cleanup_temporary_export_artifacts encountered an error",
                        exc_info=True,
                    )

        # Ensure the destination directory is empty before saving prepared model
        try:
            ensure_empty_dir(dest_dir, logger=logger)
        except Exception:
            try:
                os.makedirs(dest_dir, exist_ok=True)
            except Exception:
                logger.warning("Could not create dest_dir %s", dest_dir)
                return False
        logger.info("Preparing local model copy for %s -> %s", model_name, dest_dir)

        # Use low_cpu_mem_usage where available to reduce peak memory during load
        try:
            logger.info("Loading model %s (low_cpu_mem_usage=True) ...", model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            logger.info("Model loaded: %s", getattr(model, "__class__", type(model)))
        except TypeError:
            # Older transformers may not support low_cpu_mem_usage
            logger.info("low_cpu_mem_usage not supported; loading model without it")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            logger.info(
                "Model loaded (fallback): %s", getattr(model, "__class__", type(model))
            )
        except Exception as load_exc:
            logger.exception(
                "Failed to load model %s during local prep: %s", model_name, load_exc
            )
            return False

        try:
            if hasattr(model, "tie_weights"):
                logger.info("Invoking tie_weights() to canonicalize shared tensors")
                model.tie_weights()
                logger.info("tie_weights() completed")
            elif hasattr(model, "tie_word_embeddings"):
                logger.info(
                    "Invoking tie_word_embeddings() to canonicalize shared tensors"
                )
                model.tie_word_embeddings()
                logger.info("tie_word_embeddings() completed")
            else:
                logger.info(
                    "Model has no tie_weights/tie_word_embeddings helper; skipping canonicalization"
                )
        except Exception as e:
            logger.exception(
                "tie_weights/tie_word_embeddings failed during local prep: %s", e
            )

        # Ensure lm_head and embedding share the same Parameter reference when possible.
        try:
            emb = None
            try:
                if hasattr(model, "get_input_embeddings"):
                    ie = model.get_input_embeddings()
                    emb = getattr(ie, "weight", None)
            except Exception:
                emb = None

            lm = None
            try:
                lm_mod = getattr(model, "lm_head", None)
                lm = getattr(lm_mod, "weight", None) if lm_mod is not None else None
            except Exception:
                lm = None

            if emb is not None and lm is not None:
                # If they are not the same object, force them to reference the same
                if emb is not lm:
                    try:
                        logger.info(
                            "Forcing lm_head.weight to reference input_embeddings.weight to canonicalize tied weights"
                        )
                        # Assign the existing embedding Parameter to the lm_head module
                        setattr(lm_mod, "weight", emb)
                        logger.info(
                            "Forced tie: lm_head.weight now shares input_embeddings.weight"
                        )
                    except Exception as tie_err:
                        logger.warning(
                            "Failed to force tie between lm_head and embeddings: %s",
                            tie_err,
                        )
            else:
                logger.debug(
                    "Could not locate both embeddings and lm_head modules for forced tie (emb=%s lm=%s)",
                    bool(emb),
                    bool(lm),
                )
        except Exception:
            logger.debug("Embedding/lm_head tie enforcement failed", exc_info=True)

        try:
            logger.info("Saving prepared model to %s", dest_dir)
            model.save_pretrained(dest_dir)
            logger.info("Saved local model to %s", dest_dir)
        except Exception as e:
            logger.exception("Failed to save local model to %s: %s", dest_dir, e)
            try:
                shutil.rmtree(dest_dir)
            except Exception:
                logger.debug(
                    "Could not cleanup partial dest_dir: %s", dest_dir, exc_info=True
                )
            return False

        try:
            logger.info("Saving tokenizer for %s", model_name)
            try:
                tok = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=trust_remote_code,
                    fix_mistral_regex=True,
                )
            except TypeError:
                tok = AutoTokenizer.from_pretrained(
                    model_name, use_fast=False, trust_remote_code=trust_remote_code
                )
            tok.save_pretrained(dest_dir)
            logger.info("Saved tokenizer to %s", dest_dir)
        except Exception as e:
            logger.warning("Failed to save tokenizer during local prep: %s", e)

        return True
    except Exception:
        logger.debug("prepare_local_model_dir failed", exc_info=True)
        return False
