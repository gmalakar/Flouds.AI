# =============================================================================
# File: export_v2.py
# Date: 2026-01-13
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

import json
import os
import shutil as _shutil
import stat as _stat
import subprocess
import sys
import tempfile
import tempfile as _tempfile
import time as _time
from typing import Any, Optional

from .onnx_helpers import get_default_opset


def _run_main_export_subprocess(me_kwargs: dict, logger: Any) -> tuple[bool, str]:
    """Run `optimum.exporters.onnx.main_export(**me_kwargs)` in a child
    Python process using a temporary JSON file for arguments.

    Returns (success, stderr_output).
    """
    tmpf = None
    try:
        tmpf = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(me_kwargs, tmpf)
        tmpf.close()

        # Instead of a long `-c` one-liner (which is fragile due to quoting
        # and newline/indentation), write a small temporary Python script to
        # normalize legacy `torch_dtype` keys to `dtype` recursively and
        # call `main_export`. This avoids shell quoting issues on Windows
        # and ensures the subprocess receives a valid Python script.
        runner_fh = None
        try:
            runner_fh = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".py", encoding="utf-8"
            )
            runner_code = (
                "import json,sys\n"
                "from optimum.exporters.onnx import main_export\n\n"
                "def _norm(d):\n"
                "    if isinstance(d, dict):\n"
                "        if 'torch_dtype' in d:\n"
                "            d['dtype'] = d.pop('torch_dtype')\n"
                "        for k, v in list(d.items()):\n"
                "            _norm(v)\n"
                "    elif isinstance(d, list):\n"
                "        for i in d:\n"
                "            _norm(i)\n\n"
                "kwargs = json.load(open(sys.argv[1], 'r', encoding='utf-8'))\n"
                "_norm(kwargs)\n"
                "main_export(**kwargs)\n"
            )
            runner_fh.write(runner_code)
            runner_fh.close()
            runner = runner_fh.name
        except Exception:
            if runner_fh is not None:
                try:
                    runner_fh.close()
                except Exception:
                    pass
            runner = None

        # Prepare a sanitized environment for the child process to reduce
        # native threading and memory pressure that can cause crashes.
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
        env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
        env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", "1")

        if runner:
            proc = subprocess.run(
                [sys.executable, runner, tmpf.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        else:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import json,sys; from optimum.exporters.onnx import main_export; main_export(**json.load(open(sys.argv[1])))",
                    tmpf.name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        if proc.stdout:
            try:
                logger.debug("main_export subprocess stdout:\n%s", proc.stdout)
            except Exception:
                pass
        if proc.stderr:
            try:
                logger.debug("main_export subprocess stderr:\n%s", proc.stderr)
            except Exception:
                pass

        return (proc.returncode == 0, proc.stderr or "")
    except Exception as e:
        try:
            logger.warning("Failed to launch main_export subprocess: %s", e)
        except Exception:
            pass
        return False, str(e)
    finally:
        try:
            if tmpf is not None:
                os.unlink(tmpf.name)
        except Exception:
            pass
        try:
            if (
                runner_fh is not None
                and runner_fh.name
                and os.path.exists(runner_fh.name)
            ):
                try:
                    os.unlink(runner_fh.name)
                except Exception:
                    pass
        except Exception:
            pass


def _clear_export_temp_dirs(logger: Any, prefixes=None, max_age_seconds: int = 60):
    """Remove export-related temp directories to free disk space.

    Removes directories under the system temp folder that start with any of
    the provided `prefixes` and are older than `max_age_seconds`.
    This is conservative but helps free disk space before large exports.
    """
    try:
        tdir = _tempfile.gettempdir()
    except Exception:
        try:
            logger.debug("Could not determine system temp dir for cleanup")
        except Exception:
            pass
        return

    try:
        try:
            logger.info("Cleaning export temp dirs in %s (prefixes=%s)", tdir, prefixes)
        except Exception:
            pass
    except Exception:
        pass

    if prefixes is None:
        prefixes = ("onnx_out_", "onnx_export_", "onnx_opt_clone_", "onnx_working_")

    now = _time.time()
    try:
        for name in os.listdir(tdir):
            try:
                if not any(name.startswith(pfx) for pfx in prefixes):
                    continue
                full = os.path.join(tdir, name)
                # Only remove directories (avoid touching files)
                if not os.path.isdir(full):
                    continue
                try:
                    mtime = os.path.getmtime(full)
                except Exception:
                    mtime = None
                # remove if older than threshold (or if mtime unavailable)
                if mtime is None or (now - mtime) > max_age_seconds:
                    try:
                        # Robust remove: on Windows PermissionError can occur for
                        # read-only files or files with restricted ACLs. Use an
                        # onerror handler to make files writable and retry.
                        def _on_rm_error(func, path, exc_info):
                            try:
                                # Attempt to set write permission and retry
                                os.chmod(path, _stat.S_IWRITE)
                            except Exception:
                                try:
                                    # best-effort: remove read-only flag on Windows
                                    import ctypes

                                    FILE_ATTRIBUTE_NORMAL = 0x80
                                    try:
                                        ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_NORMAL)  # type: ignore
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            try:
                                func(path)
                            except Exception:
                                pass

                        _shutil.rmtree(full, onexc=_on_rm_error)
                        try:
                            logger.info("Cleared temp export dir: %s", full)
                        except Exception:
                            pass
                    except Exception:
                        try:
                            logger.debug(
                                "Failed to remove temp dir: %s", full, exc_info=True
                            )
                        except Exception:
                            pass
            except Exception:
                continue
    except Exception:
        try:
            logger.debug("Export temp cleanup scan failed", exc_info=True)
        except Exception:
            pass


def export_v2_main_export(
    export_source: str,
    output_dir: str,
    model_for: str,
    opset_version: Optional[int],
    device: str,
    task: Optional[str],
    framework: Optional[str],
    library: Optional[str],
    trust_remote_code: bool,
    logger: Any,
    auth_token: Optional[str] = None,
    use_auth_token: Optional[bool] = None,
    use_external_data_format: bool = True,
    no_post_process: bool = False,
    merge: bool = False,
) -> tuple[bool, bool]:
    """Run optimum.exporters.onnx.main_export with sensible retries.

    Returns (success: bool, used_trust_remote_code: bool).
    """
    # We run the heavy v2 exporter in a subprocess to isolate native crashes.
    # Do not import `optimum.exporters.onnx` in-process here to avoid loading
    # heavy native libs into the parent interpreter.

    # Initial informational log; be permissive about logger failures
    try:
        logger.info(
            "export_v2_main_export start: export_source=%s output_dir=%s model_for=%s opset=%s device=%s task=%s framework=%s library=%s trust_remote_code=%s",
            export_source,
            output_dir,
            model_for,
            opset_version,
            device,
            task,
            framework,
            library,
            trust_remote_code,
        )
    except Exception:
        try:
            logger.info("export_v2_main_export start")
        except Exception:
            pass
    # Proactively clear any stale ONNX export temp dirs to free disk space
    try:
        _clear_export_temp_dirs(logger, max_age_seconds=3600)
    except Exception:
        try:
            logger.debug("Pre-export temp cleanup failed", exc_info=True)
        except Exception:
            pass
    logger.info(
        "Using v2 main_export for %s (opset=%s device=%s)",
        export_source,
        opset_version,
        device,
    )

    # Require explicit task from caller
    export_task = task

    export_library = library
    if not export_library:
        try:
            if "sentence-transformers" in (export_source or ""):
                export_library = "sentence_transformers"
        except Exception:
            export_library = None

    if not export_library:
        try:
            from huggingface_hub import HfApi  # type: ignore

            try:
                if "/" in str(export_source):
                    info = HfApi().model_info(str(export_source))
                    tags = getattr(info, "tags", []) or []
                    pipeline_tag = getattr(info, "pipeline_tag", None)
                    if any(
                        "sentence-transformers" in t for t in tags
                    ) or "sentence-transformers" in (export_source or ""):
                        export_library = "sentence_transformers"
                    elif pipeline_tag and "feature-extraction" in pipeline_tag:
                        export_library = "sentence_transformers"
            except Exception:
                pass
        except Exception:
            pass

    if not export_library:
        try:
            from transformers import AutoConfig  # type: ignore

            try:
                cfg = AutoConfig.from_pretrained(str(export_source))
                if getattr(cfg, "architectures", None):
                    archs = [a.lower() for a in cfg.architectures if isinstance(a, str)]
                    if any("sentence" in a or "sbert" in a for a in archs):
                        export_library = "sentence_transformers"
            except Exception:
                pass
        except Exception:
            pass

    opset = opset_version or get_default_opset()
    try:
        logger.debug("Selected opset for v2: %s", opset)
    except Exception:
        pass

    me_kwargs = {
        "model_name_or_path": export_source,
        "output": output_dir,
        "task": export_task,
        "opset": opset,
        "device": device or "cpu",
        "use_external_data_format": bool(use_external_data_format),
    }
    # If caller requested merge and this looks like a decoder-only causal LLM
    # export, prefer producing a merged artifact (no-past + with-past) during
    # export. Auto-detect KV-cache capability for decoder-only models and only
    # explicitly request `use_cache` for the subprocess when auto-detected and
    # a merge is requested (conservative behavior to avoid unnecessary cache
    # artifacts).
    try:
        auto_cache = model_for in ["llm", "causal-lm", "clm", "text-generation"]
        # Set `use_cache` for the subprocess when the model_for and task
        # indicate a text-generation causal LLM.
        try:
            if auto_cache and str(task or "").lower().startswith("text-generation"):
                me_kwargs["use_cache"] = True
                if merge and str(task or "").lower().endswith("-with-past"):
                    me_kwargs["use_merged"] = True
                    me_kwargs["file_name"] = "model.onnx"
        except Exception:
            pass

    except Exception:
        pass
    if no_post_process:
        me_kwargs["no_post_process"] = True
    # On Windows, exporting with external-data can create very long
    # paths for the .onnx.data* files when the final `output_dir` is deep.
    # Use a short temporary working output directory for the subprocess
    # and move the results back to the caller's `output_dir` on success.
    working_output: Optional[str] = output_dir
    if os.name == "nt":
        try:
            import shutil as _shutil  # local import

            working_output = tempfile.mkdtemp(prefix="onnx_out_")
            me_kwargs["output"] = working_output
            try:
                logger.debug(
                    "Using short working output for v2 export: %s", working_output
                )
            except Exception:
                pass
        except Exception:
            working_output = output_dir
    # Propagate token/auth options if provided so subprocess can authenticate
    if use_auth_token:
        me_kwargs["use_auth_token"] = True
    if auth_token:
        me_kwargs["use_auth_token"] = auth_token
    try:
        logger.debug(
            "Prepared main_export kwargs (pre-trust/library): %s",
            {k: v for k, v in me_kwargs.items()},
        )
    except Exception:
        pass
    if framework:
        me_kwargs["framework"] = framework
    if export_library:
        me_kwargs["library"] = export_library

    try:
        if trust_remote_code:
            me_kwargs["trust_remote_code"] = True
        try:
            logger.debug(
                "Calling main_export (subprocess) with kwargs: %s",
                {
                    k: ("<redacted>" if k.lower().endswith("token") else v)
                    for k, v in me_kwargs.items()
                },
            )
        except Exception:
            pass

        ok, stderr = _run_main_export_subprocess(me_kwargs, logger)
        if ok:
            try:
                # If we used a short working_output on Windows, move/copy
                # the artifacts back to the requested `output_dir`.
                if working_output and working_output != output_dir:
                    import shutil as _shutil

                    try:
                        if os.path.exists(output_dir):
                            _shutil.rmtree(output_dir)
                        _shutil.copytree(working_output, output_dir)
                        _shutil.rmtree(working_output)
                    except Exception as move_exc:
                        try:
                            logger.warning(
                                "Failed to move v2 export from working_output %s -> %s: %s",
                                working_output,
                                output_dir,
                                move_exc,
                            )
                        except Exception:
                            pass
                logger.info("v2 main_export saved artifacts to %s", output_dir)
            except Exception:
                pass
            return True, bool(trust_remote_code)

        err = (stderr or "main_export subprocess failed").lower()
        try:
            logger.warning("main_export failed (subprocess): %s", stderr)
        except Exception:
            pass

        if (
            "requires you to execute" in err
            or "trust_remote_code" in err
            or "execute the configuration" in err
        ) and not trust_remote_code:
            try:
                try:
                    logger.info(
                        "Retrying main_export with trust_remote_code=True due to remote code requirement"
                    )
                except Exception:
                    pass
                me_kwargs["trust_remote_code"] = True
                ok2, stderr2 = _run_main_export_subprocess(me_kwargs, logger)
                if ok2:
                    try:
                        logger.info(
                            "v2 main_export (trust_remote_code=True) saved artifacts to %s",
                            output_dir,
                        )
                    except Exception:
                        pass
                    return True, True
                try:
                    logger.warning(
                        "Fallback main_export (trust_remote_code=True) failed: %s",
                        stderr2,
                    )
                except Exception:
                    pass
            except Exception:
                try:
                    logger.warning(
                        "Fallback main_export with trust_remote_code=True raised an exception"
                    )
                except Exception:
                    pass

        if "failed to serialize proto" in err or "encodeerror" in err:
            try:
                # If the stderr indicates an operator that requires a newer
                # opset (e.g., scaled_dot_product_attention requires opset>=14),
                # select a higher fallback opset instead of blindly using 11.
                fallback_opset = 11
                try:
                    if "scaled_dot_product_attention" in (err or ""):
                        fallback_opset = 14
                except Exception:
                    pass
                try:
                    logger.info(
                        "Retrying main_export with opset=%s as fallback", fallback_opset
                    )
                except Exception:
                    pass
                me_kwargs["opset"] = fallback_opset
                ok3, stderr3 = _run_main_export_subprocess(me_kwargs, logger)
                if ok3:
                    try:
                        logger.info(
                            "v2 main_export (opset=%s) saved artifacts to %s",
                            fallback_opset,
                            output_dir,
                        )
                    except Exception:
                        pass
                    return True, False
                try:
                    logger.warning(
                        "Fallback main_export with opset=%s failed: %s",
                        fallback_opset,
                        stderr3,
                    )
                except Exception:
                    pass
            except Exception:
                try:
                    logger.warning("Fallback main_export with opset failed (exception)")
                except Exception:
                    pass
        else:
            try:
                logger.info("main_export failure not related to proto serialization")
            except Exception:
                pass

            # Detect optimum post-processing deduplication failures which often
            # manifest as MemoryError when large initializers are read into memory.
            # In that case, retrying with `no_post_process` can allow the export to
            # complete (tradeoff: post-processing/deduplication skipped).
            try:
                if any(
                    k in (err or "")
                    for k in (
                        "post-processing of the onnx export failed",
                        "remove_duplicate_weights_from_tied_info",
                        "numpy_helper.to_array",
                        "deduplicate_gather_matmul",
                    )
                ):
                    try:
                        logger.warning(
                            "Detected optimum post-processing failure; retrying with no_post_process=True"
                        )
                    except Exception:
                        pass
                    me_kwargs_pp = me_kwargs.copy()
                    me_kwargs_pp["no_post_process"] = True
                    # Also try conservative memory reducers alongside skipping post-process
                    me_kwargs_pp["dtype"] = "float16"
                    me_kwargs_pp["use_external_data_format"] = False
                    ok_pp, stderr_pp = _run_main_export_subprocess(me_kwargs_pp, logger)
                    if ok_pp:
                        try:
                            if working_output and working_output != output_dir:
                                import shutil as _sh

                                try:
                                    if os.path.exists(output_dir):
                                        _sh.rmtree(output_dir)
                                    _sh.copytree(working_output, output_dir)
                                    _sh.rmtree(working_output)
                                except Exception as move_exc:
                                    try:
                                        logger.warning(
                                            "Failed to move v2 export from working_output %s -> %s: %s",
                                            working_output,
                                            output_dir,
                                            move_exc,
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            logger.info(
                                "v2 main_export (no_post_process + float16 + no external data) saved artifacts to %s",
                                output_dir,
                            )
                        except Exception:
                            pass
                        return True, bool(trust_remote_code)
                    try:
                        logger.warning(
                            "Retry with no_post_process failed: %s", stderr_pp
                        )
                    except Exception:
                        pass
            except Exception:
                try:
                    logger.debug(
                        "Post-process detection/retry branch failed", exc_info=True
                    )
                except Exception:
                    pass

        # Detect MemoryError or out-of-memory traces in stderr and attempt
        # a conservative retry using lower precision (`float16`) to reduce
        # peak memory usage during export. This is best-effort and may not
        # always succeed, but helps on machines with limited RAM.
        try:
            if any(
                k in (err or "")
                for k in ("memoryerror", "out of memory", "memory error")
            ):
                try:
                    logger.warning(
                        "Detected MemoryError in main_export stderr; retrying with lower precision (float16)"
                    )
                except Exception:
                    pass
                me_kwargs_low = me_kwargs.copy()
                # Use the modern `dtype` key (optimum warns that `torch_dtype`
                # is deprecated). Set to string 'float16' so the runner
                # normalizer will accept and pass through as needed.
                me_kwargs_low["dtype"] = "float16"
                ok_low, stderr_low = _run_main_export_subprocess(me_kwargs_low, logger)
                if ok_low:
                    try:
                        logger.info(
                            "v2 main_export (float16 retry) saved artifacts to %s",
                            output_dir,
                        )
                    except Exception:
                        pass
                    return True, bool(trust_remote_code)
                try:
                    logger.warning(
                        "Float16 retry for main_export failed: %s", stderr_low
                    )
                except Exception:
                    pass
                # If float16 retry failed, try disabling external-data so the
                # exporter creates a single-file model (may avoid loading
                # external shards into memory during processing). Combine with
                # float16 to reduce memory footprint.
                try:
                    me_kwargs_noext = me_kwargs.copy()
                    me_kwargs_noext["dtype"] = "float16"
                    me_kwargs_noext["use_external_data_format"] = False
                    try:
                        logger.info(
                            "Retrying main_export with dtype=float16 and use_external_data_format=False"
                        )
                    except Exception:
                        pass
                    ok_noext, stderr_noext = _run_main_export_subprocess(
                        me_kwargs_noext, logger
                    )
                    if ok_noext:
                        try:
                            # Move artifacts if a short working_output was used
                            if working_output and working_output != output_dir:
                                import shutil as _sh

                                try:
                                    if os.path.exists(output_dir):
                                        _sh.rmtree(output_dir)
                                    _sh.copytree(working_output, output_dir)
                                    _sh.rmtree(working_output)
                                except Exception as move_exc:
                                    try:
                                        logger.warning(
                                            "Failed to move v2 export from working_output %s -> %s: %s",
                                            working_output,
                                            output_dir,
                                            move_exc,
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            logger.info(
                                "v2 main_export (float16 + no external data) saved artifacts to %s",
                                output_dir,
                            )
                        except Exception:
                            pass
                        return True, bool(trust_remote_code)
                    try:
                        logger.warning(
                            "Float16 + no-external-data retry failed: %s", stderr_noext
                        )
                    except Exception:
                        pass
                except Exception:
                    try:
                        logger.debug(
                            "No-external-data retry branch failed", exc_info=True
                        )
                    except Exception:
                        pass

                # Final conservative fallback: try opset=11 with float16 and
                # no external data format. This may succeed on older opsets
                # that serialize differently and avoid loading external shards
                # into memory.
                try:
                    me_kwargs_final = me_kwargs.copy()
                    me_kwargs_final["opset"] = 11
                    me_kwargs_final["dtype"] = "float16"
                    me_kwargs_final["use_external_data_format"] = False
                    try:
                        logger.info(
                            "Final fallback: retrying main_export with opset=11, dtype=float16, use_external_data_format=False"
                        )
                    except Exception:
                        pass
                    ok_final, stderr_final = _run_main_export_subprocess(
                        me_kwargs_final, logger
                    )
                    if ok_final:
                        try:
                            if working_output and working_output != output_dir:
                                import shutil as _sh

                                try:
                                    if os.path.exists(output_dir):
                                        _sh.rmtree(output_dir)
                                    _sh.copytree(working_output, output_dir)
                                    _sh.rmtree(working_output)
                                except Exception as move_exc:
                                    try:
                                        logger.warning(
                                            "Failed to move v2 export from working_output %s -> %s: %s",
                                            working_output,
                                            output_dir,
                                            move_exc,
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            logger.info(
                                "v2 main_export (opset=11 + float16 + no external data) saved artifacts to %s",
                                output_dir,
                            )
                        except Exception:
                            pass
                        return True, bool(trust_remote_code)
                    try:
                        logger.warning(
                            "Final fallback (opset=11 + float16 + no external data) failed: %s",
                            stderr_final,
                        )
                    except Exception:
                        pass
                except Exception:
                    try:
                        logger.debug("Final fallback branch failed", exc_info=True)
                    except Exception:
                        pass
        except Exception:
            try:
                logger.debug("MemoryError handling branch failed", exc_info=True)
            except Exception:
                pass

        # If allowed, try cloning/copying the source locally and retry with trust_remote_code
        if trust_remote_code:
            try:
                import shutil

                repo = str(export_source)
                # If the source is a local directory, try running directly to avoid copying large files
                if os.path.exists(repo) and os.path.isdir(repo):
                    try:
                        logger.info(
                            "Attempting v2 main_export directly on local model path for clone-based fallback: %s",
                            repo,
                        )
                    except Exception:
                        pass
                    me_kwargs_local = me_kwargs.copy()
                    me_kwargs_local["model_name_or_path"] = repo
                    me_kwargs_local["trust_remote_code"] = True
                    ok_local, stderr_local = _run_main_export_subprocess(
                        me_kwargs_local, logger
                    )
                    if ok_local:
                        try:
                            logger.info(
                                "v2 main_export (local path) saved artifacts to %s",
                                output_dir,
                            )
                        except Exception:
                            pass
                        return True, True
                    try:
                        logger.warning(
                            "Direct local-path main_export failed (subprocess): %s",
                            stderr_local,
                        )
                    except Exception:
                        pass

                tmp = tempfile.mkdtemp(prefix="onnx_export_")
                try:
                    if os.path.exists(repo) and os.path.isdir(repo):
                        try:
                            logger.info(
                                "Copying local model %s -> %s for clone-based v2 fallback",
                                repo,
                                tmp,
                            )
                        except Exception:
                            pass
                        try:
                            shutil.copytree(repo, tmp, dirs_exist_ok=True)
                        except TypeError:
                            shutil.copytree(repo, tmp)
                    else:
                        try:
                            logger.info(
                                "Cloning %s -> %s for clone-based v2 fallback",
                                repo,
                                tmp,
                            )
                        except Exception:
                            pass
                        # If the provided repo looks like a HF model id (owner/name)
                        # and is not an existing local path or an explicit URL,
                        # construct the HTTPS clone URL so `git clone` succeeds.
                        repo_to_clone = repo
                        try:
                            if (
                                ("/" in repo)
                                and (not repo.startswith("http://"))
                                and (not repo.startswith("https://"))
                                and (not os.path.exists(repo))
                            ):
                                repo_to_clone = f"https://huggingface.co/{repo}"
                        except Exception:
                            repo_to_clone = repo

                        subprocess.check_call(
                            ["git", "clone", "--depth", "1", repo_to_clone, tmp]
                        )

                    me_kwargs_clone = me_kwargs.copy()
                    me_kwargs_clone["model_name_or_path"] = tmp
                    me_kwargs_clone["trust_remote_code"] = True
                    ok_clone, stderr_clone = _run_main_export_subprocess(
                        me_kwargs_clone, logger
                    )
                    if ok_clone:
                        try:
                            logger.info(
                                "v2 main_export (clone) saved artifacts to %s",
                                output_dir,
                            )
                        except Exception:
                            pass
                        return True, True
                    try:
                        logger.warning(
                            "Clone-based main_export failed (subprocess): %s",
                            stderr_clone,
                        )
                    except Exception:
                        pass
                finally:
                    try:
                        shutil.rmtree(tmp)
                    except Exception:
                        pass
            except Exception as clone_err:
                try:
                    logger.warning("Clone-based v2 fallback failed: %s", clone_err)
                except Exception:
                    pass

        return False, False
    except Exception:
        try:
            logger.exception("Unexpected exception in export_v2_main_export")
        except Exception:
            pass
        # Attempt best-effort cleanup of transient ONNX export temp dirs we
        # may have created in the system temp directory. Delegate to the
        # centralized `_clear_export_temp_dirs` helper to ensure the same
        # robust removal semantics (onexc handler, logging) are used.
        try:
            prefixes = ("onnx_out_", "onnx_export_", "onnx_opt_clone_", "onnx_working_")
            try:
                _clear_export_temp_dirs(logger, prefixes=prefixes, max_age_seconds=300)
            except Exception:
                try:
                    logger.debug("Fallback export temp cleanup failed", exc_info=True)
                except Exception:
                    pass
        except Exception:
            try:
                logger.debug(
                    "Failed scanning system temp dir for export temp cleanup",
                    exc_info=True,
                )
            except Exception:
                pass
        return False, False
