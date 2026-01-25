# =============================================================================
# File: export_v1.py
# Date: 2026-01-10
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Any


def export_v1_ortmodel(
    export_source: str,
    output_dir: str,
    model_for: str,
    trust_remote_code: bool,
    logger: Any,
    opset_version: int | None = None,
    device: str | None = None,
    task: str | None = None,
    framework: str | None = None,
    library: str | None = None,
    use_external_data_format: bool = True,
    merge: bool = False,
) -> tuple[bool, bool]:
    """Perform legacy ORTModel export (v1-style).

    Returns True on success, False on failure.
    """
    v1_failed = False
    try:
        logger.info(
            "export_v1_ortmodel start: export_source=%s output_dir=%s model_for=%s trust_remote_code=%s opset=%s device=%s task=%s framework=%s library=%s use_external_data_format=%s",
            export_source,
            output_dir,
            model_for,
            trust_remote_code,
            opset_version,
            device,
            task,
            framework,
            library,
            use_external_data_format,
        )
    except Exception:
        # best-effort logging, don't fail the export due to formatting issues
        try:
            logger.info("export_v1_ortmodel start")
        except Exception:
            pass
    try:

        from optimum.onnxruntime import (
            ORTModelForCausalLM,
            ORTModelForFeatureExtraction,
            ORTModelForSeq2SeqLM,
            ORTModelForSequenceClassification,
        )

        # Enable use_cache only for causal/decoder-only models when the
        # task indicates text-generation (e.g., 'text-generation' or
        # 'text-generation-with-past'). This avoids creating cache
        # artifacts for non-generation tasks.
        logger.info("Using legacy ORTModel export path (use_v1=True)")

        # Some versions of optimum/transformers accept `trust_remote_code` in
        # `from_pretrained`. Try to pass it; fall back gracefully if unsupported.
        def _try_from_pretrained(cls, *args, **kwargs):
            try:
                logger.debug(
                    "Attempting %s.from_pretrained with kwargs: %s",
                    getattr(cls, "__name__", str(cls)),
                    {
                        k: ("<redacted>" if k.lower().endswith("token") else v)
                        for k, v in kwargs.items()
                    },
                )
                return cls.from_pretrained(*args, **kwargs)
            except TypeError:
                # `trust_remote_code` not supported by this API version
                if "trust_remote_code" in kwargs:
                    logger.info(
                        "%s.from_pretrained does not accept trust_remote_code; retrying without it",
                        getattr(cls, "__name__", str(cls)),
                    )
                    kwargs.pop("trust_remote_code", None)
                return cls.from_pretrained(*args, **kwargs)

        def _load_and_save(trust_flag: bool) -> tuple[bool, bool]:
            # Choose the appropriate ORTModel class based on `model_for`, then
            # attempt a `from_pretrained` and save. We no longer accept an
            # explicit `use_cache` parameter; instead auto-detect whether to
            # enable KV-cache for decoder-only / causal LLMs.
            kwargs = {"export": True, "trust_remote_code": trust_flag}

            if model_for in ["s2s", "seq2seq-lm"]:
                cls = ORTModelForSeq2SeqLM
            elif model_for in ["sc", "sequence-classification"]:
                cls = ORTModelForSequenceClassification
            elif model_for in ["llm", "causal-lm", "clm", "text-generation"]:
                cls = ORTModelForCausalLMc
                # Enable use_cache only for causal/decoder-only models when the
                # task indicates text-generation (e.g., 'text-generation' or
                # 'text-generation-with-past'). This avoids creating cache
                # artifacts for non-generation tasks.
                try:
                    if str(task or "").lower().startswith("text-generation"):
                        kwargs["use_cache"] = True
                        if merge and str(task or "").lower().endswith("-with-past"):
                            kwargs["use_merged"] = True
                            kwargs["file_name"] = "model.onnx"
                except Exception:
                    pass
            else:
                cls = ORTModelForFeatureExtraction

            logger.info(
                "Selected ORTModel class %s for model_for=%s (use_cache=%s)",
                getattr(cls, "__name__", str(cls)),
                model_for,
                model_for in ["llm", "causal-lm", "clm", "text-generation"],
            )
            logger.debug("Calling from_pretrained with kwargs: %s", kwargs)
            try:
                ort_model = _try_from_pretrained(cls, export_source, **kwargs)
            except Exception as load_exc:
                logger.exception(
                    "Failed to load ORTModel from_pretrained: %s", load_exc
                )
                raise

            try:
                logger.info("Saving ORTModel to %s", output_dir)
                ort_model.save_pretrained(output_dir)
                logger.info("ORTModel export saved to %s", output_dir)
            except Exception as save_exc:
                logger.exception(
                    "Failed to save ORTModel to %s: %s", output_dir, save_exc
                )
                raise
            finally:
                try:
                    del ort_model
                    logger.debug("Deleted ORTModel instance to free memory")
                except Exception:
                    pass

            return True, bool(trust_flag)

        # Try the normal v1 path first with the caller's `trust_remote_code`.
        logger.debug(
            "Attempting initial v1 load_and_save with trust_remote_code=%s",
            trust_remote_code,
        )
        ok, used_trust = _load_and_save(trust_remote_code)
        if ok:
            logger.info("export_v1_ortmodel: v1 export succeeded (initial attempt)")
            return ok, used_trust
    except Exception as e:
        msg = str(e).lower()
        logger.warning("ORTModel export failed: %s", e)
        v1_failed = True
        # If the failure indicates that executing remote repo code is required,
        # optionally retry with `trust_remote_code=True` if not already set.
        if (
            "requires you to execute" in msg
            or "trust_remote_code" in msg
            or "execute the configuration" in msg
        ) and not trust_remote_code:
            try:
                logger.info(
                    "Retrying ORTModel export with trust_remote_code=True due to remote code requirement"
                )
                # Retry using the centralized loader if available.
                if ("_load_and_save" in locals()) and (not trust_remote_code):
                    try:
                        ok2, used_trust2 = _load_and_save(True)
                        if ok2:
                            logger.info(
                                "export_v1_ortmodel: v1 export succeeded on trust_remote_code retry"
                            )
                            return True, True
                    except Exception as e2:
                        logger.warning(
                            "ORTModel export retry with trust_remote_code failed: %s",
                            e2,
                        )
            except Exception as e2:
                logger.warning(
                    "ORTModel export retry with trust_remote_code failed: %s", e2
                )

    # If the v1 attempt failed, optionally attempt a clone-based fallback that
    # runs the optimum `main_export` from a local clone (this requires
    # `trust_remote_code`). If v1 did not fail, skip this fallback.
    if not v1_failed or not trust_remote_code:
        logger.info(
            "Skipping clone-based fallback: v1_failed=%s trust_remote_code=%s",
            v1_failed,
            trust_remote_code,
        )
        return False, False

    try:
        logger.info(
            "Attempting clone-based fallback for ORTModel export (trust_remote_code=True)"
        )
        import shutil
        import subprocess
        import tempfile

        tmp = tempfile.mkdtemp(prefix="onnx_export_")
        try:
            repo = str(export_source)
            # If the export source is a local directory (prepared snapshot),
            # copy it rather than attempting a git clone which will fail.
            if os.path.exists(repo) and os.path.isdir(repo):
                logger.info("Copying local model %s -> %s", repo, tmp)
                try:
                    shutil.copytree(repo, tmp, dirs_exist_ok=True)
                except TypeError:
                    shutil.copytree(repo, tmp)
            else:
                logger.info("Cloning %s -> %s", repo, tmp)
                subprocess.check_call(["git", "clone", "--depth", "1", repo, tmp])

            export_task = task

            export_library = library
            try:
                if not export_library and "sentence-transformers" in (
                    export_source or ""
                ):
                    export_library = "sentence_transformers"
            except Exception:
                export_library = export_library

            me_kwargs = {
                "model_name_or_path": tmp,
                "output": output_dir,
                "task": export_task,
                "opset": opset_version,
                "device": device or "cpu",
                "use_external_data_format": bool(use_external_data_format),
            }
            # Determine whether the clone-based exporter should enable KV-cache
            # and/or request a merged artifact following the same rules as
            # the in-process v1 path: model_for indicates causal LLM and task
            # starts with 'text-generation' to enable `use_cache`. If `use_cache`
            # is enabled and the caller requested `merge` and the task ends in
            # '-with-past', request a merged artifact and a canonical file name.
            try:
                if model_for in ["llm", "causal-lm", "clm", "text-generation"] and str(
                    task or ""
                ).lower().startswith("text-generation"):
                    me_kwargs["use_cache"] = True
                    if merge and str(task or "").lower().endswith("-with-past"):
                        me_kwargs["use_merged"] = True
                        me_kwargs["file_name"] = "model.onnx"
            except Exception:
                pass
            if framework:
                me_kwargs["framework"] = framework
            if export_library:
                me_kwargs["library"] = export_library

            try:
                from optimum.exporters.onnx import main_export as _main_export

                me_kwargs["trust_remote_code"] = True
                _main_export(**me_kwargs)
                logger.info("Clone-based export succeeded and saved to %s", output_dir)
                return True, True
            except Exception as e_clone:
                logger.warning("Clone-based main_export failed: %s", e_clone)
        except Exception as git_err:
            logger.warning(
                "Failed to clone repository for fallback export: %s", git_err
            )
        finally:
            try:
                shutil.rmtree(tmp)
                logger.info("Removed temporary clone: %s", tmp)
            except Exception:
                logger.debug("Failed to remove temporary clone: %s", tmp)
    except Exception as wrap_err:
        logger.warning("Inline clone-based export also failed: %s", wrap_err)

    return False, False
