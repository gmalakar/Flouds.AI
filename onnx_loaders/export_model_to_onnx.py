# =============================================================================
# File: export_model_to_onnx.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import gc
import glob
import logging
import os
import pathlib
import warnings

# Set ONNX opset version to 14 to support scaled_dot_product_attention
os.environ["ONNX_OPSET_VERSION"] = "14"

try:
    from torch.onnx._internal.registration import OnnxExporterWarning

    warnings.filterwarnings("ignore", category=OnnxExporterWarning)
except ImportError:
    pass

from onnxruntime import InferenceSession
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTOptimizer,
)
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import AutoTokenizer, T5EncoderModel, pipeline

from onnx import checker, load

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Symbolic function.*already registered.*")
warnings.filterwarnings(
    "ignore", message=".*were not initialized from the model checkpoint.*"
)
warnings.filterwarnings("ignore", message=".*are newly initialized.*")

logger = logging.getLogger(__name__)


def export_and_optimize_onnx(
    model_name: str,
    model_for: str = "fe",
    optimize: bool = False,
    optimization_level: int = 1,
    task: str = None,
    use_t5_encoder: bool = False,
    use_cache: bool = False,
    model_folder: str = None,
    onnx_path: str = "../onnx",
    framework: str = None,
    force: bool = False,
):
    """
    Export and optionally optimize a HuggingFace model to ONNX format.
    """
    # Input validation
    if not model_name or not model_name.strip():
        raise ValueError("model_name cannot be empty")

    if model_for not in [
        "fe",
        "s2s",
        "sc",
        "feature-extraction",
        "seq2seq-lm",
        "sequence-classification",
    ]:
        raise ValueError(f"Invalid model_for: {model_for}. Must be one of: fe, s2s, sc")

    logger.info(
        f"Starting export for {model_name} (model_for={model_for}, optimize={optimize})"
    )
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _model_for = model_for.lower()
    _output_base = os.path.join(onnx_path, "models")
    _onnx_name = "model.onnx"
    _decoder_onnx_name = "decoder_model.onnx"
    _has_decoder = False
    _export_args = {}
    _model_path = None
    _output_dir = None

    def _get_task_name(model_for: str, task_type: str):
        """
        Returns the ONNX export task name.
        If task_type is provided and not 'none', use it.
        Otherwise, infer from model_for.
        """
        if task_type and task_type.lower() != "none":
            return task_type
        model_for = model_for.lower()
        if model_for in ["s2s", "seq2seq-lm"]:
            return "seq2seq-lm"
        elif model_for in ["sc", "sequence-classification"]:
            return "sequence-classification"
        elif model_for in ["fe", "feature-extraction"]:
            return "feature-extraction"
        else:
            # Default fallback
            return "feature-extraction"

    def _verify_model(model_path: str):
        """Verify ONNX model is valid and can be loaded."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Verifying model at {model_path}...")
        try:
            onnx_model = load(model_path)
            try:
                checker.check_model(onnx_model)
                logger.info("✅ ONNX model validation passed")
            except MemoryError:
                logger.warning("⚠️ Skipping onnx.checker.check_model due to MemoryError")

            session2 = InferenceSession(model_path)
            logger.info("Model inputs: %s", [inp.name for inp in session2.get_inputs()])
            logger.info(
                "Model outputs: %s", [out.name for out in session2.get_outputs()]
            )

            del onnx_model, session2
            gc.collect()
            return True

        except Exception as e:
            logger.error(f"❌ Model verification failed: {e}")
            return False

    # --- Export logic ---
    _model_task_type = _get_task_name(model_for, task)
    if not model_folder:
        model_folder = model_name.split("/")[-1] if "/" in model_name else model_name

    _output_base = os.path.join(_output_base, _model_for)
    _output_dir = os.path.join(BASE_DIR, _output_base, model_folder)
    os.makedirs(_output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(_output_dir, "*.onnx")):
        os.remove(f)

    if use_t5_encoder:
        logger.info(f"Exporting T5 encoder-only for {model_name} ...")
        encoder = T5EncoderModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder.save_pretrained(_output_dir)
        tokenizer.save_pretrained(_output_dir)
        model = ORTModelForFeatureExtraction.from_pretrained(_output_dir, export=True)
        model.save_pretrained(_output_dir)
        logger.info(f"Encoder ONNX model exported to {_output_dir} successfully.")
        del model, tokenizer, encoder
        gc.collect()
        _model_path = os.path.join(_output_dir, "model.onnx")
        _has_decoder = False
    elif _model_for in ["s2s", "seq2seq-lm"]:
        # Check if ONNX files already exist
        encoder_exists = os.path.exists(os.path.join(_output_dir, "encoder_model.onnx"))
        decoder_exists = os.path.exists(os.path.join(_output_dir, "decoder_model.onnx"))

        _export_args = {
            "export": force or not (encoder_exists and decoder_exists),
            "use_cache": use_cache,
        }
        _onnx_name = "encoder_model.onnx"
        _decoder_onnx_name = "decoder_model.onnx"

        if force:
            logger.info(f"Force re-exporting seq2seq model {model_name}...")
        elif encoder_exists and decoder_exists:
            logger.info(f"Using existing ONNX files for {model_name}")
        else:
            logger.info(
                f"Exporting seq2seq model {model_name} for task_type {_model_task_type} and use_cache {use_cache}..."
            )
        # Set opset version for export
        os.environ["TORCH_ONNX_OPSET_VERSION"] = "14"
        # Disable scaled_dot_product_attention for ONNX export compatibility
        import torch

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        model = ORTModelForSeq2SeqLM.from_pretrained(model_name, **_export_args)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(_output_dir)
        tokenizer.save_pretrained(_output_dir)
        logger.info(f"Model exported to ONNX format in {_output_dir}")
        for fname in [
            "encoder_model.onnx",
            "decoder_model.onnx",
            "decoder_with_past_model.onnx",
        ]:
            fpath = os.path.join(_output_dir, fname)
            logger.info(f"{fpath} exists: {os.path.exists(fpath)}")
        _has_decoder = True
        _model_path = os.path.join(_output_dir, _onnx_name)
    elif _model_for in ["sc", "sequence-classification"]:
        _export_args = {
            "export": True,
            "use_cache": use_cache,
        }
        logger.info(
            f"Exporting sequence classification model {model_name} for task_type {_model_task_type} and use_cache {use_cache}..."
        )
        # Set opset version for export
        os.environ["TORCH_ONNX_OPSET_VERSION"] = "14"
        # Disable scaled_dot_product_attention for ONNX export compatibility
        import torch

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        model = ORTModelForSequenceClassification.from_pretrained(
            model_name, **_export_args
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(_output_dir)
        tokenizer.save_pretrained(_output_dir)
        logger.info(f"Model exported to ONNX format in {_output_dir}")
        _has_decoder = False
        _model_path = os.path.join(_output_dir, "model.onnx")
    else:
        from optimum.exporters.onnx import OnnxConfig

        _export_args = {"export": True}  # Do NOT include 'task' for feature extraction!
        logger.info(f"Exporting feature extraction model {model_name}...")
        # Set opset version for export
        os.environ["TORCH_ONNX_OPSET_VERSION"] = "14"
        # Disable scaled_dot_product_attention for ONNX export compatibility
        import torch

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        model = ORTModelForFeatureExtraction.from_pretrained(model_name, **_export_args)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(_output_dir)
        tokenizer.save_pretrained(_output_dir)
        logger.info(f"Model exported to ONNX format in {_output_dir}")
        _has_decoder = False
        _model_path = os.path.join(_output_dir, "model.onnx")

    logger.info(f"Model path: {_model_path}")

    # Verify model
    _verify_model(_model_path)

    # Clean up memory before optimization
    if optimize:
        logger.info("Cleaning up memory before optimization...")
        if "model" in locals():
            del model
        if "tokenizer" in locals():
            del tokenizer

        # Aggressive memory cleanup
        import torch

        gc.collect()
        gc.collect()  # Run twice for better cleanup

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force system memory cleanup
        try:
            import ctypes

            if os.name == "nt":
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except (ImportError, AttributeError, OSError):
            pass

        logger.info("Memory cleanup completed")

        # Check available memory
        import psutil

        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        logger.info(f"Available memory: {available_memory_mb:.1f} MB")

        if available_memory_mb < 500:  # Less than 500MB available
            logger.warning(
                f"Insufficient memory ({available_memory_mb:.1f} MB) for optimization. Skipping optimization."
            )
            return

    # Optimize model if required
    def _optimize_model():
        if optimize:
            try:
                optimization_config = OptimizationConfig(
                    optimization_level=optimization_level
                )
                # Optimize main model
                if _model_for in ["s2s", "seq2seq-lm"]:
                    model = ORTModelForSeq2SeqLM.from_pretrained(
                        _output_dir, use_cache=False
                    )
                elif _model_for in ["sc", "sequence-classification"]:
                    model = ORTModelForSequenceClassification.from_pretrained(
                        _output_dir
                    )
                else:
                    model = ORTModelForFeatureExtraction.from_pretrained(_output_dir)
                optimizer = ORTOptimizer.from_pretrained(model)
                optimizer.optimize(
                    save_dir=pathlib.Path(_model_path).parent,
                    optimization_config=optimization_config,
                )
                logger.info(f"✅ Optimized model saved as: {_model_path}")
                del model
                gc.collect()
            except Exception as e:
                logger.warning(
                    f"⚠️ Optimization failed with level {optimization_level}: {e}"
                )
                if optimization_level > 0:
                    logger.info("🔄 Trying with basic optimization (level 0)...")
                    try:
                        optimization_config = OptimizationConfig(optimization_level=0)
                        if _model_for in ["s2s", "seq2seq-lm"]:
                            model = ORTModelForSeq2SeqLM.from_pretrained(
                                _output_dir, use_cache=False
                            )
                        elif _model_for in ["sc", "sequence-classification"]:
                            model = ORTModelForSequenceClassification.from_pretrained(
                                _output_dir
                            )
                        else:
                            model = ORTModelForFeatureExtraction.from_pretrained(
                                _output_dir
                            )
                        optimizer = ORTOptimizer.from_pretrained(model)
                        optimizer.optimize(
                            save_dir=pathlib.Path(_model_path).parent,
                            optimization_config=optimization_config,
                        )
                        logger.info(f"✅ Basic optimization successful")
                        del model
                        gc.collect()
                    except Exception as e2:
                        logger.warning(f"⚠️ Basic optimization also failed: {e2}")
                        logger.info("📦 Using unoptimized model (still functional)")
                else:
                    logger.info("📦 Using unoptimized model (still functional)")

            # Optimize decoder if present
            if _has_decoder:
                decoder_model_path = os.path.join(_output_dir, _decoder_onnx_name)
                logger.info(
                    f"{decoder_model_path} exists: {pathlib.Path(decoder_model_path).exists()}"
                )
                _verify_model(decoder_model_path)
                try:
                    decoder_model = ORTModelForSeq2SeqLM.from_pretrained(
                        _output_dir, use_cache=False
                    )
                    decoder_optimizer = ORTOptimizer.from_pretrained(decoder_model)
                    optimized_decoder_model = decoder_optimizer.optimize(
                        save_dir=pathlib.Path(decoder_model_path).parent,
                        optimization_config=optimization_config,
                    )
                    logger.info(
                        f"Decoder model optimized and saved as: {optimized_decoder_model.name}"
                    )
                    del decoder_model, decoder_optimizer, optimized_decoder_model
                except Exception as e:
                    logger.error(f"Decoder optimization failed: {e}")
                gc.collect()

    _optimize_model()


# HINTS:
# - For summarization (BART, T5, Pegasus, etc.), use --model_for s2s and --task text2text-generation (or --task seq2seq-lm for legacy).
# - For sequence classification (BERT, RoBERTa, etc.), use --model_for sc and --task sequence-classification.
# - For embeddings/feature extraction, use --model_for fe or --use_t5_encoder for T5 encoder-only export.
# - If you do not specify --task, it will be inferred from --model_for.
# - After exporting a seq2seq model, you should see encoder_model.onnx, decoder_model.onnx, and decoder_with_past_model.onnx in the output directory.
# - If decoder_with_past_model.onnx is missing, the model cannot be used for fast autoregressive generation (greedy decoding).
# - Always verify ONNX model inputs/outputs after export to ensure compatibility with your inference pipeline.
# - Optimization is optional but recommended for production; it can reduce inference latency.
# - If you encounter export errors, check that your optimum, transformers, and onnxruntime versions are compatible.
# - For ONNX summarization inference, always start decoder_input_ids with eos_token_id for BART or pad_token_id for T5.
# - Use optimum.onnxruntime pipelines for easy ONNX inference testing after export.
