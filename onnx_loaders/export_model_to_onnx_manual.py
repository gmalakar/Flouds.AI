# =============================================================================
# File: export_and_optimize_onnx_manual.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import gc
import glob
import os
import pathlib
import traceback
import warnings

try:
    from torch.onnx._internal.registration import OnnxExporterWarning

    warnings.filterwarnings("ignore", category=OnnxExporterWarning)
except ImportError:
    pass

from pathlib import Path

from onnxruntime import InferenceSession
from optimum.exporters.onnx import export, main_export
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTOptimizer,
)
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from onnx import checker, load

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Symbolic function.*already registered.*")
warnings.filterwarnings(
    "ignore", message=".*were not initialized from the model checkpoint.*"
)
warnings.filterwarnings("ignore", message=".*are newly initialized.*")


def export_and_optimize_onnx_manual(
    model_name: str,
    model_for: str = "fe",
    optimize: bool = False,
    optimization_level: int = 1,
    task: str = None,
    model_folder: str = None,
    onnx_path: str = "../onnx",
    opset_version: int = 14,
):
    """
    Export and optionally optimize a HuggingFace model to ONNX format.
    """

    # Set environment variables to handle large models and protobuf limits
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PYTHONHASHSEED"] = "0"

    # Increase protobuf message size limits - handle different protobuf versions
    try:
        import google.protobuf.message

        # Try the newer method first
        if hasattr(google.protobuf.message.Message, "_SetGlobalDefaultMaxMessageSize"):
            google.protobuf.message.Message._SetGlobalDefaultMaxMessageSize(2**31 - 1)
            print(
                "✅ Protobuf message size limit set using _SetGlobalDefaultMaxMessageSize"
            )
        elif hasattr(google.protobuf.message.Message, "SetGlobalDefaultMaxMessageSize"):
            google.protobuf.message.Message.SetGlobalDefaultMaxMessageSize(2**31 - 1)
            print(
                "✅ Protobuf message size limit set using SetGlobalDefaultMaxMessageSize"
            )
        else:
            # Set environment variable as fallback
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
            print("⚠️ Using environment variable fallback for protobuf configuration")
    except Exception as e:
        print(f"⚠️ Could not configure protobuf message size: {e}")
        # Set additional environment variables that might help
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
        print("⚠️ Using environment variable fallback")

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

    print(
        f"Starting export for {model_name} (model_for={model_for}, optimize={optimize})"
    )
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _model_for = model_for.lower()

    # Validate onnx_path to prevent path traversal
    onnx_path = os.path.normpath(onnx_path)
    if ".." in onnx_path:
        raise ValueError("Path traversal detected in onnx_path")
    _output_base = os.path.join(onnx_path, "models")

    _onnx_name = "model.onnx"
    _decoder_onnx_name = "decoder_model.onnx"
    _encoder_onnx_name = "encoder_model.onnx"
    _model_path = None

    if not model_folder:
        model_folder = model_name.split("/")[-1] if "/" in model_name else model_name

    # Validate model_folder to prevent path traversal
    model_folder = os.path.basename(model_folder)
    if ".." in model_folder or "/" in model_folder or "\\" in model_folder:
        raise ValueError("Invalid model_folder name")

    _output_base = os.path.join(_output_base, _model_for)
    _output_dir = os.path.join(BASE_DIR, _output_base, model_folder)

    def _verify_model(model_names: list, output_dir: str):
        """Verify ONNX models are valid and can be loaded."""
        print(f"Starting verification of {len(model_names)} model(s)...")
        for fname in model_names:
            model_path = os.path.join(output_dir, fname)
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                continue

            print(f"Verifying model at {model_path}...")
            try:
                onnx_model = load(model_path)
                try:
                    checker.check_model(onnx_model)
                    print(f"✅ {fname} validation passed")
                except MemoryError:
                    print(
                        f"⚠️ Skipping onnx.checker.check_model for {fname} due to MemoryError"
                    )

                session2 = InferenceSession(model_path)
                print(f"{fname} inputs:", [inp.name for inp in session2.get_inputs()])
                print(f"{fname} outputs:", [out.name for out in session2.get_outputs()])

                del onnx_model, session2
                gc.collect()

            except Exception as e:
                print(f"❌ {fname} verification failed: {e}")
                try:
                    os.remove(model_path)
                    print(f"Removed corrupted file: {model_path}")
                except:
                    pass
                return False
        return True

    def _export_with_optimum_ort(model, tokenizer, output_path, model_for):
        """Alternative export method using optimum's ORTModel classes directly."""
        print("Using optimum ORTModel for alternative export...")

        # Disable scaled_dot_product_attention for ONNX export compatibility
        try:
            import torch

            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            print("✅ Torch backends configured for ONNX compatibility")
        except ImportError:
            print("⚠️ PyTorch not available, skipping backend configuration")

        # Save model and tokenizer first
        print("Saving model and tokenizer...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Clear memory before ONNX conversion
        gc.collect()

        # Configure export args
        export_args = {"export": True}
        print(f"Export args: {export_args}")

        # Use ORTModel to convert to ONNX
        print(f"Converting to ONNX with model_for={model_for}...")
        try:
            if model_for == "s2s" or model_for == "seq2seq-lm":
                print("Using ORTModelForSeq2SeqLM...")
                ort_model = ORTModelForSeq2SeqLM.from_pretrained(
                    output_path, **export_args
                )
            elif model_for == "sc" or model_for == "sequence-classification":
                print("Using ORTModelForSequenceClassification...")
                ort_model = ORTModelForSequenceClassification.from_pretrained(
                    output_path, **export_args
                )
            else:  # feature-extraction
                print("Using ORTModelForFeatureExtraction...")
                ort_model = ORTModelForFeatureExtraction.from_pretrained(
                    output_path, **export_args
                )

            # Save the ONNX model
            print("Saving ONNX model...")
            ort_model.save_pretrained(output_path)
            print("✅ Alternative export successful")

            # Clean up
            del ort_model
            gc.collect()

        except Exception as e:
            print(f"❌ Alternative export failed: {e}")
            raise e

    def _export_model_to_onnx(
        model_for: str,
        model_name: str,
        output_dir: str,
        task: str = "auto",
        device: str = "cpu",
        opset: int = 14,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_names = []
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try to detect the actual model type first
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.model_type
            print(f"Detected model type: {model_type}")
            print(f"Model config class: {config.__class__.__name__}")
        except Exception as e:
            print(f"Could not detect model type: {e}")
            model_type = None

        # Load the appropriate model based on model_for, with fallback handling
        try:
            if model_for == "s2s" or model_for == "seq2seq-lm":
                print(f"Attempting to load as seq2seq model...")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model_names = [_encoder_onnx_name, _decoder_onnx_name]
                model_path = os.path.join(_output_dir, _encoder_onnx_name)
            elif model_for == "sc" or model_for == "sequence-classification":
                print(f"Attempting to load as sequence classification model...")
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model_names = [_onnx_name]
                model_path = os.path.join(_output_dir, _onnx_name)
            elif model_for == "fe" or model_for == "feature-extraction":
                print(f"Attempting to load as feature extraction model...")
                model = AutoModel.from_pretrained(model_name)
                model_names = [_onnx_name]
                model_path = os.path.join(_output_dir, _onnx_name)
            else:
                raise ValueError(f"Invalid model_for: {model_for}")
        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                print(f"❌ Model type mismatch: {e}")
                print(
                    f"💡 Model '{model_name}' appears to be a {model_type} model, not suitable for {model_for}"
                )
                print("💡 Suggestions:")
                if model_type in ["llama", "mistral", "qwen", "phi", "gemma"]:
                    print("   - Try using --model_for 'fe' for feature extraction")
                    print(
                        "   - This appears to be a causal language model, not a seq2seq model"
                    )
                elif model_for == "s2s":
                    print("   - Use models like BART, T5, Pegasus for seq2seq tasks")
                    print("   - Or try --model_for 'fe' if you want embeddings")
                raise ValueError(
                    f"Model type {model_type} is not compatible with model_for='{model_for}'"
                )
            else:
                raise e

        print(f"Loading model {model_name} for {model_for} and task {task}")

        # Save tokenizer and config
        tokenizer.save_pretrained(output_path)
        model.config.save_pretrained(output_path)
        if hasattr(model, "generation_config"):
            if model.generation_config is not None:
                model.generation_config.save_pretrained(output_path)
            else:
                # Get the generation_config attribute and save it manually
                generation_config = getattr(model, "generation_config", None)
                if generation_config is not None:
                    import json

                    config_path = output_path / "generation_config.json"
                    with open(config_path, "w") as f:
                        json.dump(generation_config.to_dict(), f, indent=2)

        # Force memory cleanup before export
        gc.collect()
        if hasattr(model, "cpu"):
            model = model.cpu()  # Move to CPU to free GPU memory
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Export encoder.onnx directly to output_path
        try:
            # Try with different export strategies to handle serialization issues
            print(f"Attempting ONNX export with opset {opset}...")

            # First attempt: Standard export
            main_export(
                model_name_or_path=model_name,
                output=output_path,
                task=task,
                opset=opset,
                device=device,
            )
        except Exception as e:
            error_msg = str(e).lower()
            print(f"❌ Primary export failed: {e}")

            if "failed to serialize proto" in error_msg or "encodeerror" in error_msg:
                print("🔄 Protobuf serialization error detected. Trying workarounds...")

                # Workaround 1: Try with lower opset version
                if opset > 11:
                    print(f"🔄 Retrying with lower opset version (11)...")
                    try:
                        main_export(
                            model_name_or_path=model_name,
                            output=output_path,
                            task=task,
                            opset=11,
                            device=device,
                        )
                        print("✅ Export successful with opset 11")
                    except Exception as e2:
                        print(f"❌ Export with opset 11 also failed: {e2}")

                        # Workaround 2: Try manual export with optimum's ORTModel
                        print("🔄 Trying alternative export method...")
                        try:
                            _export_with_optimum_ort(
                                model, tokenizer, output_path, model_for
                            )
                            print("✅ Alternative export method successful")
                        except Exception as e3:
                            print(f"❌ Alternative export also failed: {e3}")
                            print("Stack trace:")
                            traceback.print_exc()
                else:
                    print("Stack trace:")
                    traceback.print_exc()
            else:
                print("Stack trace:")
                traceback.print_exc()
            # Continue execution to allow verification of any partial files

        print(f"Model exported to ONNX format in {_output_dir}")
        for fname in model_names:
            fpath = os.path.join(_output_dir, fname)
            print(f"{fpath} exists: {os.path.exists(fpath)}")

        print(f"Exported encoder and decoder to {output_path}")

        # Clean up model from memory after export
        del model, tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return model_path, model_names

    def _optimize_model():
        if optimize:
            print("Starting optimization process...")

            # Delete existing optimized files
            for file in os.listdir(_output_dir):
                if file.endswith("_optimized.onnx"):
                    opt_file_path = os.path.join(_output_dir, file)
                    try:
                        os.remove(opt_file_path)
                        print(f"🗑️ Removed existing optimized file: {file}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {file}: {e}")

            # Check available memory before optimization
            try:
                import psutil

                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
                print(f"Available memory: {available_memory_mb:.1f} MB")

                # Skip optimization if very low memory
                if available_memory_mb < 1000:  # Less than 1GB available
                    print(
                        f"⚠️ Insufficient memory ({available_memory_mb:.1f} MB) for optimization. Skipping optimization."
                    )
                    print("📦 Using unoptimized model (still functional)")
                    return
            except ImportError:
                print("⚠️ psutil not available, cannot check memory")

            # Aggressive memory cleanup before optimization
            print("Performing memory cleanup before optimization...")
            gc.collect()
            gc.collect()  # Run twice

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("✅ CUDA cache cleared")
            except ImportError:
                pass

            try:
                optimization_config = OptimizationConfig(
                    optimization_level=optimization_level
                )
                print(f"Attempting optimization with level {optimization_level}...")

                # Load model for optimization
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

                print("Model loaded for optimization...")
                optimizer = ORTOptimizer.from_pretrained(model)
                print("Optimizer created, starting optimization...")

                optimizer.optimize(
                    save_dir=pathlib.Path(_model_path).parent,
                    optimization_config=optimization_config,
                )
                # Print all optimized files
                optimized_files = [
                    f for f in os.listdir(_output_dir) if f.endswith("_optimized.onnx")
                ]
                for opt_file in optimized_files:
                    print(
                        f"✅ Optimized model saved as: {os.path.join(_output_dir, opt_file)}"
                    )
                del model, optimizer
                gc.collect()

            except Exception as e:
                error_msg = str(e).lower()
                print(f"❌ Optimization failed: {e}")

                if "bad allocation" in error_msg or "memory" in error_msg:
                    print("💡 Memory allocation error detected. This usually means:")
                    print("   - The model is too large for available memory")
                    print("   - Try closing other applications to free memory")
                    print("   - Consider using a machine with more RAM")
                    print("   - The unoptimized model will still work for inference")

                elif "not available yet" in error_msg or "mpnet" in error_msg:
                    print(f"ℹ️ Optimization not supported for this model type")

                elif "error parsing message" in error_msg or "modelproto" in error_msg:
                    print(f"❌ Optimization failed due to corrupted ONNX model")
                    print("📦 Model may be corrupted, but export might have succeeded")

                else:
                    # Try with lower optimization level as fallback
                    if optimization_level > 0:
                        print(f"🔄 Retrying with basic optimization (level 0)...")
                        try:
                            basic_config = OptimizationConfig(optimization_level=0)

                            # Re-load model for basic optimization
                            if _model_for in ["s2s", "seq2seq-lm"]:
                                model = ORTModelForSeq2SeqLM.from_pretrained(
                                    _output_dir, use_cache=False
                                )
                            elif _model_for in ["sc", "sequence-classification"]:
                                model = (
                                    ORTModelForSequenceClassification.from_pretrained(
                                        _output_dir
                                    )
                                )
                            else:
                                model = ORTModelForFeatureExtraction.from_pretrained(
                                    _output_dir
                                )

                            optimizer = ORTOptimizer.from_pretrained(model)
                            optimizer.optimize(
                                save_dir=pathlib.Path(_model_path).parent,
                                optimization_config=basic_config,
                            )
                            # Print all optimized files
                            optimized_files = [
                                f
                                for f in os.listdir(_output_dir)
                                if f.endswith("_optimized.onnx")
                            ]
                            for opt_file in optimized_files:
                                print(
                                    f"✅ Basic optimization successful: {os.path.join(_output_dir, opt_file)}"
                                )
                            del model, optimizer
                            gc.collect()

                        except Exception as e2:
                            print(f"❌ Basic optimization also failed: {e2}")
                            print("📦 Using unoptimized model (still functional)")
                    else:
                        print("📦 Using unoptimized model (still functional)")

                # Final cleanup
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    # --- Export logic ---
    _model_path, _model_names = _export_model_to_onnx(
        model_for=_model_for,
        model_name=model_name,
        output_dir=_output_dir,
        task=task,
        opset=opset_version,
    )

    print(f"Model path: {_model_path}")

    # Verify models
    print("Starting model verification...")
    _verify_model(_model_names, _output_dir)

    # Optimize model if requested
    _optimize_model()

    gc.collect()
