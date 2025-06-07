import gc
import glob
import os
import pathlib

import onnx
import onnxruntime as ort
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTOptimizer,
)
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import AutoTokenizer, T5EncoderModel


def export_and_optimize_onnx(
    model_name: str,
    optimize: bool = False,
    optimization_level: int = 1,
    task: str = "none",
    use_t5_encoder: bool = False,
    use_cache: bool = False,
):
    """
    Export and optionally optimize a HuggingFace model to ONNX format.
    HINTS:
    - For T5-based models (like sentence-t5-base), set use_t5_encoder=True to export only the encoder for embeddings.
    - For BERT/MiniLM/etc., use the default feature-extraction export.
    - The exported ONNX model will be verified for correct inputs/outputs.
    - Optimization is optional but recommended for production.
    """
    output_base = "../app/onnx/models/"
    onnx_name = "model.onnx"
    has_decoder = False
    decoder_onnx_name = "decoder_model.onnx"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    export_args = {}
    model_folder = model_name.split("/")[-1] if "/" in model_name else model_name

    def _get_model_class(task: str) -> type:
        """
        Get the model class based on the task and folder.
        """
        if task == "seq2seq-lm":
            return ORTModelForSeq2SeqLM
        elif task == "sequence-classification":
            return ORTModelForSequenceClassification
        else:
            return ORTModelForFeatureExtraction

    def _verify_model(model_path: str):
        """
        Verify the ONNX model by loading it and checking its inputs and outputs.
        For large models, skip onnx.checker.check_model to avoid MemoryError.
        """
        print(f"Verifying model at {model_path}...")
        onnx_model = onnx.load(model_path)
        try:
            onnx.checker.check_model(onnx_model)
        except MemoryError:
            print(
                "Warning: Skipping onnx.checker.check_model due to MemoryError (model is too large)."
            )
        session2 = ort.InferenceSession(model_path)
        print("Inputs:", session2.get_inputs())
        print("Outputs:", session2.get_outputs())
        del onnx_model
        del session2
        # Clear memory
        gc.collect()

    def _optimize_model(task: str, optimization_level: int):
        # Optimization
        if optimize:
            modelclass = _get_model_class(task)
            model_path = os.path.join(output_dir, onnx_name)
            optimization_config = OptimizationConfig(
                optimization_level=optimization_level
            )
            model = modelclass.from_pretrained(output_dir, model_path)
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.optimize(
                save_dir=pathlib.Path(model_path).parent,
                optimization_config=optimization_config,
            )
            print(f"Optimized model saved as: {model_path}")
            del model
            # Clear memory
            gc.collect()

            if has_decoder:
                # Verify model
                decoder_model_path = os.path.join(output_dir, decoder_onnx_name)
                print(
                    f"{decoder_model_path} exists: {pathlib.Path(decoder_model_path).exists()}"
                )
                _verify_model(decoder_model_path)
                decoder_model = modelclass.from_pretrained(model_name, **export_args)
                decoder_optimizer = ORTOptimizer.from_pretrained(decoder_model)
                optimized_decoder_model = decoder_optimizer.optimize(
                    save_dir=pathlib.Path(decoder_model_path).parent,
                    optimization_config=optimization_config,
                )
                print(
                    f"Decoder model optimized and saved as: {optimized_decoder_model.name}"
                )
                del decoder_model
                del decoder_optimizer
                del optimized_decoder_model
                # Clear memory
                gc.collect()

    # --- Export logic ---
    if task == "seq2seq-lm":
        export_args = {"export": True, "task": "seq2seq-lm", "use_cache": use_cache}
        onnx_name = "encoder_model.onnx"
        decoder_onnx_name = "decoder_model.onnx"
        # has_decoder = True  # Uncomment if you want to handle decoder export
    elif task == "sequence-classification":
        export_args = {
            "export": True,
            "task": "sequence-classification",
            "use_cache": use_cache,
        }
    else:
        task = "feature-extraction"
        if use_t5_encoder:
            print(
                f"Exporting model {model_name} for task {task} (T5 encoder-only export)..."
            )
            output_base = os.path.join(output_base, task)
            print(f"Output base directory: {output_base}")
            output_dir = os.path.join(BASE_DIR, output_base, model_folder)
            print(f"Output directory: {output_dir}")
            encoder = T5EncoderModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            os.makedirs(output_dir, exist_ok=True)
            # Clean up any old ONNX files
            for f in glob.glob(os.path.join(output_dir, "*.onnx")):
                os.remove(f)
            # Save encoder and tokenizer
            encoder.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            # Export encoder to ONNX using optimum
            model = ORTModelForFeatureExtraction.from_pretrained(
                output_dir, export=True
            )
            model.save_pretrained(output_dir)
            print(f"Encoder ONNX model exported to {output_dir} successfully.")
            del model
            del tokenizer
            del encoder
            # Clear memory
            gc.collect()
            # Verify model
            _verify_model(os.path.join(output_dir, onnx_name))
            _optimize_model(task, optimization_level)
            print(f"Model {model_name} exported successfully for task {task}.")
            return
        else:
            export_args = {"export": True}

    print(f"Exporting model {model_name} for task {task}...")
    output_base = os.path.join(output_base, task)
    print(f"Output base directory: {output_base}")
    output_dir = os.path.join(BASE_DIR, output_base, model_folder)
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    # export_args["save_dir"] = output_dir

    # Clean up any old ONNX files
    for f in glob.glob(os.path.join(output_dir, "*.onnx")):
        os.remove(f)

    # Export model to ONNX
    model = _get_model_class(task).from_pretrained(model_name, **export_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model exported to ONNX format in {output_dir}")

    # Model path
    model_path = os.path.join(output_dir, onnx_name)
    print(f"{model_path} exists: {pathlib.Path(model_path).exists()}")

    del model
    del tokenizer
    # Clear memory
    gc.collect()

    # Verify model
    _verify_model(model_path)

    # Optimize model if required
    _optimize_model(task, optimization_level)


# HINTS:
# - For T5-based models, always use use_t5_encoder=True for embeddings.
# - For BERT/MiniLM/etc., use the default export.
# - Check ONNX model inputs after export to ensure only encoder inputs are present for embeddings.
# - Optimization is optional but recommended for production.
