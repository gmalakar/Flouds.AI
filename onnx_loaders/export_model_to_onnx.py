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
from transformers import AutoTokenizer


# Export and optimize a model to ONNX format
def export_and_optimize_onnx(
    model_name: str,
    model_folder: str,
    optimize: bool = False,
    optimization_level: int = 1,
    task: str = "none",
    use_cache: bool = False,
):
    output_base = "../app/onnx/models/"
    onnx_name = "model.onnx"
    has_decoder = False
    decoder_onnx_name = "decoder_model.onnx"

    if task == "seq2seq-lm":
        ModelClass = ORTModelForSeq2SeqLM
        export_args = {"export": True, "task": "seq2seq-lm", "use_cache": use_cache}
        onnx_name = "encoder_model.onnx"
        # has_decoder = True
        decoder_onnx_name = "decoder_model.onnx"
    elif task == "sequence-classification":
        ModelClass = ORTModelForSequenceClassification
        export_args = {
            "export": True,
            "task": "sequence-classification",
            "use_cache": use_cache,
        }
    else:
        ModelClass = ORTModelForFeatureExtraction
        export_args = {"export": True}
        task = "feature-extraction"

    print(f"Exporting model {model_name} for task {task}...")
    output_base = os.path.join(output_base, task)
    print(f"Output base directory: {output_base}")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, output_base, model_folder)
    print(f"Output directory: {output_dir}")

    # Export model to ONNX
    model = ModelClass.from_pretrained(model_name, **export_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model exported to ONNX format in {output_dir}")

    # Model path
    model_path = os.path.join(output_dir, onnx_name)
    print(f"{model_path} exists: {pathlib.Path(model_path).exists()}")

    # Verify model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    print("Inputs:", session.get_inputs())
    print("Outputs:", session.get_outputs())

    # Optimization
    if optimize:
        optimization_config = OptimizationConfig(optimization_level=optimization_level)
        model = ModelClass.from_pretrained(output_dir, model_path)
        optimizer = ORTOptimizer.from_pretrained(model)
        optimized_model = optimizer.optimize(
            save_dir=pathlib.Path(model_path).parent,
            optimization_config=optimization_config,
        )
        print(f"Optimized model saved as: {model_path}")

        if has_decoder:
            # Verify model
            decoder_model_path = os.path.join(output_dir, decoder_onnx_name)
            print(
                f"{decoder_model_path} exists: {pathlib.Path(decoder_model_path).exists()}"
            )

            # Load the ONNX model
            session2 = ort.InferenceSession(decoder_model_path)
            print("Inputs:", session2.get_inputs())
            print("Outputs:", session2.get_outputs())

            decoder_onnx_model = onnx.load(decoder_model_path)
            onnx.checker.check_model(decoder_onnx_model)
            decoder_model = ModelClass.from_pretrained(model_name, **export_args)
            decoder_optimizer = ORTOptimizer.from_pretrained(decoder_model)
            optimized_decoder_model = decoder_optimizer.optimize(
                save_dir=pathlib.Path(decoder_model_path).parent,
                optimization_config=optimization_config,
            )
            print(
                f"Decoder model optimized and saved as: {optimized_decoder_model.name}"
            )
