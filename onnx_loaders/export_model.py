import argparse

from export_model_to_onnx import export_and_optimize_onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and optimize ONNX model.")
    parser.add_argument(
        "--model_name", required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--model_folder", required=True, help="Folder name for saving the ONNX model"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Whether to optimize the ONNX model"
    )
    parser.add_argument(
        "--optimization_level",
        type=int,
        default=1,
        help="ONNX optimization level (default: 1)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="none",
        help="Export task (e.g., seq2seq-lm, sequence-classification, feature-extraction)",
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Whether to use cached model"
    )
    args = parser.parse_args()

    export_and_optimize_onnx(
        model_name=args.model_name,
        model_folder=args.model_folder,
        optimize=args.optimize,
        optimization_level=args.optimization_level,
        task=args.task,
        use_cache=args.use_cache,
    )
