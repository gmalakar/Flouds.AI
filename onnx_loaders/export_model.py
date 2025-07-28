# =============================================================================
# File: export_model.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# HINTS:
# - For summarization (BART, T5, Pegasus, etc.), use --task seq2seq-lm and --use_cache.
# - For sequence classification (BERT, RoBERTa, etc.), use --task sequence-classification.
# - For embeddings/feature extraction, use --task feature-extraction or --use_t5_encoder for T5 encoder-only export.
# - After exporting a seq2seq model, you should see encoder_model.onnx, decoder_model.onnx, and decoder_with_past_model.onnx in the output directory.
# - If decoder_with_past_model.onnx is missing, the model cannot be used for fast autoregressive generation (greedy decoding).
# - Always verify ONNX model inputs/outputs after export to ensure compatibility with your inference pipeline.
# - Optimization is optional but recommended for production; it can reduce inference latency.
# - If you encounter export errors, check that your optimum/transformers/onnxruntime versions are compatible.
# - For ONNX summarization inference, always start decoder_input_ids with eos_token_id for BART or pad_token_id for T5.
# - Use optimum.onnxruntime pipelines for easy ONNX inference testing after export.

import argparse
import os

from export_model_to_onnx import export_and_optimize_onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and optimize ONNX model.")
    parser.add_argument(
        "--model_name", required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--model_for",
        type=str,
        default="fe",
        help="Model purpose: s2s (seq2seq-lm), sc (sequence-classification), fe (feature-extraction)",
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
        default=None,
        help="Export task (e.g., seq2seq-lm, sequence-classification, feature-extraction)",
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Whether to use cache for seq2seq-lm"
    )
    parser.add_argument(
        "--use_t5_encoder", action="store_true", help="Whether to use T5 encoder"
    )
    parser.add_argument("--model_folder", help="HuggingFace model folder or path")
    parser.add_argument(
        "--onnx_path",
        help="Path to ONNX output directory (default: ../onnx or ONNX_PATH env var)",
    )
    args = parser.parse_args()

    # Check for ONNX path: parameter > env variable > default
    onnx_path = args.onnx_path or os.getenv("ONNX_PATH", "../onnx")
    print(f"Using ONNX path: {os.path.abspath(onnx_path)}")

    export_and_optimize_onnx(
        model_name=args.model_name,
        model_for=args.model_for,
        optimize=args.optimize,
        optimization_level=args.optimization_level,
        task=args.task,
        use_t5_encoder=args.use_t5_encoder,
        use_cache=args.use_cache,
        model_folder=args.model_folder,
        onnx_path=onnx_path,
    )
