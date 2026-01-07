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
import sys
import warnings

# Ensure the repo root is on sys.path so package imports work when running
# this file directly as a script: `python onnx_loaders/export_model.py`.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Suppress warnings during ONNX export
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message=".*torch.tensor results are registered as constants.*"
)

from onnx_loaders.export_model_consolidated import (
    export_and_optimize_onnx_unified as export_unified,
)

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
    parser.add_argument(
        "--finetine",
        action="store_true",
        help="Fine-tune T5 before export (only for T5 models)",
    )
    # Alias for typo: also accept --finetune and map to the same destination
    parser.add_argument(
        "--finetune", action="store_true", dest="finetine", help="Alias for --finetine"
    )
    parser.add_argument("--model_folder", help="HuggingFace model folder or path")
    parser.add_argument(
        "--onnx_path",
        help="Path to ONNX output directory (default: ../onnx or ONNX_PATH env var)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Framework to use for ONNX export (e.g., pt, tf).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-export even if ONNX files exist"
    )
    parser.add_argument(
        "--usev1",
        action="store_true",
        help="Use legacy exporter (v1). By default v2 exporter is used.",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=None,
        help="ONNX opset version to use for export (v2 supports this).",
    )
    parser.add_argument(
        "--pack_single_file",
        action="store_true",
        help="If exported ONNX uses external_data, repack into a single-file model.",
    )
    parser.add_argument(
        "--pack_single_threshold_mb",
        type=int,
        default=None,
        help=(
            "Size threshold in MB for single-file repack. If external weights exceed the"
            " threshold, repack is skipped. If omitted, exporter default (1536 MB) is used."
        ),
    )
    args = parser.parse_args()

    # Check for ONNX path: parameter > env variable > default
    onnx_path = args.onnx_path or os.getenv("ONNX_PATH", "onnx")
    print(f"Using ONNX path: {os.path.abspath(onnx_path)}")

    # Build unified kwargs and call the consolidated exporter
    unified_kwargs = dict(
        model_name=args.model_name,
        model_for=args.model_for,
        optimize=args.optimize,
        optimization_level=args.optimization_level,
        model_folder=args.model_folder,
        onnx_path=onnx_path,
        task=(args.task if args.model_for != "fe" else None),
        use_t5_encoder=args.use_t5_encoder,
        use_cache=args.use_cache,
        finetine=args.finetine,
        force=args.force,
        opset_version=(args.opset_version if hasattr(args, "opset_version") else None),
        use_v1=args.usev1,
        pack_single_file=args.pack_single_file,
        framework=args.framework,
        pack_single_threshold_mb=args.pack_single_threshold_mb,
    )

    if args.usev1:
        print("Using consolidated exporter in v1 compatibility mode")
    else:
        print("Using consolidated exporter (v2 behavior by default)")

    export_unified(**unified_kwargs)
