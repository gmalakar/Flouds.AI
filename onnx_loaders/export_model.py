# =============================================================================
# File: export_model.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# HINTS:
# - For summarization (BART, T5, Pegasus, etc.), use `--task seq2seq-lm`.
#   KV-cache exports are auto-detected; supply `--task text2text-generation-with-past` to request KV-cache behavior.
# - For sequence classification (BERT, RoBERTa, etc.), use --task sequence-classification.
# - For embeddings/feature extraction, use --task feature-extraction
# - After exporting a seq2seq model, you should see encoder_model.onnx, decoder_model.onnx, and decoder_with_past_model.onnx in the output directory.
# - If decoder_with_past_model.onnx is missing, the model cannot be used for fast autoregressive generation (greedy decoding).
# - Always verify ONNX model inputs/outputs after export to ensure compatibility with your inference pipeline.
# - Optimization is optional but recommended for production; it can reduce inference latency.
# - If you encounter export errors, check that your optimum/transformers/onnxruntime versions are compatible.
# - For ONNX summarization inference, always start decoder_input_ids with eos_token_id for BART or pad_token_id for T5.
# - Use optimum.onnxruntime pipelines for easy ONNX inference testing after export.

import argparse
import inspect
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
        "--model-name",
        dest="model_name",
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--model-for",
        dest="model_for",
        type=str,
        default="fe",
        help="Model purpose: s2s (seq2seq-lm), sc (sequence-classification), fe (feature-extraction)",
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Whether to optimize the ONNX model"
    )
    parser.add_argument(
        "--optimization-level",
        dest="optimization_level",
        type=int,
        default=99,
        help="ONNX optimization level (default: 1)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Export task (e.g., seq2seq-lm, sequence-classification, feature-extraction) (required)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune T5 before export (only for T5 models)",
    )
    parser.add_argument(
        "--model-folder", dest="model_folder", help="HuggingFace model folder or path"
    )
    parser.add_argument(
        "--onnx-path",
        dest="onnx_path",
        help="Path to ONNX output directory (default: ../onnx or ONNX_PATH env var)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Framework to use for ONNX export (e.g., pt, tf).",
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow executing custom code from model repos that require it (use with caution)",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Request the validator to L2-normalize sentence embeddings before comparison",
    )
    parser.add_argument(
        "--require-validator",
        action="store_true",
        help="Require the consolidated validator to pass; fail export if validation fails.",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip numeric ONNX validation (do not run validate_onnx_model).",
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
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (default: cuda). Use 'cpu' to force CPU export.",
    )
    parser.add_argument(
        "--pack-single-file",
        dest="pack_single_file",
        action="store_true",
        help="If exported ONNX uses external_data, repack into a single-file model.",
    )
    parser.add_argument(
        "--no-use-external-data-format",
        dest="use_external_data_format",
        action="store_false",
        default=True,
        help="Disable external data format; prefer single-file ONNX when possible.",
    )
    parser.add_argument(
        "--pack-single-threshold-mb",
        dest="pack_single_threshold_mb",
        type=int,
        default=None,
        help=(
            "Size threshold in MB for single-file repack. If external weights exceed the"
            " threshold, repack is skipped. If omitted, exporter default (1536 MB) is used."
        ),
    )
    parser.add_argument(
        "--no-local-prep",
        action="store_true",
        help="Skip creating a prepared local copy (temp_local) for LLMs before export",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        type=str,
        default=None,
        help="HuggingFace access token (synonym for HUGGINGFACE_HUB_TOKEN/HF_TOKEN)",
    )
    parser.add_argument(
        "--library",
        type=str,
        default=None,
        help="Export library hint (e.g., 'sentence_transformers' to use sentence-transformers exporter)",
    )
    parser.add_argument(
        "--merge",
        dest="merge",
        action="store_true",
        help=(
            "Request model merging where applicable. Merge is only applicable to "
            "decoder-only causal LLMs that support text-generation-with-past (KV-cache). "
            "No merging is performed after optimization; the exporter will apply merging "
            "at the appropriate stage. Use this flag to request a merged decoder artifact."
        ),
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="When set, remove extraneous ONNX files after optimization following prioritized cleanup rules",
    )
    parser.add_argument(
        "--prune-canonical",
        dest="prune_canonical",
        action="store_true",
        help="When set, remove canonical ONNX files (e.g., decoder_model.onnx) if merged artifacts exist",
    )
    parser.add_argument(
        "--no-post-process",
        dest="no_post_process",
        action="store_true",
        help="Skip optimum post-processing steps (deduplication). Useful to avoid MemoryError during large-model post-processing",
    )
    parser.add_argument(
        "--portable",
        dest="portable",
        action="store_true",
        help="Prefer conservative/portable ONNX optimizations (avoid hardware-specific passes)",
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
        portable=args.portable,
        model_folder=args.model_folder,
        onnx_path=onnx_path,
        task=args.task,
        finetune=args.finetune,
        force=args.force,
        opset_version=(args.opset_version if hasattr(args, "opset_version") else None),
        use_v1=args.usev1,
        pack_single_file=args.pack_single_file,
        use_external_data_format=args.use_external_data_format,
        framework=args.framework,
        pack_single_threshold_mb=args.pack_single_threshold_mb,
        require_validator=args.require_validator,
        trust_remote_code=args.trust_remote_code,
        normalize_embeddings=args.normalize_embeddings,
        skip_validator=args.skip_validator,
        device=args.device,
        hf_token=(args.hf_token),
        library=args.library,
        merge=args.merge,
        cleanup=args.cleanup,
        prune_canonical=args.prune_canonical,
        no_post_process=args.no_post_process,
        no_local_prep=args.no_local_prep,
    )

    # Validate unified_kwargs keys against the exported function signature.
    # If export_unified does not accept **kwargs, reject unknown parameter names
    # to give clear feedback (e.g., typos or removed underscore-style aliases).
    try:
        sig = inspect.signature(export_unified)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not accepts_kwargs:
            allowed = [
                name
                for name, p in params.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            invalid = [k for k in unified_kwargs.keys() if k not in allowed]
            if invalid:
                reason = (
                    "Possible typos or use of removed underscore-style aliases. "
                    "Check flag names and use hyphenated forms."
                )
                parser.error(
                    f"Invalid parameter name(s) passed to exporter: {', '.join(invalid)}. "
                    f"Allowed parameters: {', '.join(allowed)}. {reason}"
                )
    except Exception:
        # Be conservative: if introspection fails for any reason, do not block export.
        pass

    if args.usev1:
        print("Using consolidated exporter in v1 compatibility mode")
    else:
        print("Using consolidated exporter (v2 behavior by default)")

    export_unified(**unified_kwargs)
