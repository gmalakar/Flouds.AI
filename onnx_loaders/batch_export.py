#!/usr/bin/env python3
# =============================================================================
# File: batch_export.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Batch export multiple models to ONNX format."""

import subprocess
import sys
from pathlib import Path

# Model configurations: (model_for, model_name, optimize, task, use_cache, use_t5_encoder, model_folder)
MODELS = [
    # Embedding models
    ("fe", "sentence-transformers/all-MiniLM-L6-v2", True, None, False, False, None),
    (
        "fe",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        True,
        None,
        False,
        False,
        None,
    ),
    ("fe", "sentence-transformers/sentence-t5-base", True, None, False, True, None),
    # Summarization models
    ("s2s", "t5-small", True, "seq2seq-lm", True, False, None),
    (
        "s2s",
        "Falconsai/text_summarization",
        True,
        "seq2seq-lm",
        True,
        False,
        "falconsai_text_summarization",
    ),
    ("s2s", "facebook/bart-large-cnn", True, "seq2seq-lm", True, False, None),
]


def export_model(
    model_for, model_name, optimize, task, use_cache, use_t5_encoder, model_folder
):
    """Export a single model to ONNX format."""
    cmd = [
        "python",
        "onnx_loaders/export_model.py",
        "--model_for",
        model_for,
        "--model_name",
        model_name,
    ]

    if optimize:
        cmd.append("--optimize")
    if task:
        cmd.extend(["--task", task])
    if use_cache:
        cmd.append("--use_cache")
    if use_t5_encoder:
        cmd.append("--use_t5_encoder")
    if model_folder:
        cmd.extend(["--model_folder", model_folder])

    print(f"🔄 Exporting {model_name}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )  # 30 min timeout

        if result.returncode == 0:
            print(f"✅ {model_name} exported successfully")
            return True
        else:
            print(f"❌ {model_name} failed:")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"💥 {model_name} crashed: {e}")
        return False


def main():
    """Export all models in batch."""
    print("🚀 Starting batch ONNX model export...")
    print(f"📦 Total models to export: {len(MODELS)}")
    print("=" * 60)

    success_count = 0
    failed_models = []

    for i, (
        model_for,
        model_name,
        optimize,
        task,
        use_cache,
        use_t5_encoder,
        model_folder,
    ) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Processing {model_name}")

        success = export_model(
            model_for,
            model_name,
            optimize,
            task,
            use_cache,
            use_t5_encoder,
            model_folder,
        )

        if success:
            success_count += 1
        else:
            failed_models.append(model_name)

        print("-" * 60)

    # Summary
    print(f"\n📊 Export Summary:")
    print(f"✅ Successful: {success_count}/{len(MODELS)}")
    print(f"❌ Failed: {len(failed_models)}/{len(MODELS)}")

    if failed_models:
        print(f"\n💥 Failed models:")
        for model in failed_models:
            print(f"  - {model}")
        sys.exit(1)
    else:
        print(f"\n🎉 All models exported successfully!")


if __name__ == "__main__":
    main()
