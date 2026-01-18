# =============================================================================
# File: test_model_availability_flags.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest  # noqa: F401

from app.services.base_nlp_service import BaseNLPService


def test_encoder_flag_set_true(tmp_path):
    # Create model files under the configured root so validate_safe_path accepts them
    base_root = BaseNLPService.get_root_path()
    model_dir = os.path.join(base_root, "models", "model_true")
    os.makedirs(model_dir, exist_ok=True)
    # Create a default encoder filename fallback 'model.onnx'
    with open(os.path.join(model_dir, "model.onnx"), "wb") as f:
        f.write(b"dummy")

    cfg = SimpleNamespace()
    cfg.tasks = ["embedding"]
    cfg.model_folder_name = "model_true"

    # Initially no flag in metadata
    assert BaseNLPService._get_model_metadata("model_true") is None

    res = BaseNLPService._validate_model_availability(
        "model_true",
        task="embedding",
        perform_filesystem_check=True,
        cfg=cfg,
        model_path=model_dir,
    )

    assert res is True
    md = BaseNLPService._get_model_metadata("model_true") or {}
    assert md.get("encoder_model_exists") == "true"


def test_encoder_flag_set_false(tmp_path):
    base_root = BaseNLPService.get_root_path()
    model_dir = os.path.join(base_root, "models", "model_false")
    os.makedirs(model_dir, exist_ok=True)

    cfg = SimpleNamespace()
    cfg.tasks = ["embedding"]
    cfg.model_folder_name = "model_false"

    res = BaseNLPService._validate_model_availability(
        "model_false",
        task="embedding",
        perform_filesystem_check=True,
        cfg=cfg,
        model_path=model_dir,
    )

    assert res is False
    md = BaseNLPService._get_model_metadata("model_false") or {}
    assert md.get("encoder_model_exists") == "false"


def test_decoder_flag_set_true(tmp_path):
    base_root = BaseNLPService.get_root_path()
    model_dir = os.path.join(base_root, "models", "model_seq")
    os.makedirs(model_dir, exist_ok=True)
    # Create encoder and decoder files
    with open(os.path.join(model_dir, "model.onnx"), "wb") as f:
        f.write(b"enc")
    with open(os.path.join(model_dir, "decoder_model.onnx"), "wb") as f:
        f.write(b"dec")

    cfg = SimpleNamespace()
    cfg.tasks = ["prompt"]
    cfg.model_folder_name = "model_seq"
    cfg.use_seq2seqlm = True

    res = BaseNLPService._validate_model_availability(
        "model_seq",
        task="prompt",
        perform_filesystem_check=True,
        cfg=cfg,
        model_path=model_dir,
    )

    assert res is True
    md = BaseNLPService._get_model_metadata("model_seq") or {}
    assert md.get("encoder_model_exists") == "true"
    assert md.get("decoder_model_exists") == "true"


def test_decoder_flag_set_false_when_missing(tmp_path):
    base_root = BaseNLPService.get_root_path()
    model_dir = os.path.join(base_root, "models", "model_seq2")
    os.makedirs(model_dir, exist_ok=True)
    # Only encoder present
    with open(os.path.join(model_dir, "model.onnx"), "wb") as f:
        f.write(b"enc")

    cfg = SimpleNamespace()
    cfg.tasks = ["prompt"]
    cfg.model_folder_name = "model_seq2"
    cfg.use_seq2seqlm = True

    res = BaseNLPService._validate_model_availability(
        "model_seq2",
        task="prompt",
        perform_filesystem_check=True,
        cfg=cfg,
        model_path=model_dir,
    )

    assert res is False
    md = BaseNLPService._get_model_metadata("model_seq2") or {}
    assert md.get("encoder_model_exists") == "true"
    assert md.get("decoder_model_exists") == "false"


def test_shortcut_honors_cached_true_and_skips_filesystem():
    # If flags are cached true, the shortcut should return True and must not call filesystem
    cfg = SimpleNamespace()
    cfg.tasks = ["embedding"]
    cfg.model_folder_name = "cached_true_model"
    # Seed metadata cache to simulate a cached positive flag
    BaseNLPService._set_model_metadata("cached_true_model", {"encoder_model_exists": "true"})

    # Patch file_exists_in_model to fail if called
    with patch("app.services.base_nlp_service.BaseNLPService.file_exists_in_model") as mock_check:
        mock_check.side_effect = AssertionError(
            "Filesystem check should not be called when perform_filesystem_check=False"
        )

        res = BaseNLPService._validate_model_availability(
            "cached_true_model",
            task="embedding",
            perform_filesystem_check=False,
            cfg=cfg,
            model_path="/does/not/matter",
        )

        assert res is True


def test_shortcut_honors_cached_false_and_skips_filesystem():
    # If flags are cached false, the shortcut should return False immediately
    cfg = SimpleNamespace()
    cfg.tasks = ["embedding"]
    cfg.model_folder_name = "cached_false_model"
    # Seed metadata cache to simulate a cached negative flag
    BaseNLPService._set_model_metadata("cached_false_model", {"encoder_model_exists": "false"})

    with patch("app.services.base_nlp_service.BaseNLPService.file_exists_in_model") as mock_check:
        mock_check.side_effect = AssertionError(
            "Filesystem check should not be called when perform_filesystem_check=False"
        )

        res = BaseNLPService._validate_model_availability(
            "cached_false_model",
            task="embedding",
            perform_filesystem_check=False,
            cfg=cfg,
            model_path="/does/not/matter",
        )

        assert res is False


def test_shortcut_allows_not_checked_without_filesystem_call():
    # If flags are not_checked, shortcut should return True and not call filesystem
    cfg = SimpleNamespace()
    cfg.tasks = ["prompt"]
    cfg.model_folder_name = "not_checked_model"
    # Seed metadata cache: encoder=true, decoder=not_checked
    BaseNLPService._set_model_metadata(
        "not_checked_model",
        {"encoder_model_exists": "true", "decoder_model_exists": "not_checked"},
    )
    cfg.use_seq2seqlm = True

    with patch("app.services.base_nlp_service.BaseNLPService.file_exists_in_model") as mock_check:
        mock_check.side_effect = AssertionError(
            "Filesystem check should not be called when perform_filesystem_check=False"
        )

        res = BaseNLPService._validate_model_availability(
            "not_checked_model",
            task="prompt",
            perform_filesystem_check=False,
            cfg=cfg,
            model_path="/does/not/matter",
        )

        # Since decoder is not cached false, method returns True (caller defers filesystem checks)
        assert res is True
