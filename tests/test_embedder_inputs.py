# =============================================================================
# File: test_embedder_inputs.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from types import SimpleNamespace

import numpy as np

from app.services.embedder import SentenceTransformer


def test_prepare_onnx_inputs_batch_and_dtype():
    # Tokenizer stub: returns 1D arrays for a single text (no batch dim)
    def tokenizer(text, padding, truncation, max_length, return_tensors):
        return {
            "input_ids": np.array([10, 20, 30]),
            "attention_mask": np.array([1, 1, 1]),
        }

    class Input:
        def __init__(self, name):
            self.name = name

    class SessionStub:
        def get_inputs(self):
            return [Input("input_ids"), Input("attention_mask")]

    model_config = SimpleNamespace(inputnames=None, max_length=10)
    session = SessionStub()

    inputs = SentenceTransformer._prepare_onnx_inputs(
        "hello world", tokenizer, model_config, session
    )

    assert "input_ids" in inputs
    assert inputs["input_ids"].shape == (1, 3)
    assert inputs["input_ids"].dtype == np.int64

    assert "attention_mask" in inputs
    assert inputs["attention_mask"].shape == (1, 3)
    assert inputs["attention_mask"].dtype == np.int64


def test_attention_mask_normalization_various_shapes():
    # Create distinguishable token embeddings so pooling result is predictable
    # seq_len was previously unused; remove assignment
    dim = 4
    # embeddings: tokens 0,1,2 vectors = [1],[2],[3] repeated
    emb_tokens = np.array([[1.0] * dim, [2.0] * dim, [3.0] * dim])
    emb_3d = emb_tokens[None, :, :]  # shape (1, seq_len, dim)

    model_config = SimpleNamespace(pooling_strategy="mean", force_pooling=False, normalize=False)

    # Case A: 1D mask -> should be treated as (1, seq_len)
    mask_1d = np.array([1, 0, 1])
    out = SentenceTransformer._process_embedding_output(
        emb_3d.copy(), model_config, mask_1d, projected_dimension=None, normalize=False
    )

    # Case B: mask shaped (1, seq_len) and embedding 2D
    mask_2d_row = np.array([[1, 0, 1]])
    emb_2d = emb_tokens.copy()  # shape (seq_len, dim)
    out2 = SentenceTransformer._process_embedding_output(
        emb_2d.copy(),
        model_config,
        mask_2d_row,
        projected_dimension=None,
        normalize=False,
    )

    # Case C: full (batch, seq_len) mask with batch=1
    mask_batch = np.array([[1, 0, 1]])
    out3 = SentenceTransformer._process_embedding_output(
        emb_3d.copy(),
        model_config,
        mask_batch,
        projected_dimension=None,
        normalize=False,
    )

    # All three mask shapes should result in the same pooled vector (up to
    # numeric differences caused by normalization). Flatten and compare.
    f1 = np.ravel(out)
    f2 = np.ravel(out2)
    f3 = np.ravel(out3)

    assert f1.shape == (dim,)
    assert np.allclose(f1, f2)
    assert np.allclose(f1, f3)
    assert np.isfinite(f1).all()
