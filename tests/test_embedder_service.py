# =============================================================================
# File: test_embedder_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from unittest.mock import patch

import numpy as np
import pytest

from app.models.embedding_request import EmbeddingBatchRequest, EmbeddingRequest
from app.models.embedding_response import EmbededChunk
from app.services.embedder_service import SentenceTransformer


class DummyTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))

    def __call__(self, text, **kwargs):
        length = len(self.encode(text))
        return {
            "input_ids": np.ones((1, length), dtype=np.int64),
            "attention_mask": np.ones((1, length), dtype=np.int64),
        }


class DummySession:
    def run(self, _, inputs):
        # Return a dummy embedding of shape (1, 8)
        return [np.ones((1, 8), dtype=np.float32)]

    def get_outputs(self):
        class DummyOutput:
            def __init__(self, name):
                self.name = name

        return [DummyOutput("output")]

    def get_inputs(self):
        class DummyInput:
            def __init__(self, name):
                self.name = name

        return [DummyInput("input_ids"), DummyInput("attention_mask")]


class DummyConfig:
    embedder_task = "fe"
    encoder_onnx_model = "model.onnx"
    normalize = True
    pooling_strategy = "mean"
    chunk_logic = "sentence"
    chunk_overlap = 1
    inputnames = type(
        "inputnames",
        (),
        {
            "input": "input_ids",
            "mask": "attention_mask",
            "max_length": 8,
            "tokentype": None,
            "position": None,
            "use_decoder_input": False,
            "decoder_input_name": None,
        },
    )()
    outputnames = type(
        "outputnames",
        (),
        {
            "logits": False,
        },
    )()


@pytest.fixture
def dummy_model_config():
    return DummyConfig()


@pytest.fixture(autouse=True)
def isolate_tests():
    import logging

    # Clear all handlers before test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    yield
    # Clear all handlers after test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def test_truncate_text_to_token_limit():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight Nine. Ten."
    tokenizer = DummyTokenizer()
    truncated = SentenceTransformer._truncate_text_to_token_limit(
        text, tokenizer, max_tokens=5
    )
    assert len(truncated) <= 20  # 5 tokens * 4 chars estimate


def test_split_text_into_chunks():
    text = "Sentence one. Sentence two. Sentence three."
    tokenizer = DummyTokenizer()
    config = DummyConfig()
    from app.utils.chunking_strategies import ChunkingStrategies

    chunks = ChunkingStrategies.split_text_into_chunks(
        text, tokenizer, max_tokens=3, model_config=config
    )
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 0


@patch("app.services.embedder_service.SentenceTransformer._get_model_config")
@patch(
    "app.services.embedder_service.SentenceTransformer._get_tokenizer_threadsafe",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.embedder_service.SentenceTransformer._get_encoder_session",
    return_value=DummySession(),
)
def test_embed_text_success(
    mock_session, mock_tokenizer, mock_config, dummy_model_config
):
    mock_config.return_value = dummy_model_config
    req = EmbeddingRequest(
        model="dummy-model",
        input="This is a test. Another sentence.",
        projected_dimension=8,
    )
    response = SentenceTransformer.embed_text(req)
    assert response.success is True
    assert response.model == "dummy-model"
    assert isinstance(response.results, list)
    for chunk in response.results:
        assert hasattr(chunk, "vector")
        assert hasattr(chunk, "chunk")
        assert isinstance(chunk.vector, list)
        assert all(
            isinstance(x, (float, np.floating, np.float32, np.float64))
            for x in chunk.vector
        )
        assert isinstance(chunk.chunk, str)


def test_small_text_embedding_returns_flat_list(dummy_model_config):
    embedding = SentenceTransformer._small_text_embedding(
        "Hello world",
        dummy_model_config,
        DummyTokenizer(),
        DummySession(),
        projected_dimension=8,
    )
    assert isinstance(embedding.EmbeddingResults, list)
    assert len(embedding.EmbeddingResults) == 8  # Should match projected_dimension
    assert all(
        isinstance(x, (float, np.floating, np.float32, np.float64))
        for x in embedding.EmbeddingResults
    )


def test_project_embedding_dimension():
    emb = np.ones(8)
    projected = SentenceTransformer._project_embedding(emb, projected_dimension=4)
    assert projected.shape == (4,)


def test_embed_text_handles_exception(monkeypatch, dummy_model_config):
    def raise_exc(*a, **kw):
        raise Exception("fail")

    monkeypatch.setattr(SentenceTransformer, "_embed_text_local", raise_exc)

    req = EmbeddingRequest(
        model="dummy-model", input="fail test", projected_dimension=8
    )

    response = SentenceTransformer.embed_text(req)
    assert response.success is False
    assert "fail" in response.message or "Error" in response.message


@patch("app.services.embedder_service.SentenceTransformer._get_model_config")
@patch(
    "app.services.embedder_service.SentenceTransformer._get_tokenizer_threadsafe",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.embedder_service.SentenceTransformer._get_encoder_session",
    return_value=DummySession(),
)
@pytest.mark.asyncio
async def test_embed_batch_async(
    mock_session, mock_tokenizer, mock_config, dummy_model_config
):
    mock_config.return_value = dummy_model_config
    requests = EmbeddingBatchRequest(
        model="dummy-model",
        projected_dimension=8,
        inputs=["First text.", "Second text."],
    )
    response = await SentenceTransformer.embed_batch_async(requests)
    assert response.success is True
    assert response.model == "dummy-model"
    assert len(response.results) > 0  # Should have some results
    print(f"Response results: {response.results}")
    for chunk in response.results:
        assert isinstance(chunk, EmbededChunk)
