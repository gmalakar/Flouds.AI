from unittest.mock import patch

import numpy as np
import pytest

from app.services.embedder_service import STOP_WORDS, SentenceTransformer


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


@pytest.fixture
def dummy_model_config():
    class DummyConfig:
        embedder_task = "feature-extraction"
        encoder_onnx_model = "model.onnx"
        normalize = True
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
            },
        )()

    return DummyConfig()


def test_preprocess_text_removes_stopwords():
    text = "The quick brown fox jumps over the lazy dog."
    processed = SentenceTransformer._preprocess_text(text)
    processed_words = set(processed.lower().split())
    for stop_word in STOP_WORDS:
        assert stop_word not in processed_words
    assert "quick" in processed_words


def test_truncate_text_to_token_limit():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight Nine. Ten."
    tokenizer = DummyTokenizer()
    truncated = SentenceTransformer._truncate_text_to_token_limit(
        text, tokenizer, max_tokens=5
    )
    assert len(truncated.split()) <= 5


def test_split_text_into_chunks():
    text = "Sentence one. Sentence two. Sentence three."
    tokenizer = DummyTokenizer()
    chunks = SentenceTransformer._split_text_into_chunks(text, tokenizer, max_tokens=3)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 0


@patch("app.services.embedder_service.SentenceTransformer._get_model_config")
@patch(
    "app.services.embedder_service.SentenceTransformer._get_tokenizer",
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
    text = "This is a test. Another sentence."
    response = SentenceTransformer.embed_text(
        text, "dummy-model", projected_dimension=8
    )
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
    assert isinstance(embedding, list)
    assert len(embedding) == 8
    assert all(
        isinstance(x, (float, np.floating, np.float32, np.float64)) for x in embedding
    )


def test_project_embedding_dimension():
    emb = np.ones(8)
    projected = SentenceTransformer._project_embedding(emb, projected_dimension=4)
    assert projected.shape == (4,)


def test_embed_text_handles_exception(monkeypatch, dummy_model_config):
    def raise_exc(*a, **kw):
        raise Exception("fail")

    monkeypatch.setattr(
        SentenceTransformer, "_get_model_config", lambda *a, **k: dummy_model_config
    )
    monkeypatch.setattr(
        SentenceTransformer, "_get_tokenizer", lambda *a, **k: DummyTokenizer()
    )
    monkeypatch.setattr(
        SentenceTransformer, "_get_encoder_session", lambda *a, **k: DummySession()
    )
    monkeypatch.setattr(SentenceTransformer, "_split_text_into_chunks", raise_exc)
    response = SentenceTransformer.embed_text(
        "fail test", "dummy-model", projected_dimension=8
    )
    assert response.success is False
    assert "Error generating embedding" in response.message
