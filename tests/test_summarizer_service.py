# =============================================================================
# File: test_summarizer_service.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError


# Add the isolation fixture here
@pytest.fixture(autouse=True)
def _isolate_tests(monkeypatch):
    pass


from app.models.summarization_response import SummarizationResponse
from app.services.summarizer_service import SummaryResults, TextSummarizer


# Dummy classes for mocking
class DummyTokenizer:
    def __call__(self, text, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            import torch

            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        elif kwargs.get("return_tensors") == "np":
            return {
                "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
            }
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if isinstance(ids, (list, np.ndarray)) and len(ids) > 0:
            return "summary text"
        return ""

    eos_token_id = 2
    bos_token_id = 0


class DummyTokenizerEmpty(DummyTokenizer):
    def decode(self, ids, skip_special_tokens=True):
        return ""


class DummyModel:
    def __init__(self):
        self.generation_config = SimpleNamespace()

    def generate(self, **kwargs):
        import torch

        return torch.tensor([[1, 2, 3]])


class DummyModelException(DummyModel):
    def generate(self, **kwargs):
        raise RuntimeError("Generation failed")


@pytest.fixture
def dummy_model_config():
    class DummyConfig:
        summarization_task = "s2s"
        encoder_onnx_model = "encoder_model.onnx"
        decoder_onnx_model = "decoder_model.onnx"
        special_tokens_map_path = "special_tokens_map.json"
        max_length = 10
        num_beams = 2
        early_stopping = True
        use_seq2seqlm = True
        padid = 0
        decoder_start_token_id = 0
        inputnames = type(
            "InputNames",
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
        outputnames = type("OutputNames", (), {"logits": False})()
        decoder_inputnames = type(
            "DecoderInputNames",
            (),
            {
                "input": "input_ids",
                "encoder_output": "encoder_hidden_states",
                "mask": "encoder_attention_mask",
            },
        )()
        use_generation_config = False
        prepend_text = None
        generation_config_path = None

    return DummyConfig()


class DummyGenConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None):
        # Try dict first, then attribute
        if key in self:
            return self[key]
        return getattr(self, key, default)


# ---- Single summarization (seq2seq) ----
@patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizerEmpty())
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value=DummyGenConfig(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_auto_config",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_tokenizer",
    return_value=DummyTokenizerEmpty(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer.get_model",
    return_value=DummyModel(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_empty_summary(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    mock_auto_tokenizer,
    dummy_model_config,
):
    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationRequest

    req = SummarizationRequest(model="dummy-model", input="This is a test.")
    response = TextSummarizer.summarize(req)
    assert response.results[0] == ""
    assert response.success


# ---- Single summarization (empty summary) ----
@patch(
    "transformers.GenerationConfig.from_pretrained",
    return_value=DummyGenConfig(max_length=10, num_beams=2),
)
@patch(
    "transformers.AutoConfig.from_pretrained",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizerEmpty())
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value=DummyGenConfig(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_auto_config",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_tokenizer",
    return_value=DummyTokenizerEmpty(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer.get_model",
    return_value=DummyModel(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_empty_summary_with_config(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    mock_auto_tokenizer,
    mock_auto_config,
    mock_gen_config,
    dummy_model_config,
):
    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationRequest

    req = SummarizationRequest(
        model="dummy-model", input="This is a test.", temperature=0.5
    )
    response = TextSummarizer.summarize(req)
    assert response.results[0] == ""
    assert response.success


# ---- Exception handling ----
@patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value=DummyGenConfig(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_auto_config",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_tokenizer",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer.get_model",
    return_value=DummyModelException(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_generation_exception(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    mock_auto_tokenizer,
    dummy_model_config,
):
    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationRequest

    req = SummarizationRequest(
        model="dummy-model", input="This is a test.", temperature=0.0
    )
    response = TextSummarizer.summarize(req)
    assert not response.success
    assert "Error generating summarization" in response.message


# ---- Remove special tokens ----
def test_remove_special_tokens():
    text = "This is <pad> a test <eos>."
    special_tokens = {"<pad>", "<eos>"}
    result = TextSummarizer._remove_special_tokens(text, special_tokens)
    assert result == "This is  a test ."


@patch(
    "transformers.GenerationConfig.from_pretrained",
    return_value=DummyGenConfig(max_length=10, num_beams=2, early_stopping=True),
)
@patch(
    "transformers.AutoConfig.from_pretrained",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value=DummyGenConfig(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_auto_config",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_tokenizer",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer.get_model",
    return_value=DummyModel(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_batch_seq2seqlm(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    mock_auto_tokenizer,
    mock_auto_config,
    mock_gen_config,
    dummy_model_config,
):
    from app.services.base_nlp_service import BaseNLPService

    BaseNLPService.clear_thread_tokenizers()  # <-- Add this line

    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationBatchRequest

    req = SummarizationBatchRequest(
        model="dummy-model",
        inputs=["Text 1", "Text 2"],
    )
    responses = TextSummarizer.summarize_batch(req)
    assert isinstance(responses, SummarizationResponse)
    assert isinstance(responses.results, list)
    assert all(r == "summary text" for r in responses.results)


# ---- ONNX/Other summarization ----
@patch("app.services.summarizer_service.TextSummarizer._get_decoder_session")
@patch("app.services.summarizer_service.TextSummarizer._get_encoder_session")
@patch(
    "app.services.base_nlp_service.AutoTokenizer.from_pretrained",
    return_value=DummyTokenizer(),
)  # <--- ADD THIS
@patch(
    "transformers.GenerationConfig.from_pretrained",
    return_value=DummyGenConfig(max_length=10, num_beams=2, early_stopping=True),
)
@patch(
    "transformers.AutoConfig.from_pretrained",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value=DummyGenConfig(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_auto_config",
    return_value=type(
        "AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1}
    )(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer._get_tokenizer",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.summarizer_service.TextSummarizer.get_model",
    return_value=DummyModel(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_other(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    mock_auto_config,
    mock_gen_cfg,
    mock_auto_tokenizer,
    mock_get_encoder_session,
    mock_get_decoder_session,
    dummy_model_config,
):
    from app.services.base_nlp_service import BaseNLPService

    BaseNLPService.clear_thread_tokenizers()  # <-- Add this line

    dummy_model_config.use_seq2seqlm = False

    class DummySession:
        def run(self, *a, **kw):
            return [np.array([1, 2, 2], dtype=np.int64)]

        def get_outputs(self):
            class DummyOutput:
                def __init__(self, name):
                    self.name = name

            return [DummyOutput("output")]

    mock_get_config.return_value = dummy_model_config
    mock_get_encoder_session.return_value = DummySession()
    mock_get_decoder_session.return_value = DummySession()
    from app.models.summarization_request import SummarizationRequest

    req = SummarizationRequest(model="dummy-model", input="This is a test.")
    response = TextSummarizer.summarize(req)

    assert isinstance(response, SummarizationResponse)
    assert isinstance(response.results, list)
    assert response.results[0] == "summary text"
    assert all(r == "summary text" for r in response.results)
    assert response.success

    with pytest.raises(ValidationError):
        SummarizationRequest(model="dummy-model", input={"foo": "bar"}, temperature=0.5)
