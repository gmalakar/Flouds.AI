from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.models.summarization_response import SummarizationResponse, SummaryResults
from app.services.summarizer_service import TextSummarizer

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
        decoder_start_token_id = 0  # <-- Add this line
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

# ---- Single summarization (seq2seq) ----
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value={},
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
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_seq2seqlm(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    dummy_model_config,
):
    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationRequest

    req = SummarizationRequest(
        model="dummy-model", input="This is a test.", use_optimized_model=False
    )
    response = TextSummarizer.summarize(req)
    assert isinstance(response, SummarizationResponse)
    assert isinstance(response.results, SummaryResults)
    assert response.results.summary == "summary text"

# ---- Single summarization (empty summary) ----
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value={},
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
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_empty_summary(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    dummy_model_config,
):
    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationRequest

    req = SummarizationRequest(
        model="dummy-model", input="This is a test.", use_optimized_model=False
    )
    response = TextSummarizer.summarize(req)
    assert response.results.summary == ""
    assert response.success

# ---- Exception handling ----
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value={},
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
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_generation_exception(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    dummy_model_config,
):
    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationRequest

    req = SummarizationRequest(
        model="dummy-model", input="This is a test.", use_optimized_model=False
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

# ---- Batch summarization ----
@patch(
    "app.services.summarizer_service.TextSummarizer._get_generation_config",
    return_value={},
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
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_batch_seq2seqlm(
    mock_get_config,
    mock_get_model,
    mock_get_tokenizer,
    mock_get_auto_config,
    mock_get_generation_config,
    dummy_model_config,
):
    mock_get_config.return_value = dummy_model_config
    from app.models.summarization_request import SummarizationBatchRequest

    req = SummarizationBatchRequest(
        model="dummy-model",
        inputs=["Text 1", "Text 2"],
        use_optimized_model=False,
    )
    responses = TextSummarizer.summarize_batch(req)
    assert all(isinstance(r, SummarizationResponse) for r in responses)
    assert all(isinstance(r.results, SummaryResults) for r in responses)
    assert all(r.results.summary == "summary text" for r in responses)

# ---- ONNX/Other summarization ----
@patch("transformers.GenerationConfig.from_pretrained", return_value=type("GenCfg", (), {"to_dict": lambda self: {}})())
@patch("transformers.AutoConfig.from_pretrained", return_value=type("AutoConfig", (), {"model_type": "t5", "pad_token_id": 0, "eos_token_id": 1})())
@patch("app.services.summarizer_service.TextSummarizer._get_generation_config", return_value={"decoder_start_token_id": 0})
@patch("app.services.summarizer_service.TextSummarizer._get_tokenizer", return_value=DummyTokenizer())
@patch("app.services.summarizer_service.TextSummarizer._get_encoder_session")
@patch("app.services.summarizer_service.TextSummarizer._get_decoder_session")
@patch("app.services.summarizer_service.TextSummarizer._get_special_tokens", return_value={"<pad>", "<eos>"})
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_other(
    mock_get_config,
    mock_get_special_tokens,
    mock_get_decoder_session,
    mock_get_encoder_session,
    mock_get_tokenizer,
    mock_get_generation_config,
    mock_auto_config,
    mock_gen_cfg,
    dummy_model_config,
):
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

    req = SummarizationRequest(
        model="dummy-model", input="This is a test.", use_optimized_model=False
    )
    response = TextSummarizer.summarize(req)
    assert isinstance(response, SummarizationResponse)
    assert isinstance(response.results, SummaryResults)
    assert response.results.summary == "summary text"

    with pytest.raises(ValidationError):
        SummarizationRequest(
            model="dummy-model", input={"foo": "bar"}, use_optimized_model=False
        )
