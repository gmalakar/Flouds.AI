import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.summarizer_service import TextSummarizer
from app.models.summarization_response import SummarizationResponse, SummaryResults

class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    def decode(self, ids, skip_special_tokens=True):
        return "summary text"
    eos_token_id = 2
    bos_token_id = 0

class DummyModel:
    def generate(self, **kwargs):
        return [[1, 2, 3]]

@pytest.fixture
def dummy_model_config():
    class DummyConfig:
        summarization_task = "seq2seq-lm"
        encoder_onnx_model = "encoder_model.onnx"
        decoder_onnx_model = "decoder_model.onnx"
        special_tokens_map_path = "special_tokens_map.json"
        max_length = 10
        num_beams = 2
        early_stopping = True
        use_seq2seqlm = True
        padid = 0
        inputnames = type("InputNames", (), {})()
        outputnames = type("OutputNames", (), {})()
        decoder_inputnames = type("DecoderInputNames", (), {})()
        use_generation_config = False
        prepend_text = None
    return DummyConfig()

@patch("app.services.summarizer_service.TextSummarizer._get_tokenizer", return_value=DummyTokenizer())
@patch("app.services.summarizer_service.TextSummarizer.get_model", return_value=DummyModel())
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_seq2seqlm(mock_get_config, mock_get_model, mock_get_tokenizer, dummy_model_config):
    mock_get_config.return_value = dummy_model_config
    response = TextSummarizer.summarize("dummy-model", "This is a test.")
    assert isinstance(response, SummarizationResponse)
    assert isinstance(response.results, SummaryResults)
    assert response.results.summary == "summary text"

@patch("app.services.summarizer_service.TextSummarizer._get_tokenizer", return_value=DummyTokenizer())
@patch("app.services.summarizer_service.TextSummarizer.get_model", return_value=DummyModel())
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_batch_seq2seqlm(mock_get_config, mock_get_model, mock_get_tokenizer, dummy_model_config):
    mock_get_config.return_value = dummy_model_config
    responses = TextSummarizer.summarize_batch("dummy-model", ["Text 1", "Text 2"])
    assert all(isinstance(r, SummarizationResponse) for r in responses)
    assert all(isinstance(r.results, SummaryResults) for r in responses)
    assert all(r.results.summary == "summary text" for r in responses)

@patch("app.services.summarizer_service.TextSummarizer._get_tokenizer", return_value=DummyTokenizer())
@patch("app.services.summarizer_service.TextSummarizer._get_encoder_session")
@patch("app.services.summarizer_service.TextSummarizer._get_decoder_session")
@patch("app.services.summarizer_service.TextSummarizer._get_special_tokens", return_value={"<pad>", "<eos>"})
@patch("app.services.summarizer_service.ConfigLoader.get_onnx_config")
def test_summarize_other(
    mock_get_config, mock_get_special_tokens, mock_get_decoder_session, mock_get_encoder_session, mock_get_tokenizer, dummy_model_config
):
    # Simulate non-seq2seq-lm config
    dummy_model_config.use_seq2seqlm = False

    class DummySession:
        def run(self, *a, **kw):
            # Simulate output token ids as a 1D array
            return [np.array([1, 2, 2], dtype=np.int64)]
        def get_outputs(self):
            class DummyOutput:
                def __init__(self, name):
                    self.name = name
            return [DummyOutput("output")]

    mock_get_config.return_value = dummy_model_config
    mock_get_encoder_session.return_value = DummySession()
    mock_get_decoder_session.return_value = DummySession()

    response = TextSummarizer.summarize("dummy-model", "This is a test.")
    assert isinstance(response, SummarizationResponse)
    assert isinstance(response.results, SummaryResults)
    assert response.results.summary == "summary text"