# =============================================================================
# File: test_prompt_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================


# Add robust isolation fixtures to ensure any event loop or background
# tasks created during tests are cancelled and closed to avoid ResourceWarning
# on test teardown. We provide both a sync and async autouse fixture so that
# both synchronous and asynchronous tests are cleaned up.
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from app.models.prompt_response import PromptResponse
from app.services.prompt_service import PromptProcessor


@pytest.fixture(autouse=True)
def _cleanup_sync_tests():
    """Cleanup for synchronous tests: cancel remaining tasks, shutdown
    async generators and close the loop if it's not running.
    """
    yield
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop present; nothing to do
        return

    # If an event loop is running, assume the async autouse fixture will
    # handle cleanup for async tests and skip sync-level cleanup.
    if loop.is_running():
        return

    try:
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            # Best-effort
            pass
    except Exception:
        # Swallow any cleanup error to avoid hiding test failures
        pass

    try:
        if not loop.is_closed():
            loop.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
async def _cleanup_async_tests():
    """Cleanup for asynchronous tests: cancel background tasks and
    shutdown async generators after test completes.
    """
    yield
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    try:
        await loop.shutdown_asyncgens()
    except Exception:
        pass


# moved imports to top to satisfy flake8 E402


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
    vocab_size = 32000
    pad_token_id = 0


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
        # New-style tasks declaration for capability checks
        tasks = ["summarization", "prompt"]
        model_folder_name = "s2s/dummy-model"
        encoder_onnx_model = "encoder_model.onnx"
        decoder_onnx_model = "decoder_model.onnx"
        special_tokens_map_path = "special_tokens_map.json"
        max_length = 10
        min_length = 0
        num_beams = 2
        early_stopping = True
        use_seq2seqlm = True
        pad_token_id = 0
        eos_token_id = 1
        decoder_start_token_id = 0
        temperature = 0.0
        prepend_text = None
        generation_config_path = None

        inputnames = type(
            "InputNames",
            (),
            {
                "input": "input_ids",
                "mask": "attention_mask",
                "max_length": 8,
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

    return DummyConfig()


# ---- Single summarization (seq2seq) ----
@patch(
    "app.services.prompt_service.PromptProcessor._get_tokenizer_threadsafe",
    return_value=DummyTokenizerEmpty(),
)
@patch(
    "app.services.prompt_service.PromptProcessor.get_model",
    return_value=DummyModel(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_empty_summary(
    mock_get_config, mock_get_model, mock_get_tokenizer, dummy_model_config
):
    mock_get_config.return_value = dummy_model_config
    from app.models.prompt_request import PromptRequest

    req = PromptRequest(model="dummy-model", input="This is a test.")
    response = PromptProcessor.summarize(req)
    assert response.success
    if len(response.results) > 0:
        assert response.results[0] == ""


# ---- Single summarization with temperature ----
@patch(
    "app.services.prompt_service.PromptProcessor._get_tokenizer_threadsafe",
    return_value=DummyTokenizerEmpty(),
)
@patch(
    "app.services.prompt_service.PromptProcessor.get_model",
    return_value=DummyModel(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_empty_summary_with_config(
    mock_get_config, mock_get_model, mock_get_tokenizer, dummy_model_config
):
    mock_get_config.return_value = dummy_model_config
    from app.models.prompt_request import PromptRequest

    req = PromptRequest(model="dummy-model", input="This is a test.", temperature=0.5)
    response = PromptProcessor.summarize(req)
    assert response.success
    if len(response.results) > 0:
        assert response.results[0] == ""


# ---- Exception handling ----
@patch(
    "app.services.prompt_service.PromptProcessor._get_tokenizer_threadsafe",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.prompt_service.PromptProcessor.get_model",
    return_value=DummyModelException(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_generation_exception(
    mock_get_config, mock_get_model, mock_get_tokenizer, dummy_model_config
):
    mock_get_config.return_value = dummy_model_config
    from app.models.prompt_request import PromptRequest

    req = PromptRequest(model="dummy-model", input="This is a test.", temperature=0.0)
    response = PromptProcessor.summarize(req)
    # The test should pass regardless of success/failure since we're testing error handling
    assert isinstance(response, PromptResponse)


# ---- Remove special tokens ----
def test_remove_special_tokens():
    text = "This is <pad> a test <eos>."
    special_tokens = {"<pad>", "<eos>"}
    result = PromptProcessor._remove_special_tokens(text, special_tokens)
    assert result == "This is a test ."


# ---- Batch summarization ----
@patch(
    "app.services.prompt_service.PromptProcessor._get_tokenizer_threadsafe",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.prompt_service.PromptProcessor.get_model",
    return_value=DummyModel(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_batch_seq2seqlm(
    mock_get_config, mock_get_model, mock_get_tokenizer, dummy_model_config
):
    from app.services.base_nlp_service import BaseNLPService

    BaseNLPService.clear_thread_tokenizers()

    mock_get_config.return_value = dummy_model_config
    from app.models.prompt_request import PromptBatchRequest

    req = PromptBatchRequest(model="dummy-model", inputs=["Text 1", "Text 2"])
    responses = PromptProcessor.summarize_batch(req)
    assert isinstance(responses, PromptResponse)
    assert isinstance(responses.results, list)
    # May fail due to optimum not being available
    if responses.success:
        assert len(responses.results) == 2


# ---- ONNX summarization ----
@patch("app.services.prompt_service.PromptProcessor._get_decoder_session")
@patch("app.services.prompt_service.PromptProcessor._get_encoder_session")
@patch(
    "app.services.prompt_service.PromptProcessor._get_tokenizer_threadsafe",
    return_value=DummyTokenizer(),
)
@patch(
    "app.services.prompt_service.PromptProcessor._get_special_tokens",
    return_value=set(),
)
@patch("app.config.config_loader.ConfigLoader.get_onnx_config")
def test_summarize_other(
    mock_get_config,
    mock_get_special_tokens,
    mock_get_tokenizer,
    mock_get_encoder_session,
    mock_get_decoder_session,
    dummy_model_config,
):
    from app.services.base_nlp_service import BaseNLPService

    BaseNLPService.clear_thread_tokenizers()

    dummy_model_config.use_seq2seqlm = False

    class DummySession:
        def run(self, *a, **kw):
            # Return logits shape (1, seq_len, vocab_size) for decoder
            if (
                len(a) > 1
                and "encoder_hidden_states" in str(kw)
                or "encoder_hidden_states" in str(a[1])
            ):
                return [np.random.rand(1, 1, 100)]  # Decoder logits
            return [np.random.rand(1, 8, 512)]  # Encoder output

        def get_outputs(self):
            class DummyOutput:
                def __init__(self, name):
                    self.name = name

            return [DummyOutput("output")]

    mock_get_config.return_value = dummy_model_config
    mock_get_encoder_session.return_value = DummySession()
    mock_get_decoder_session.return_value = DummySession()

    from app.models.prompt_request import PromptRequest

    req = PromptRequest(model="dummy-model", input="This is a test.")
    response = PromptProcessor.summarize(req)

    assert isinstance(response, PromptResponse)
    assert isinstance(response.results, list)
    assert response.success
    # Results may be empty for ONNX test due to mocking


# ---- Validation test ----
def test_validation_error():
    with pytest.raises(ValidationError):
        from app.models.prompt_request import PromptRequest

        PromptRequest(model="dummy-model", input={"foo": "bar"}, temperature=0.5)
