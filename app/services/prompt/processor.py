# =============================================================================
# File: processor.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Main prompt processor orchestrating the full text generation pipeline."""

import threading
import time
from asyncio import TimeoutError as AsyncTimeoutError
from asyncio import gather
from typing import Any, Optional

from app.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ProcessingTimeoutError,
    TokenizerError,
)
from app.logger import get_logger
from app.models.prompt_request import PromptBatchRequest, PromptRequest
from app.models.prompt_response import PromptResponse
from app.services.base_nlp_service import BaseNLPService
from app.services.prompt import generator as generator_module

# Import submodule functions
from app.services.prompt.config import (
    get_model_config,
    get_model_path,
    get_tokenizer_threadsafe,
    validate_model_availability,
)
from app.services.prompt.generator import (
    run_decoder_only_generation,
    run_encoder_only_generation,
    run_onnx_generation,
    run_seq2seq_generation,
)
from app.services.prompt.models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    SummaryResults,
)
from app.services.prompt.parameters import build_generation_params
from app.services.prompt.resource_manager import (
    clear_all_caches,
    get_cached_model,
    get_decoder_session,
    get_encoder_session,
    get_special_tokens,
)
from app.services.prompt.text_utils import capitalize_sentences, remove_special_tokens

# prepare_input_text not used in this module
from app.utils.batch_limiter import BatchLimiter
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger(__name__)


class PromptProcessor(BaseNLPService):
    """Main processor for text generation and summarization.

    This class acts as a facade, orchestrating the modular components:
    - config: Resolves model configurations
    - resource_manager: Manages cached sessions and models
    - generator: Implements generation strategies
    - text_utils: Preprocesses/postprocesses text
    - parameters: Builds generation parameters
    """

    @staticmethod
    def process_prompt(request: PromptRequest) -> PromptResponse:
        """Process prompt for text generation/summarization.

        Main entry point for prompt processing. Orchestrates resource loading,
        generation strategy selection, and response formatting.

        Args:
            request: Prompt request with input text and model name

        Returns:
            PromptResponse with generated text or error message
        """
        start_time = time.time()
        logger.debug(
            "Processing prompt for model: %s, input: %s",
            sanitize_for_log(request.model),
            sanitize_for_log(request.input[:100]) if request.input else "",
        )

        response = PromptResponse(
            success=True,
            message="Prompt processed successfully",
            model=request.model or DEFAULT_MODEL,
            results=[],
            time_taken=0.0,
        )

        try:
            timeout = getattr(request, "timeout", DEFAULT_TIMEOUT)
            timeout_occurred = [False]

            def timeout_handler():
                timeout_occurred[0] = True

            timer = threading.Timer(timeout, timeout_handler)
            timer.start()

            try:
                result: SummaryResults = PromptProcessor._process_prompt_local(request)
                if timeout_occurred[0]:
                    raise ProcessingTimeoutError(
                        f"Prompt processing timed out after {timeout} seconds"
                    )
                elif isinstance(result, SummaryResults) and result.success:
                    response.results.append(result.summary)
                    response.message = result.message
                else:
                    response.success = False
                    response.message = result.message
            finally:
                timer.cancel()

        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            response.success = False
            response.message = str(e)

        # finalize timing and return outside finally to avoid silencing exceptions (B012)
        response.time_taken = time.time() - start_time
        return response

    @staticmethod
    def summarize(request: PromptRequest) -> PromptResponse:
        """Backward compatibility method for summarization.

        Delegates to process_prompt for consistent behavior.

        Args:
            request: Prompt request

        Returns:
            PromptResponse with summarization results
        """
        return PromptProcessor.process_prompt(request)

    @staticmethod
    def _process_prompt_local(request: PromptRequest) -> SummaryResults:
        """Core prompt processing logic with generation strategy selection.

        Args:
            request: Prompt request

        Returns:
            SummaryResults with generated text or error

        Raises:
            Various exceptions for model loading/configuration errors
        """
        model_to_use = request.model or DEFAULT_MODEL

        try:
            model_config = get_model_config(model_to_use)
            if not model_config:
                return SummaryResults(
                    summary="",
                    message=f"Model '{model_to_use}' not found",
                    success=False,
                )

            model_path, tokenizer = PromptProcessor._prepare_model_resources(
                model_config, model_to_use
            )

            # Log generation path selection
            try:
                use_seq2seqlm = bool(getattr(model_config, "use_seq2seqlm", False))
                encoder_only = bool(getattr(model_config, "encoder_only", False))
                decoder_only = bool(getattr(model_config, "decoder_only", False))
                logger.info(
                    "Generation path selection for model=%s: use_seq2seqlm=%s encoder_only=%s decoder_only=%s",
                    sanitize_for_log(model_to_use),
                    use_seq2seqlm,
                    encoder_only,
                    decoder_only,
                )
            except Exception:
                logger.debug("Failed to log generation path flags")

            # If no explicit flags provided, attempt to infer decoder-only
            # when encoder artifact is missing but decoder artifact exists.
            try:
                if not use_seq2seqlm and not decoder_only and not encoder_only:
                    from app.services.prompt.config import get_model_file_paths

                    enc_path, dec_path = get_model_file_paths(model_path, model_config)
                    import os

                    enc_exists = os.path.exists(enc_path) if enc_path else False
                    dec_exists = os.path.exists(dec_path) if dec_path else False
                    if not enc_exists and dec_exists:
                        logger.info(
                            "Inferring decoder-only for %s (encoder missing, decoder present)",
                            sanitize_for_log(model_to_use),
                        )
                        decoder_only = True

            except Exception:
                # Non-fatal: fall back to explicit flags
                logger.debug("Failed to infer decoder-only; using explicit flags if present")

            # Dispatch to appropriate generation strategy
            if use_seq2seqlm:
                logger.info("Using seq2seq generation for %s", sanitize_for_log(model_to_use))
                model = get_cached_model(model_path, model_config)
                if not model:
                    return SummaryResults(
                        summary="",
                        message="Failed to load seq2seq model",
                        success=False,
                    )
                return run_seq2seq_generation(model, tokenizer, model_config, request)

            elif decoder_only:
                logger.info(
                    "Using decoder-only generation for %s",
                    sanitize_for_log(model_to_use),
                )
                return run_decoder_only_generation(model_path, tokenizer, model_config, request)

            elif encoder_only:
                logger.info(
                    "Using encoder-only generation for %s",
                    sanitize_for_log(model_to_use),
                )
                return run_encoder_only_generation(model_path, tokenizer, model_config, request)

            else:
                logger.info(
                    "Using encoder+decoder ONNX generation for %s",
                    sanitize_for_log(model_to_use),
                )

                # Get encoder and decoder
                from app.services.prompt.config import get_model_file_paths

                encoder_path, decoder_path = get_model_file_paths(model_path, model_config)
                encoder_session = PromptProcessor._get_encoder_session(encoder_path)
                decoder_session = PromptProcessor._get_decoder_session(decoder_path)

                if not encoder_session or not decoder_session:
                    return SummaryResults(
                        summary="",
                        message="Failed to load encoder/decoder sessions",
                        success=False,
                    )

                special_tokens = get_special_tokens(
                    get_special_tokens_path(model_path, model_config)
                )

                return run_onnx_generation(
                    encoder_session,
                    decoder_session,
                    tokenizer,
                    special_tokens,
                    model_config,
                    request,
                    encoder_path,
                )

        except FileNotFoundError:
            raise ModelNotFoundError("Model files not accessible")
        except OSError as e:
            raise ModelLoadError(f"System error accessing model: {e}")
        except (ValueError, KeyError) as e:
            raise InferenceError(f"Invalid model configuration: {e}")
        except Exception as e:
            raise InferenceError(f"Error generating text: {e}")

    @staticmethod
    def _prepare_model_resources(model_config: Any, model_name: str) -> tuple[str, Any]:
        """Prepare model path and tokenizer for generation.

        Args:
            model_config: ONNX model configuration
            model_name: Model name

        Returns:
            Tuple of (model_path, tokenizer)

        Raises:
            ModelLoadError: If resources cannot be loaded
            TokenizerError: If tokenizer cannot be loaded
        """
        model_path = get_model_path(model_name)
        if not model_path:
            raise ModelLoadError(f"Failed to resolve model path for {model_name}")

        if not validate_model_availability(
            model_name,
            task="prompt",
            perform_filesystem_check=False,
            cfg=model_config,
            model_path=model_path,
        ):
            raise ModelLoadError(f"Model '{model_name}' not available or model files missing")

        logger.info("Using model path: %s", sanitize_for_log(model_path))

        use_legacy = getattr(model_config, "legacy_tokenizer", False)
        tokenizer = PromptProcessor._get_tokenizer_threadsafe(model_path, use_legacy)
        if not tokenizer:
            raise TokenizerError(f"Failed to load tokenizer: {model_path}")

        return model_path, tokenizer

    @staticmethod
    def _get_encoder_session(encoder_model_path: str) -> Optional[Any]:
        """Get or create ONNX encoder session.

        Args:
            encoder_model_path: Path to encoder model

        Returns:
            ONNX inference session, or None if creation failed
        """
        try:
            return get_encoder_session(encoder_model_path)
        except Exception as e:
            logger.error(f"Failed to create encoder session: {e}")
            return None

    @staticmethod
    def _get_decoder_session(decoder_model_path: str) -> Optional[Any]:
        """Get or create ONNX decoder session (wrapper for test overrides)."""
        try:
            return get_decoder_session(decoder_model_path)
        except Exception as e:
            logger.error(f"Failed to create decoder session: {e}")
            return None

    @staticmethod
    def clear_model_cache():
        """Clear all cached models, sessions, and special tokens."""
        logger.info("Clearing all model/session/special token caches.")
        clear_all_caches()

    @staticmethod
    async def summarize_batch_async(request: PromptBatchRequest) -> PromptResponse:
        """Asynchronous batch summarization with partial success reporting.

        Args:
            request: Batch request with multiple input texts

        Returns:
            PromptResponse with results for each input
        """
        start_time = time.time()
        logger.debug(
            "Batch summarization for model: %s, batch size: %d",
            sanitize_for_log(request.model),
            len(request.inputs),
        )

        response = PromptResponse(
            success=True,
            message="Batch prompt processing completed successfully",
            model=request.model or DEFAULT_MODEL,
            results=[],
            time_taken=0.0,
        )

        try:
            # Validate batch size
            BatchLimiter.validate_batch_size(request.inputs, max_size=DEFAULT_BATCH_SIZE)

            import asyncio

            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(
                    None,
                    PromptProcessor._process_prompt_local,
                    PromptRequest(
                        model=request.model,
                        input=text,
                        temperature=getattr(request, "temperature", None),
                        tenant_code=request.tenant_code,
                    ),
                )
                for text in request.inputs
            ]
            results = await gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    err_msg = str(result)
                    logger.error(f"Error in input {idx}: {err_msg}")
                    response.success = False
                    response.message = f"Error in input {idx}: {err_msg}"
                elif getattr(result, "success", True) is False:
                    err_msg = getattr(result, "message", "")
                    logger.error(f"Error in input {idx}: {err_msg}")
                    response.success = False
                    response.message = f"Error in input {idx}: {err_msg}"
                else:
                    summary_val = getattr(result, "summary", None)
                    if summary_val is not None:
                        response.results.append(summary_val)

        except (ValueError, TypeError) as e:
            response.success = False
            response.message = "Invalid batch request parameters"
            logger.error("Batch validation error: %s", str(e))

        except (AsyncTimeoutError, TimeoutError) as e:
            response.success = False
            response.message = "Batch summarization timed out"
            logger.error("Batch timeout: %s", str(e))

        except Exception as e:
            response.success = False
            response.message = f"Error generating summarization: {str(e)}"
            logger.exception("Unexpected error during batch summarization")

        # finalize timing and return outside finally to avoid silencing exceptions (B012)
        response.time_taken = time.time() - start_time
        return response

    @classmethod
    def summarize_batch(cls, request: PromptBatchRequest) -> PromptResponse:
        """Backward compatibility for synchronous batch processing.

        Args:
            request: Batch request

        Returns:
            PromptResponse with batch results
        """
        import asyncio

        return asyncio.run(cls.summarize_batch_async(request))

    # ------------------------------------------------------------------
    # Legacy-compatible helper wrappers for tests and compatibility
    # ------------------------------------------------------------------
    @staticmethod
    def _get_tokenizer_threadsafe(model_path: str, use_legacy: bool = False) -> Any:
        """Delegate to shared tokenizer loader (kept as override point for tests)."""
        return get_tokenizer_threadsafe(model_path, use_legacy)

    @staticmethod
    def _decode_output(output_ids, tokenizer, special_tokens, input_text):
        return generator_module._decode_output(output_ids, tokenizer, special_tokens, input_text)

    @staticmethod
    def _remove_special_tokens(text, special_tokens):
        return remove_special_tokens(text, special_tokens)

    @staticmethod
    def _capitalize_sentences(text):
        return capitalize_sentences(text)

    @staticmethod
    def _hash_array(arr, quantize_dtype=None):
        return generator_module._hash_array(arr, quantize_dtype)

    @staticmethod
    def _build_generation_cache_key(
        decoder_session,
        encoder_outputs,
        inputs,
        token_config,
        model_config,
        request,
        tokenizer=None,
    ):
        return generator_module._build_generation_cache_key(
            decoder_session,
            encoder_outputs,
            inputs,
            token_config,
            model_config,
            request,
            tokenizer,
        )

    @staticmethod
    def _generate_tokens(
        decoder_session,
        encoder_outputs,
        inputs,
        token_config,
        model_config,
        request,
        tokenizer=None,
    ):
        return generator_module._generate_tokens(
            decoder_session,
            encoder_outputs,
            inputs,
            token_config,
            model_config,
            request,
            tokenizer,
        )

    @staticmethod
    def _build_generation_params(model_config, request):
        return build_generation_params(model_config, request)


def get_special_tokens_path(model_path: str, model_config: Any) -> str:
    """Helper to get special tokens file path.

    Args:
        model_path: Root model path
        model_config: Model configuration

    Returns:
        Path to special tokens JSON file
    """
    import os

    special_tokens_file = getattr(model_config, "special_tokens_file", "special_tokens.json")
    return os.path.join(model_path, special_tokens_file)
