# =============================================================================
# File: embedder.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Main SentenceTransformer class with public embedding API."""

import time
from typing import Any, Callable, List, Optional, cast

from app.exceptions import InferenceError, ModelNotFoundError
from app.logger import get_logger
from app.models.embedding_request import EmbeddingBatchRequest, EmbeddingRequest
from app.models.embedding_response import EmbeddingBatchResponse, EmbeddingResponse
from app.services.base_nlp_service import BaseNLPService
from app.services.embedder import inference, resource_manager
from app.services.embedder.models import ChunkEmbeddingResult

logger = get_logger("embedder_service")


class SentenceTransformer(BaseNLPService):
    """Static class for sentence embedding using ONNX models."""

    # Legacy static methods for backward compatibility with existing code
    _merge_vectors = staticmethod(
        lambda chunks, method="mean": __import__(
            "app.services.embedder.text_utils", fromlist=["merge_vectors"]
        ).merge_vectors(chunks, method)
    )
    _merge_vectors = _merge_vectors
    # legacy loader removed — keep explicit `_preprocess_text` implementation below

    @staticmethod
    def _truncate_text_to_token_limit(text: str, tokenizer: Any, max_tokens: int = 128) -> str:
        """Backward compatibility for tests - truncates text by character count."""
        from app.services.embedder.text_utils import preprocess_text

        return preprocess_text(text)[: max_tokens * 4]

    @staticmethod
    def _preprocess_text(text, lowercase=True, remove_emojis=False):
        """Preprocess text with configurable options."""
        from app.services.embedder.text_utils import preprocess_text

        return preprocess_text(text, lowercase, remove_emojis)

    @staticmethod
    def _prepare_text_for_embedding(text, model_config):
        """Prepare text for embedding using model config."""
        lowercase = getattr(model_config, "lowercase", True)
        remove_emojis = getattr(model_config, "remove_emojis", False)
        return SentenceTransformer._preprocess_text(text, lowercase, remove_emojis)

    _prepare_onnx_inputs = staticmethod(
        lambda processed_text, tokenizer, model_config, session: __import__(
            "app.services.embedder.onnx_utils", fromlist=["prepare_onnx_inputs"]
        ).prepare_onnx_inputs(processed_text, tokenizer, model_config, session)
    )
    _add_optional_inputs = staticmethod(
        lambda inputs, input_names_config, model_input_names, session, seq_length=None: __import__(
            "app.services.embedder.onnx_utils", fromlist=["_add_optional_inputs"]
        )._add_optional_inputs(inputs, input_names_config, model_input_names, session, seq_length)
    )
    _log_onnx_outputs = staticmethod(
        lambda outputs, session: __import__(
            "app.services.embedder.onnx_utils", fromlist=["log_onnx_outputs"]
        ).log_onnx_outputs(outputs, session)
    )
    _process_embedding_output = staticmethod(
        lambda hidden_states, model_config, attention_mask, projected_dimension, **kwargs: __import__(
            "app.services.embedder.processing", fromlist=["process_embedding_output"]
        ).process_embedding_output(
            hidden_states, model_config, attention_mask, projected_dimension, **kwargs
        )
    )

    @staticmethod
    def _project_embedding(embedding, projected_dimension, prefetched_proj_matrix=None):
        """Backward compatibility wrapper for _project_embedding."""
        from app.services.embedder.processing import project_embedding

        native_dimension = embedding.shape[-1]
        return project_embedding(
            embedding, native_dimension, projected_dimension, prefetched_proj_matrix
        )

    _quantize_embedding = staticmethod(
        lambda embedding, quantization_type: __import__(
            "app.services.embedder.processing", fromlist=["quantize_embedding"]
        ).quantize_embedding(embedding, quantization_type)
    )
    _small_text_embedding = staticmethod(
        lambda small_text, model_config, tokenizer, session, projected_dimension, **kwargs: __import__(
            "app.services.embedder.inference", fromlist=["small_text_embedding"]
        ).small_text_embedding(
            small_text, model_config, tokenizer, session, projected_dimension, **kwargs
        )
    )
    _prepare_text_chunks = staticmethod(
        lambda text, tokenizer, model_config: __import__(
            "app.services.embedder.inference", fromlist=["prepare_text_chunks"]
        ).prepare_text_chunks(text, tokenizer, model_config)
    )
    _process_text_chunks = staticmethod(
        lambda chunks, model_config, tokenizer, session, projected_dimension, **kwargs: __import__(
            "app.services.embedder.inference", fromlist=["process_text_chunks"]
        ).process_text_chunks(
            chunks, model_config, tokenizer, session, projected_dimension, **kwargs
        )
    )
    _process_single_chunk = staticmethod(
        lambda chunk, model_config, tokenizer, session, projected_dimension, **kwargs: __import__(
            "app.services.embedder.inference", fromlist=["process_single_chunk"]
        ).process_single_chunk(
            chunk, model_config, tokenizer, session, projected_dimension, **kwargs
        )
    )
    _join_chunk_embeddings = staticmethod(
        lambda chunk_results, original_text, join_by_pooling_strategy, model_config, output_large_text_upon_join: __import__(
            "app.services.embedder.inference", fromlist=["join_chunk_embeddings"]
        ).join_chunk_embeddings(
            chunk_results,
            original_text,
            join_by_pooling_strategy,
            model_config,
            output_large_text_upon_join,
        )
    )
    _get_prefetched_proj_matrix = staticmethod(
        lambda model_config, projected_dimension: __import__(
            "app.services.embedder.processing", fromlist=["get_prefetched_proj_matrix"]
        ).get_prefetched_proj_matrix(model_config, projected_dimension)
    )
    _prepare_embedding_resources = staticmethod(
        lambda model: resource_manager.prepare_embedding_resources(model, SentenceTransformer)
    )
    _override_config_with_request = staticmethod(resource_manager.override_config_with_request)

    # Override _get_model_path to use single-argument signature (backward compatibility)
    @staticmethod
    def _get_model_path(model, model_config=None):
        """Get model path - backward compatible with both old and new signatures."""
        # Delegate to BaseNLPService which takes only model name
        return BaseNLPService._get_model_path(model)

    _load_tokenizer = staticmethod(
        lambda model_to_use_path, model_config: resource_manager.load_tokenizer(
            model_to_use_path, model_config, SentenceTransformer
        )
    )
    _load_session = staticmethod(
        lambda model_to_use_path, model_config: resource_manager.load_session(
            model_to_use_path, model_config, SentenceTransformer
        )
    )
    _get_native_dimension_from_session = staticmethod(
        lambda session: __import__(
            "app.services.embedder.onnx_utils",
            fromlist=["get_native_dimension_from_session"],
        ).get_native_dimension_from_session(session)
    )
    _get_output_names_from_session = staticmethod(
        lambda session: __import__(
            "app.services.embedder.onnx_utils",
            fromlist=["get_output_names_from_session"],
        ).get_output_names_from_session(session)
    )
    _get_embedding_model_path = staticmethod(
        lambda model_to_use_path, model_config: resource_manager.get_embedding_model_path(
            model_to_use_path, model_config, SentenceTransformer
        )
    )
    _get_model_filename = staticmethod(
        cast(Callable[..., str], resource_manager.get_model_filename)
    )
    _can_use_batch_inference = staticmethod(
        lambda texts, model: inference.can_use_batch_inference(
            texts, model, SentenceTransformer._get_model_config
        )
    )
    _embed_batch_texts = staticmethod(
        lambda texts, model, projected_dimension, request_params=None, **kwargs: SentenceTransformer._embed_batch_texts_impl(
            texts, model, projected_dimension, request_params, **kwargs
        )
    )

    @staticmethod
    def _embed_batch_texts_impl(
        texts: List[str],
        model: str,
        projected_dimension: Optional[int],
        request_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> ChunkEmbeddingResult:
        """Implementation of batch embedding logic."""
        model_config, tokenizer, session = resource_manager.prepare_embedding_resources(
            model, SentenceTransformer
        )

        # Override config with request parameters
        if request_params:
            model_config = resource_manager.override_config_with_request(
                model_config, request_params
            )

        return inference.embed_batch_texts(
            texts, model_config, tokenizer, session, projected_dimension, **kwargs
        )

    @staticmethod
    def embed_text(request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for text with optional chunking.

        Main entry point for text embedding with full parameter support.

        Args:
            request: EmbeddingRequest with text and embedding parameters

        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        response = EmbeddingResponse(
            results=[], success=True, model=request.model, message="", time_taken=0.0
        )
        start_time = time.time()

        try:
            # Validate model capabilities (with fallback for older BaseNLPService)
            _validate = getattr(BaseNLPService, "_validate_model_capability", None)
            if _validate is not None:
                if not _validate(request.model, "embedding"):
                    response.success = False
                    response.message = f"Model '{request.model}' does not support embeddings"
                    return response

            # Extract request parameters
            pooling_strategy = request.pooling_strategy
            max_length = request.max_length
            chunk_logic = request.chunk_logic
            chunk_overlap = request.chunk_overlap
            chunk_size = request.chunk_size
            join_chunks = request.join_chunks
            join_by_pooling_strategy = request.join_by_pooling_strategy
            output_large_text_upon_join = request.output_large_text_upon_join
            legacy_tokenizer = request.legacy_tokenizer
            projected_dimension = request.projected_dimension
            normalize = request.normalize
            force_pooling = request.force_pooling
            lowercase = request.lowercase
            remove_emojis = request.remove_emojis
            request_params = {
                "pooling_strategy": pooling_strategy,
                "max_length": max_length,
                "chunk_logic": chunk_logic,
                "chunk_overlap": chunk_overlap,
                "chunk_size": chunk_size,
                "legacy_tokenizer": legacy_tokenizer,
                "normalize": normalize,
                "force_pooling": force_pooling,
                "lowercase": lowercase,
                "remove_emojis": remove_emojis,
            }

            result = SentenceTransformer._embed_text_local(
                request.input,  # Changed from request.text to request.input
                request.model,
                projected_dimension,
                join_chunks,
                join_by_pooling_strategy,
                output_large_text_upon_join,
                request_params,
            )

            if not result.success:
                response.success = False
                response.message = result.message
            else:
                response.results = result.embedding_chunks
                response.used_parameters = getattr(result, "used_parameters", {})
                response.warnings = getattr(result, "warnings", [])

        except ModelNotFoundError as e:
            response.success = False
            response.message = str(e)
        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
        # finalize timing and return outside finally to avoid silencing exceptions (B012)
        response.time_taken = time.time() - start_time
        return response

    @staticmethod
    def embed_batch_async(request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
        """Generate embeddings for multiple texts with batch optimization.

        Args:
            request: EmbeddingBatchRequest with list of texts

        Returns:
            EmbeddingBatchResponse with all embeddings
        """
        response = EmbeddingBatchResponse(
            results=[], success=True, model=request.model, message="", time_taken=0.0
        )
        start_time = time.time()

        try:
            # Validate model capabilities
            _validate = getattr(BaseNLPService, "_validate_model_capability", None)
            if _validate is not None:
                if not _validate(request.model, "embedding"):
                    response.success = False
                    response.message = f"Model '{request.model}' does not support embeddings"
                    return response

            # Extract common parameters
            projected_dimension = request.projected_dimension
            pooling_strategy = request.pooling_strategy
            max_length = request.max_length
            join_chunks = request.join_chunks
            join_by_pooling_strategy = request.join_by_pooling_strategy
            output_large_text_upon_join = request.output_large_text_upon_join

            request_params = {
                "pooling_strategy": pooling_strategy,
                "max_length": max_length,
                "chunk_logic": request.chunk_logic,
                "chunk_overlap": request.chunk_overlap,
                "chunk_size": request.chunk_size,
                "legacy_tokenizer": request.legacy_tokenizer,
                "normalize": request.normalize,
                "force_pooling": request.force_pooling,
                "lowercase": request.lowercase,
                "remove_emojis": request.remove_emojis,
                # 'use_optimized' removed: not supported in requests anymore
            }

            # Try batch inference if possible
            if SentenceTransformer._can_use_batch_inference(
                request.inputs,
                request.model,  # Changed from request.texts to request.inputs
            ):
                try:
                    logger.info(
                        "Using optimized batch inference for %d texts",
                        len(request.inputs),
                    )
                    result = SentenceTransformer._embed_batch_texts(
                        request.inputs,  # Changed from request.texts to request.inputs
                        request.model,
                        projected_dimension,
                        request_params,
                    )
                    response.results = result.embedding_chunks
                    response.used_parameters = result.used_parameters
                    response.warnings = result.warnings
                except InferenceError as e:
                    logger.warning(
                        f"Batch inference failed: {e}, falling back to sequential processing"
                    )
                    # Fall through to sequential processing
                else:
                    # Batch inference succeeded
                    response.time_taken = time.time() - start_time
                    return response

            # Sequential processing fallback
            logger.info("Using sequential processing for %d texts", len(request.inputs))
            for idx, text_input in enumerate(
                request.inputs
            ):  # Changed from request.texts to request.inputs
                result = SentenceTransformer._embed_text_local(
                    text_input,
                    request.model,
                    projected_dimension,
                    join_chunks,
                    join_by_pooling_strategy,
                    output_large_text_upon_join,
                    request_params,
                )

                if not result.success:
                    response.success = False
                    response.message = f"Error in input {idx}: {result.message}"
                    break
                else:
                    response.results.extend(result.embedding_chunks)
                    if idx == 0:  # Set used_parameters and warnings from first result
                        response.used_parameters = getattr(result, "used_parameters", {})
                        response.warnings = getattr(result, "warnings", [])

        except ModelNotFoundError as e:
            response.success = False
            response.message = str(e)
        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
        # finalize timing and return outside finally to avoid silencing exceptions (B012)
        response.time_taken = time.time() - start_time
        return response

    @staticmethod
    def _embed_text_local(
        text: str,
        model: str,
        projected_dimension: Optional[int],
        join_chunks: Optional[bool] = False,
        join_by_pooling_strategy: Optional[str] = None,
        output_large_text_upon_join: Optional[bool] = False,
        request_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> ChunkEmbeddingResult:
        """Core embedding logic for text processing.

        Args:
            text: Input text to embed
            model: Model name
            projected_dimension: Target dimension (None uses native)
            join_chunks: Whether to merge chunk embeddings
            join_by_pooling_strategy: Pooling strategy for joining
            output_large_text_upon_join: Include full text in joined result
            request_params: Request parameter overrides
            **kwargs: Additional parameters

        Returns:
            ChunkEmbeddingResult with embeddings and metadata
        """
        try:
            model_config, tokenizer, session = resource_manager.prepare_embedding_resources(
                model, SentenceTransformer
            )

            # Override config with request parameters if provided
            if request_params:
                model_config = resource_manager.override_config_with_request(
                    model_config, request_params
                )

            chunks = inference.prepare_text_chunks(text, tokenizer, model_config)

            # Process chunks
            chunk_results = inference.process_text_chunks(
                chunks, model_config, tokenizer, session, projected_dimension, **kwargs
            )

            # Join chunks if requested
            should_join = join_chunks and len(chunk_results) > 1
            if should_join:
                chunk_results = inference.join_chunk_embeddings(
                    chunk_results,
                    text,
                    join_by_pooling_strategy,
                    model_config,
                    output_large_text_upon_join,
                )

            # Extract used parameters from final config
            used_params = {
                "pooling_strategy": getattr(model_config, "pooling_strategy", "mean"),
                "projected_dimension": (
                    projected_dimension
                    if projected_dimension is not None
                    else getattr(model_config, "dimension", None)
                ),
                "dimension_used": getattr(model_config, "dimension", None),
                "max_length": getattr(model_config, "max_length", 256),
                "chunk_logic": getattr(model_config, "chunk_logic", "sentence"),
                "chunk_overlap": getattr(model_config, "chunk_overlap", 1),
                "chunk_size": getattr(model_config, "chunk_size", None),
                "normalize": getattr(model_config, "normalize", True),
                "force_pooling": getattr(model_config, "force_pooling", False),
                "legacy_tokenizer": getattr(model_config, "legacy_tokenizer", False),
                "remove_emojis": getattr(model_config, "remove_emojis", False),
                "lowercase": getattr(model_config, "lowercase", False),
            }

            # Collect warnings
            warnings = []
            if hasattr(model_config, "_dimension_warning"):
                warnings.append(model_config._dimension_warning)

            return ChunkEmbeddingResult(
                embedding_chunks=chunk_results,
                message="Embedding generated successfully",
                success=True,
                used_parameters=used_params,
                warnings=warnings,
            )

        except FileNotFoundError:
            raise ModelNotFoundError("Model files not accessible")
        except OSError as e:
            from app.exceptions import ModelLoadError

            raise ModelLoadError(f"System error accessing model: {e}")
        except (ValueError, KeyError) as e:
            raise InferenceError(f"Invalid model configuration: {e}")
        except Exception as e:
            raise InferenceError(f"Error generating embedding: {e}")
