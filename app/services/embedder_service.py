# =============================================================================
# File: embedder_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# Removed asyncio imports to prevent hanging
import functools
import os
import re
import time
import unicodedata
from typing import Any, List

import numpy as np
from numpy import ndarray
from pydantic import BaseModel, Field

from app.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
    TokenizerError,
)
from app.logger import get_logger
from app.models.embedded_chunk import EmbededChunk
from app.models.embedding_request import EmbeddingBatchRequest, EmbeddingRequest
from app.models.embedding_response import EmbeddingBatchResponse, EmbeddingResponse
from app.services.base_nlp_service import BaseNLPService
from app.utils.batch_limiter import BatchLimiter
from app.utils.chunking_strategies import ChunkingStrategies
from app.utils.error_handler import ErrorHandler, handle_errors
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.path_validator import validate_safe_path
from app.utils.pooling_strategies import PoolingStrategies

logger = get_logger("embedder_service")

# Constants
DEFAULT_MAX_LENGTH = 128
DEFAULT_PROJECTED_DIMENSION = 128
DEFAULT_BATCH_SIZE = 50
RANDOM_SEED = 42


class _EmbeddingResults(BaseModel):
    EmbeddingResults: List[float]  # For _small_text_embedding, this is a list of floats
    message: str
    success: bool = Field(default=True)


class _ChunkEmbeddingResults(BaseModel):
    EmbeddingResults: List[
        EmbededChunk
    ]  # For chunk processing, this is a list of chunks
    message: str
    success: bool = Field(default=True)


class SentenceTransformer(BaseNLPService):
    """Static class for sentence embedding using ONNX models."""

    @staticmethod
    def _merge_vectors(chunks: List[EmbededChunk], method: str = "mean") -> List[float]:
        """Merge embedding vectors using mean or max pooling."""
        vectors = [
            np.array(chunk.vector) for chunk in chunks if hasattr(chunk, "vector")
        ]
        if not vectors:
            return []

        # Validate pooling method
        if method not in ["mean", "max"]:
            method = "mean"

        stacked = np.stack(vectors)
        merged = (
            np.max(stacked, axis=0) if method == "max" else np.mean(stacked, axis=0)
        )
        return merged.tolist()

    @staticmethod
    def _preprocess_text(
        text: str, lowercase: bool = True, remove_emojis: bool = False
    ) -> str:
        """Clean and normalize raw text for embedding."""

        # Normalize Unicode characters (e.g. curly quotes, accented letters)
        text = unicodedata.normalize("NFKC", text)

        # Remove HTML or XML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Replace all types of whitespace (tabs, line breaks, multiple spaces) with single space
        text = re.sub(r"\s+", " ", text)

        # Optional: Remove emojis and non-ASCII characters
        if remove_emojis:
            text = re.sub(r"[^\x00-\x7F]+", "", text)

        # Optional: Convert to lowercase
        if lowercase:
            text = text.lower()

        # Final cleanup: trim leading and trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def _small_text_embedding(
        small_text: str,
        model_config: Any,
        tokenizer: Any,
        session: Any,
        projected_dimension: int = DEFAULT_PROJECTED_DIMENSION,
        **kwargs,
    ) -> _EmbeddingResults:
        """Generate embedding for a single text chunk."""
        try:
            # Tokenize and prepare inputs
            processed_text = SentenceTransformer._prepare_text_for_embedding(
                small_text, model_config
            )
            inputs = SentenceTransformer._prepare_onnx_inputs(
                processed_text, tokenizer, model_config, session
            )

            # Run inference
            outputs = session.run(None, inputs)
            SentenceTransformer._log_onnx_outputs(outputs, session)

            # Process embedding
            embedding = SentenceTransformer._process_embedding_output(
                outputs[0],
                model_config,
                inputs.get("attention_mask"),
                projected_dimension,
                **kwargs,
            )

            return _EmbeddingResults(
                EmbeddingResults=embedding.flatten().tolist(),
                message="Embedding generated successfully",
                success=True,
            )

        except (ValueError, TypeError) as e:
            raise InferenceError(f"Invalid embedding parameters: {e}")
        except (RuntimeError, OSError) as e:
            raise InferenceError(f"Model inference failed: {e}")
        except Exception as e:
            raise InferenceError(f"Error generating embedding: {e}")

    @staticmethod
    def _prepare_text_for_embedding(text: str, model_config: Any) -> str:
        """Prepare text for embedding processing."""
        lowercase = getattr(model_config, "lowercase", True)
        remove_emojis = getattr(model_config, "remove_emojis", False)
        return SentenceTransformer._preprocess_text(text, lowercase, remove_emojis)

    @staticmethod
    def _prepare_onnx_inputs(
        processed_text: str, tokenizer: Any, model_config: Any, session: Any
    ) -> dict:
        """Prepare ONNX inputs for inference."""
        input_names = getattr(model_config, "inputnames", {})
        max_length = getattr(input_names, "max_length", DEFAULT_MAX_LENGTH)

        encoding = tokenizer(
            processed_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding.get("attention_mask")

        inputs = {getattr(input_names, "input", "input_ids"): input_ids}

        if attention_mask is not None:
            inputs[getattr(input_names, "mask", "attention_mask")] = (
                attention_mask.astype(np.int64)
            )

        # Add optional inputs
        SentenceTransformer._add_optional_inputs(
            inputs, input_names, session, input_ids.shape[1]
        )

        return inputs

    @staticmethod
    def _add_optional_inputs(
        inputs: dict, input_names: Any, session: Any, seq_len: int
    ):
        """Add optional inputs like position_ids, token_type_ids, decoder_input_ids."""
        # Position IDs
        position_name = getattr(input_names, "position", None)
        if position_name:
            inputs[position_name] = np.arange(seq_len, dtype=np.int64)[None, :]

        # Token type IDs
        required_inputs = [inp.name for inp in session.get_inputs()]
        tokentype_name = getattr(input_names, "tokentype", None)
        if not tokentype_name:
            tokentype_name = "token_type_ids"
        if tokentype_name in required_inputs:
            inputs[tokentype_name] = np.zeros((1, seq_len), dtype=np.int64)

        # Decoder input IDs
        if getattr(input_names, "use_decoder_input", False):
            decoder_input_name = getattr(
                input_names, "decoder_input_name", "decoder_input_ids"
            )
            inputs[decoder_input_name] = np.zeros((1, seq_len), dtype=np.int64)

    @staticmethod
    def _process_embedding_output(
        embedding: ndarray,
        model_config: Any,
        attention_mask: ndarray,
        projected_dimension: int,
        **kwargs,
    ) -> ndarray:
        """Process embedding output with pooling, normalization, and projection."""
        output_names = getattr(model_config, "outputnames", {})

        # Apply softmax if logits
        is_logits = getattr(output_names, "logits", False)
        if is_logits or SentenceTransformer._is_logits_output([embedding], None):
            embedding = SentenceTransformer._softmax(embedding)

        # Apply pooling
        pooling_strategy = getattr(model_config, "pooling_strategy", "mean")
        force_pooling = getattr(model_config, "force_pooling", False)

        logger.debug(
            "Applying pooling: strategy=%s, force=%s, shape=%s",
            sanitize_for_log(pooling_strategy),
            sanitize_for_log(str(force_pooling)),
            sanitize_for_log(str(embedding.shape)),
        )

        embedding = PoolingStrategies.apply(
            embedding, pooling_strategy, attention_mask, force_pooling
        )
        logger.debug("Pooling result shape: %s", sanitize_for_log(str(embedding.shape)))

        # Normalize
        normalize = kwargs.get("normalize", getattr(model_config, "normalize", True))
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Project dimensions
        if projected_dimension > 0 and embedding.shape[-1] != projected_dimension:
            embedding = SentenceTransformer._project_embedding(
                embedding, projected_dimension
            )

        return embedding

    @staticmethod
    def _project_embedding(
        embedding: np.ndarray, projected_dimension: int
    ) -> np.ndarray:
        """Project embedding to target dimension using fixed random matrix."""
        input_dim = embedding.shape[-1]
        rng = np.random.default_rng(seed=RANDOM_SEED)
        random_matrix = rng.uniform(-1, 1, (input_dim, projected_dimension))
        return np.dot(embedding, random_matrix)

    @staticmethod
    def _truncate_text_to_token_limit(
        text: str, tokenizer: Any, max_tokens: int = DEFAULT_MAX_LENGTH
    ) -> str:
        """Backward compatibility for tests."""
        return SentenceTransformer._preprocess_text(text, True, False)[: max_tokens * 4]

    @staticmethod
    @handle_errors(context="embedding", include_traceback=True)
    def embed_text(req: EmbeddingRequest, **kwargs: Any) -> EmbeddingResponse:
        """Main embedding function for single text."""
        start_time = time.time()
        response = EmbeddingResponse(
            success=True,
            message="Embedding generated successfully",
            model=req.model,
            results=[],
            time_taken=0.0,
        )

        try:
            result = SentenceTransformer._embed_text_local(
                text=req.input,
                model=req.model,
                projected_dimension=req.projected_dimension,
                join_chunks=req.join_chunks,
                join_by_pooling_strategy=req.join_by_pooling_strategy,
                output_large_text_upon_join=req.output_large_text_upon_join,
                **kwargs,
            )

            if result.success:
                response.results = result.EmbeddingResults
            else:
                response.success = False
                response.message = result.message

        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
        finally:
            response.time_taken = time.time() - start_time
            return response

    @staticmethod
    async def embed_batch_async(
        requests: EmbeddingBatchRequest, **kwargs: Any
    ) -> EmbeddingBatchResponse:
        """Synchronous batch embedding (async wrapper)."""
        start_time = time.time()
        response = EmbeddingBatchResponse(
            success=True,
            message="Batch embedding generated successfully",
            model=requests.model,
            results=[],
            time_taken=0.0,
        )

        try:
            BatchLimiter.validate_batch_size(
                requests.inputs, max_size=DEFAULT_BATCH_SIZE
            )

            for idx, input_text in enumerate(requests.inputs):
                result = SentenceTransformer._embed_text_local(
                    text=input_text,
                    model=requests.model,
                    projected_dimension=requests.projected_dimension,
                    join_chunks=requests.join_chunks,
                    join_by_pooling_strategy=requests.join_by_pooling_strategy,
                    output_large_text_upon_join=requests.output_large_text_upon_join,
                    **kwargs,
                )

                if not result.success:
                    response.success = False
                    response.message = f"Error in input {idx}: {result.message}"
                    break
                else:
                    response.results.extend(result.EmbeddingResults)

        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
        finally:
            response.time_taken = time.time() - start_time
            return response

    @staticmethod
    def _embed_text_local(
        text: str,
        model: str,
        projected_dimension: int,
        join_chunks: bool = False,
        join_by_pooling_strategy: str = None,
        output_large_text_upon_join: bool = False,
        **kwargs: Any,
    ) -> _ChunkEmbeddingResults:
        """Core embedding logic for text processing."""
        try:
            model_config, tokenizer, session = (
                SentenceTransformer._prepare_embedding_resources(model)
            )
            chunks = SentenceTransformer._prepare_text_chunks(
                text, tokenizer, model_config
            )

            # Process chunks
            chunk_results = SentenceTransformer._process_text_chunks(
                chunks, model_config, tokenizer, session, projected_dimension, **kwargs
            )

            # Join chunks if requested
            should_join = join_chunks and len(chunk_results) > 1
            if should_join:
                chunk_results = SentenceTransformer._join_chunk_embeddings(
                    chunk_results,
                    text,
                    join_by_pooling_strategy,
                    model_config,
                    output_large_text_upon_join,
                )

            return _ChunkEmbeddingResults(
                EmbeddingResults=chunk_results,
                message="Embedding generated successfully",
                success=True,
            )

        except FileNotFoundError:
            raise ModelNotFoundError("Model files not accessible")
        except OSError as e:
            raise ModelLoadError(f"System error accessing model: {e}")
        except (ValueError, KeyError) as e:
            raise InferenceError(f"Invalid model configuration: {e}")
        except Exception as e:
            raise InferenceError(f"Error generating embedding: {e}")

    @staticmethod
    def _prepare_embedding_resources(model: str) -> tuple[Any, Any, Any]:
        """Prepare model config, tokenizer, and session for embedding."""
        model_config = SentenceTransformer._get_model_config(model)
        model_to_use_path = SentenceTransformer._get_model_path(model, model_config)
        tokenizer = SentenceTransformer._load_tokenizer(model_to_use_path, model_config)
        session = SentenceTransformer._load_session(model_to_use_path, model_config)
        return model_config, tokenizer, session

    @staticmethod
    def _get_model_path(model: str, model_config: Any) -> str:
        """Get validated model path."""
        return validate_safe_path(
            os.path.join(
                SentenceTransformer._root_path,
                "models",
                getattr(model_config, "embedder_task", "fe"),
                model,
            ),
            SentenceTransformer._root_path,
        )

    @staticmethod
    def _load_tokenizer(model_to_use_path: str, model_config: Any) -> Any:
        """Load and validate tokenizer."""
        use_legacy = getattr(model_config, "legacy_tokenizer", False)
        tokenizer = SentenceTransformer._get_tokenizer_threadsafe(
            model_to_use_path, use_legacy
        )
        if not tokenizer:
            raise TokenizerError(f"Failed to load tokenizer: {model_to_use_path}")
        return tokenizer

    @staticmethod
    def _load_session(model_to_use_path: str, model_config: Any) -> Any:
        """Load and validate ONNX session."""
        model_path = SentenceTransformer._get_embedding_model_path(
            model_to_use_path, model_config
        )
        session = SentenceTransformer._get_encoder_session(model_path)
        if not session:
            raise ModelLoadError(f"Failed to load ONNX session: {model_path}")
        return session

    @staticmethod
    def _get_embedding_model_path(model_to_use_path: str, model_config: Any) -> str:
        """Get the path to the embedding model file."""
        model_filename = SentenceTransformer._get_model_filename(
            model_config, is_embedding=True
        )
        return validate_safe_path(
            os.path.join(model_to_use_path, model_filename),
            SentenceTransformer._root_path,
        )

    @staticmethod
    def _get_model_filename(model_config: Any, is_embedding: bool = True) -> str:
        """Get model filename based on optimization settings."""
        use_optimized = getattr(model_config, "use_optimized", False)

        if is_embedding:
            if use_optimized:
                return getattr(
                    model_config, "encoder_optimized_onnx_model", "model_optimized.onnx"
                )
            else:
                return getattr(model_config, "encoder_onnx_model", None) or "model.onnx"
        else:
            # For decoder models
            if use_optimized:
                return getattr(
                    model_config,
                    "decoder_optimized_onnx_model",
                    "decoder_model_optimized.onnx",
                )
            else:
                return getattr(model_config, "decoder_onnx_model", "decoder_model.onnx")

    @staticmethod
    def _prepare_text_chunks(text: str, tokenizer: Any, model_config: Any) -> List[str]:
        """Prepare text chunks for processing."""
        max_tokens = getattr(
            getattr(model_config, "inputnames", {}), "max_length", DEFAULT_MAX_LENGTH
        )
        return ChunkingStrategies.split_text_into_chunks(
            text, tokenizer, max_tokens, model_config
        )

    @staticmethod
    def _process_text_chunks(
        chunks: List[str],
        model_config: Any,
        tokenizer: Any,
        session: Any,
        projected_dimension: int,
        **kwargs,
    ) -> List[EmbededChunk]:
        """Process text chunks into embeddings."""
        results = []
        for chunk in chunks:
            embedding_result = SentenceTransformer._process_single_chunk(
                chunk, model_config, tokenizer, session, projected_dimension, **kwargs
            )
            results.append(
                EmbededChunk(vector=embedding_result.EmbeddingResults, chunk=chunk)
            )
        return results

    @staticmethod
    def _process_single_chunk(
        chunk: str,
        model_config: Any,
        tokenizer: Any,
        session: Any,
        projected_dimension: int,
        **kwargs,
    ) -> Any:
        """Process a single text chunk into embedding."""
        embedding_result = SentenceTransformer._small_text_embedding(
            small_text=chunk,
            model_config=model_config,
            tokenizer=tokenizer,
            session=session,
            projected_dimension=projected_dimension,
            **kwargs,
        )
        if not embedding_result.success:
            raise InferenceError(embedding_result.message)
        return embedding_result

    @staticmethod
    def _join_chunk_embeddings(
        chunk_results: List[EmbededChunk],
        original_text: str,
        join_by_pooling_strategy: str,
        model_config: Any,
        output_large_text_upon_join: bool,
    ) -> List[EmbededChunk]:
        """Join chunk embeddings using specified pooling strategy."""
        pooling_strategy = join_by_pooling_strategy or getattr(
            model_config, "pooling_strategy", "mean"
        )

        if pooling_strategy not in ["mean", "max", "first", "last"]:
            pooling_strategy = "mean"

        merged_vector = SentenceTransformer._merge_vectors(
            chunk_results, pooling_strategy
        )

        return [
            EmbededChunk(
                vector=merged_vector,
                joined_chunk=True,
                only_vector=not output_large_text_upon_join,
                chunk="" if not output_large_text_upon_join else original_text,
            )
        ]
