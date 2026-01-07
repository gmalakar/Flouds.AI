# =============================================================================
# File: embedder_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# Removed asyncio imports to prevent hanging

import logging
import os
import re
import time
import unicodedata
from typing import Any, List, Optional, cast

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
from app.modules.concurrent_dict import ConcurrentDict
from app.services.base_nlp_service import BaseNLPService
from app.utils.batch_limiter import BatchLimiter
from app.utils.chunking_strategies import ChunkingStrategies
from app.utils.log_sanitizer import sanitize_for_log
from app.utils.path_validator import validate_safe_path
from app.utils.pooling_strategies import PoolingStrategies

logger = get_logger("embedder_service")

# Constants
DEFAULT_MAX_LENGTH = 256
DEFAULT_PROJECTED_DIMENSION = 128
DEFAULT_BATCH_SIZE = 50
from app.utils.constants import NORM_EPS, RANDOM_SEED

# Projection matrices are stored in the central cache registry to allow
# cache manager control and centralized clearing under memory pressure.


class SingleEmbeddingResult(BaseModel):
    vector: List[float]
    message: str
    success: bool = Field(default=True)

    @property
    def embedding_results(self) -> List[float]:
        """Backward compatibility property."""
        return self.vector

    # Legacy property for existing tests
    EmbeddingResults = embedding_results


class ChunkEmbeddingResult(BaseModel):
    embedding_chunks: List[
        EmbededChunk
    ]  # For chunk processing, this is a list of chunks
    message: str
    success: bool = Field(default=True)
    used_parameters: dict = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    @property
    def embedding_results(self) -> List[EmbededChunk]:
        """Backward compatibility property."""
        return self.embedding_chunks

    # Legacy property for existing tests
    EmbeddingResults = embedding_results


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
    ) -> SingleEmbeddingResult:
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

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Final embedding shape before flatten: %s", embedding.shape
                )
            flattened = embedding.flatten()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Final embedding shape after flatten: %s", flattened.shape)

            return SingleEmbeddingResult(
                vector=flattened.tolist(),
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
        """Prepare ONNX inputs for inference with auto-detected input names."""
        max_length = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)

        encoding = tokenizer(
            processed_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask")

        # Ensure arrays have batch dimension (ONNX expects batch-first tensors)
        def _ensure_batch_dim(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.ndim == 1:
                return arr[None, :]
            return arr

        input_ids = _ensure_batch_dim(input_ids)
        if attention_mask is not None:
            attention_mask = _ensure_batch_dim(attention_mask)

        # Ensure correct dtype without unnecessary copying
        if input_ids.dtype != np.int64:
            input_ids = input_ids.astype(np.int64)
        if attention_mask is not None and attention_mask.dtype != np.int64:
            attention_mask = attention_mask.astype(np.int64)

        # Auto-detect input names from ONNX model with config override
        model_input_names = [inp.name for inp in session.get_inputs()]
        model_input_names_lc = [n.lower() for n in model_input_names]
        input_names_config = getattr(model_config, "inputnames", {})

        # Local helper: try exact config override, then exact match, case-insensitive, then substring
        def _select_name(
            config_name: Optional[str],
            candidates: List[str],
            default: Optional[str] = None,
        ) -> Optional[str]:
            # Config override
            if config_name:
                return config_name
            # Exact matches
            for cand in candidates:
                if cand in model_input_names:
                    return cand
            # Case-insensitive matches
            for cand in candidates:
                lc = cand.lower()
                if lc in model_input_names_lc:
                    return model_input_names[model_input_names_lc.index(lc)]
            # Substring matches
            for cand in candidates:
                lc = cand.lower()
                for name, name_lc in zip(model_input_names, model_input_names_lc):
                    if lc in name_lc:
                        return name
            return default

        # Map tokenizer outputs to ONNX inputs
        # Priority: config override > auto-detection > defaults
        input_id_name = _select_name(
            getattr(input_names_config, "input", None),
            ["input_ids", "input"],
            "input_ids",
        )
        inputs = {input_id_name: input_ids}
        logger.debug(f"Using input name '{input_id_name}' for input_ids")

        if attention_mask is not None:
            mask_name = _select_name(
                getattr(input_names_config, "mask", None),
                ["attention_mask", "mask"],
                None,
            )
            if mask_name:
                inputs[mask_name] = attention_mask
                logger.debug(f"Using input name '{mask_name}' for attention_mask")

        # Add optional inputs (position_ids, token_type_ids, decoder inputs)
        SentenceTransformer._add_optional_inputs(
            inputs, input_names_config, model_config, session, input_ids.shape[1]
        )

        return inputs

    @staticmethod
    def _find_matching_input(
        model_input_names: List[str], candidates: List[str]
    ) -> Optional[str]:
        """Find first matching input name from candidates in model inputs."""
        for candidate in candidates:
            if candidate in model_input_names:
                return candidate
        return None

    @staticmethod
    def _add_optional_inputs(
        inputs: dict,
        input_names_config: Any,
        model_config: Any,
        session: Any,
        seq_len: int,
    ):
        """Add optional inputs like position_ids, token_type_ids, decoder_input_ids."""
        model_input_names = [inp.name for inp in session.get_inputs()]

        # Position IDs - config override or auto-detect
        position_name = getattr(
            input_names_config, "position", None
        ) or SentenceTransformer._find_matching_input(
            model_input_names, ["position_ids"]
        )
        if position_name and position_name in model_input_names:
            inputs[position_name] = np.arange(seq_len, dtype=np.int64)[None, :]
            logger.debug(f"Added position_ids as '{position_name}'")

        # Token type IDs - config override or auto-detect
        tokentype_name = getattr(
            input_names_config, "tokentype", None
        ) or SentenceTransformer._find_matching_input(
            model_input_names, ["token_type_ids"]
        )
        if tokentype_name and tokentype_name in model_input_names:
            inputs[tokentype_name] = np.zeros((1, seq_len), dtype=np.int64)
            logger.debug(f"Added token_type_ids as '{tokentype_name}'")

        # Decoder input IDs - config override (special case for T5 models)
        use_decoder_input = getattr(input_names_config, "use_decoder_input", False)
        if use_decoder_input:
            decoder_names = getattr(
                model_config, "decoder_inputnames", input_names_config
            )
            decoder_input_name = getattr(
                decoder_names, "decoder_input_name", "decoder_input_ids"
            )
            if decoder_input_name in model_input_names:
                inputs[decoder_input_name] = np.zeros((1, seq_len), dtype=np.int64)
                logger.debug(f"Added decoder_input_ids as '{decoder_input_name}'")

    @staticmethod
    def _process_embedding_output(
        embedding: ndarray,
        model_config: Any,
        attention_mask: Optional[ndarray],
        projected_dimension: Optional[int],
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

        logger.debug(
            "Before pooling: strategy=%s, shape=%s",
            sanitize_for_log(pooling_strategy),
            sanitize_for_log(str(embedding.shape)),
        )

        # Normalize attention_mask shape to align with embedding for pooling.
        # PoolingStrategies expects masks shaped like embedding.shape[:2]
        # (batch, seq_len) for 3D embeddings, or no mask for 2D embeddings.
        def _normalize_mask(mask: Optional[ndarray], emb: ndarray) -> Optional[ndarray]:
            if mask is None:
                return None
            m = np.asarray(mask)
            # If mask is 1D and embedding is 3D, add batch dim
            if m.ndim == 1 and emb.ndim == 3:
                return m[None, :]
            # If mask is 2D and embedding is 2D (seq_len, dim) and mask has leading batch dim
            if m.ndim == 2 and emb.ndim == 2:
                # common case: mask shape (1, seq_len) -> squeeze
                if m.shape[0] == 1 and m.shape[1] == emb.shape[0]:
                    return m[0]
                # If mask shape equals (seq_len, dim) it's invalid for pooling; drop it
                return None
            # If mask is 2D and embedding is 3D, ensure shapes match else try to adapt
            if m.ndim == 2 and emb.ndim == 3:
                if m.shape[0] == emb.shape[0] and m.shape[1] == emb.shape[1]:
                    return m
                # If mask is (seq_len,) mistakenly shaped as (seq_len, 1) or transposed, try simple fixes
                if m.shape[0] == 1 and m.shape[1] == emb.shape[1]:
                    return m
                if m.shape[0] == emb.shape[1] and m.shape[1] == emb.shape[0]:
                    return m.T
                return None
            # If mask is 1D and embedding is 2D, assume it aligns with seq_len
            if m.ndim == 1 and emb.ndim == 2:
                return m
            return None

        normalized_mask = _normalize_mask(attention_mask, embedding)
        embedding = PoolingStrategies.apply(
            embedding, pooling_strategy, cast(ndarray, normalized_mask), force_pooling
        )
        logger.debug("After pooling: shape=%s", sanitize_for_log(str(embedding.shape)))

        # Project dimensions - only for downsampling (reducing dimensions)
        # Upsampling (increasing dimensions) would add noise without information gain
        logger.debug(
            "Before projection: shape=%s, dimension=%s",
            str(embedding.shape),
            str(projected_dimension),
        )
        current_dim = embedding.shape[-1]
        if (
            projected_dimension is not None
            and projected_dimension > 0
            and current_dim != projected_dimension
        ):
            if projected_dimension > current_dim:
                # Warn about upsampling - cannot create information from nothing
                logger.warning(
                    "Requested projected_dimension (%d) is larger than native dimension (%d). "
                    "Upsampling adds noise without information gain. Using native dimension instead.",
                    projected_dimension,
                    current_dim,
                )
            else:
                # Downsampling - reduces dimensions using random projection
                logger.debug(
                    "Applying projection (downsampling) from %s to %s",
                    str(current_dim),
                    str(projected_dimension),
                )
                # Vectorized projection: use a prefetched matrix if provided by
                # the caller to avoid repeated cache lookups per chunk. If not
                # provided or invalid, fall back to fetching/creating from the
                # central projection cache.
                prefetched = kwargs.pop("prefetched_proj_matrix", None)
                random_matrix = None
                if prefetched is not None:
                    # Validate shape and dtype of prefetched matrix
                    if (
                        getattr(prefetched, "shape", None)
                        == (current_dim, projected_dimension)
                        and getattr(prefetched, "dtype", None) == np.float32
                    ):
                        random_matrix = prefetched

                if random_matrix is None:
                    from app.services.cache_registry import get_projection_matrix_cache

                    proj_cache = get_projection_matrix_cache()
                    cache_key = f"proj:{current_dim}x{projected_dimension}"
                    random_matrix = proj_cache.get(cache_key)
                    # Validate cached matrix shape/dtype
                    if (
                        random_matrix is None
                        or getattr(random_matrix, "shape", None)
                        != (current_dim, projected_dimension)
                        or getattr(random_matrix, "dtype", None) != np.float32
                    ):
                        rng = np.random.default_rng(seed=RANDOM_SEED)
                        random_matrix = rng.uniform(
                            -1, 1, (current_dim, projected_dimension)
                        ).astype(np.float32)
                        proj_cache.put(cache_key, random_matrix)

                # Ensure embedding is float32 for efficient matmul
                if embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)

                # Perform single vectorized matmul for both 1D and 2D embeddings
                embedding = embedding @ random_matrix
                logger.debug("After projection: shape=%s", str(embedding.shape))
        else:
            logger.debug(
                "Skipping projection: projected_dimension=%s, current_dim=%s",
                str(projected_dimension),
                str(current_dim),
            )

        # Normalize AFTER projection to ensure normalized vectors for cosine similarity
        normalize = kwargs.get("normalize", getattr(model_config, "normalize", True))
        if normalize:
            # Hybrid normalization: use optimized approach for large batches, np.linalg.norm for small
            if embedding.ndim == 1:
                # For single vectors, np.linalg.norm is well-optimized
                norm = np.linalg.norm(embedding)
                if norm > NORM_EPS:
                    embedding = embedding / norm
            elif embedding.ndim == 2:
                # For batches, use element-wise ops which scale better with large batches
                # Optimized ~30-50% faster for batches > 64 rows
                row_norms = np.sqrt(
                    np.sum(embedding * embedding, axis=1, keepdims=True)
                )
                # Avoid division by zero for zero vectors
                row_norms_safe = np.where(row_norms > NORM_EPS, row_norms, 1.0)
                embedding = embedding / row_norms_safe
            else:
                norm = np.linalg.norm(embedding)
                if norm > NORM_EPS:
                    embedding = embedding / norm

        # Quantization AFTER normalization (if enabled)
        quantize = kwargs.get("quantize", getattr(model_config, "quantize", False))
        if quantize:
            quantize_type = kwargs.get(
                "quantize_type", getattr(model_config, "quantize_type", "int8")
            )
            embedding = SentenceTransformer._quantize_embedding(
                embedding, quantize_type
            )

        return embedding

    @staticmethod
    def _project_embedding(
        embedding: np.ndarray, projected_dimension: int
    ) -> np.ndarray:
        """Project embedding to target dimension using cached random matrix for consistency."""
        input_dim = embedding.shape[-1]
        cache_key = f"proj:{input_dim}x{projected_dimension}"

        # Import registry accessor here to avoid import-order cycles
        from app.services.cache_registry import get_projection_matrix_cache

        proj_cache = get_projection_matrix_cache()
        random_matrix = proj_cache.get(cache_key)
        if (
            random_matrix is None
            or getattr(random_matrix, "shape", None) != (input_dim, projected_dimension)
            or getattr(random_matrix, "dtype", None) != np.float32
        ):
            rng = np.random.default_rng(seed=RANDOM_SEED)
            random_matrix = rng.uniform(-1, 1, (input_dim, projected_dimension)).astype(
                np.float32
            )
            proj_cache.put(cache_key, random_matrix)

        # Ensure dtype compatibility
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        return embedding @ random_matrix

    @staticmethod
    def _quantize_embedding(embedding: np.ndarray, quantize_type: str) -> np.ndarray:
        """
        Quantize embeddings to reduce storage and memory footprint.

        Args:
            embedding: Input embedding (1D or 2D array)
            quantize_type: Quantization method - 'int8', 'uint8', or 'binary'

        Returns:
            Quantized embedding as numpy array

        Notes:
            - int8: Maps [-1, 1] to [-127, 127], ~4x storage reduction
            - uint8: Maps [min, max] to [0, 255], ~4x storage reduction
            - binary: Thresholds at 0, converts to {-1, 1}, ~32x storage reduction
        """
        # Handle empty arrays
        if embedding.size == 0:
            if quantize_type in ["int8", "binary"]:
                return embedding.astype(np.int8)
            elif quantize_type == "uint8":
                return embedding.astype(np.uint8)
            else:
                return embedding

        if quantize_type == "int8":
            # Assume normalized embeddings in [-1, 1] range
            # Scale to int8 range [-127, 127] for symmetric quantization
            return np.clip(np.round(embedding * 127), -127, 127).astype(np.int8)

        elif quantize_type == "uint8":
            # Use min-max normalization to [0, 255]
            # Works better for non-normalized embeddings
            if embedding.ndim == 1:
                e_min, e_max = embedding.min(), embedding.max()
            else:
                e_min = embedding.min(axis=1, keepdims=True)
                e_max = embedding.max(axis=1, keepdims=True)

            # Avoid division by zero
            scale = e_max - e_min
            scale = np.where(scale > 1e-10, scale, 1.0)

            normalized = (embedding - e_min) / scale
            return np.clip(np.round(normalized * 255), 0, 255).astype(np.uint8)

        elif quantize_type == "binary":
            # Simple sign-based binarization: > 0 → 1, <= 0 → -1
            # Extremely compact but loses magnitude information
            return np.where(embedding > 0, 1, -1).astype(np.int8)

        else:
            logger.warning(
                f"Unknown quantization type '{quantize_type}', returning original embedding"
            )
            return embedding

    @staticmethod
    def _truncate_text_to_token_limit(
        text: str, tokenizer: Any, max_tokens: int = DEFAULT_MAX_LENGTH
    ) -> str:
        """Backward compatibility for tests."""
        return SentenceTransformer._preprocess_text(text, True, False)[: max_tokens * 4]

    @staticmethod
    def embed_text(req: EmbeddingRequest, **kwargs: Any) -> EmbeddingResponse:
        """Main embedding function for single text."""
        start_time = time.time()
        payload = {
            "success": True,
            "message": "Embedding generated successfully",
            "model": req.model,
            "results": [],
            "time_taken": 0.0,
        }
        response = EmbeddingResponse(**cast(Any, payload))

        try:
            # Ensure model exists in configuration
            if BaseNLPService._get_model_config(req.model) is None:
                raise ModelNotFoundError(f"Model '{req.model}' not found")

            # Check model capability for embeddings (requires explicit `tasks` in config)
            if not BaseNLPService._can_perform_task(req.model, "embedding"):
                payload = {
                    "success": False,
                    "message": f"Model '{req.model}' cannot perform 'embedding' task",
                    "model": req.model,
                    "results": [],
                    "time_taken": 0.0,
                }
                return EmbeddingResponse(**cast(Any, payload))
            # Extract request parameters for config override
            request_params = {
                "pooling_strategy": getattr(req, "pooling_strategy", None),
                "max_length": getattr(req, "max_length", None),
                "chunk_logic": getattr(req, "chunk_logic", None),
                "chunk_overlap": getattr(req, "chunk_overlap", None),
                "chunk_size": getattr(req, "chunk_size", None),
                "legacy_tokenizer": getattr(req, "legacy_tokenizer", None),
                "normalize": getattr(req, "normalize", None),
                "force_pooling": getattr(req, "force_pooling", None),
                "lowercase": getattr(req, "lowercase", None),
                "remove_emojis": getattr(req, "remove_emojis", None),
                "use_optimized": getattr(req, "use_optimized", None),
            }

            result = SentenceTransformer._embed_text_local(
                text=req.input,
                model=req.model,
                projected_dimension=req.projected_dimension,
                join_chunks=req.join_chunks,
                join_by_pooling_strategy=req.join_by_pooling_strategy,
                output_large_text_upon_join=req.output_large_text_upon_join,
                request_params=request_params,
                **kwargs,
            )

            if result.success:
                response.results = result.embedding_chunks
                response.used_parameters = getattr(result, "used_parameters", {})
                response.warnings = getattr(result, "warnings", [])
            else:
                response.success = False
                response.message = result.message

        except ModelNotFoundError as e:
            response.success = False
            response.message = str(e)
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
        """Batch embedding with optimized ONNX inference."""
        start_time = time.time()
        payload = {
            "success": True,
            "message": "Batch embedding generated successfully",
            "model": requests.model,
            "results": [],
            "time_taken": 0.0,
        }
        response = EmbeddingBatchResponse(**cast(Any, payload))

        try:
            # Ensure model exists in configuration
            if BaseNLPService._get_model_config(requests.model) is None:
                raise ModelNotFoundError(f"Model '{requests.model}' not found")

            # Check model capability for embeddings
            if not BaseNLPService._can_perform_task(requests.model, "embedding"):
                payload = {
                    "success": False,
                    "message": f"Model '{requests.model}' cannot perform 'embedding' task",
                    "model": requests.model,
                    "results": [],
                    "time_taken": 0.0,
                }
                return EmbeddingBatchResponse(**cast(Any, payload))
            BatchLimiter.validate_batch_size(
                requests.inputs, max_size=DEFAULT_BATCH_SIZE
            )

            # Try batch processing if all inputs are short enough
            use_batch_inference = SentenceTransformer._can_use_batch_inference(
                requests.inputs, requests.model
            )

            if use_batch_inference:
                # Use optimized batch ONNX inference
                logger.info(
                    f"Using batch ONNX inference for {len(requests.inputs)} inputs"
                )
                request_params = {
                    "pooling_strategy": requests.pooling_strategy,
                    "max_length": requests.max_length,
                    "chunk_logic": requests.chunk_logic,
                    "chunk_overlap": requests.chunk_overlap,
                    "chunk_size": requests.chunk_size,
                    "legacy_tokenizer": requests.legacy_tokenizer,
                    "normalize": requests.normalize,
                    "force_pooling": requests.force_pooling,
                    "lowercase": requests.lowercase,
                    "remove_emojis": requests.remove_emojis,
                    "use_optimized": requests.use_optimized,
                }
                try:
                    result = SentenceTransformer._embed_batch_texts(
                        texts=requests.inputs,
                        model=requests.model,
                        projected_dimension=requests.projected_dimension,
                        request_params=request_params,
                        **kwargs,
                    )
                    if result.success:
                        response.results = result.embedding_chunks
                        response.used_parameters = getattr(
                            result, "used_parameters", {}
                        )
                        response.warnings = getattr(result, "warnings", [])
                    else:
                        response.success = False
                        response.message = result.message
                except (InferenceError, ValueError) as e:
                    # Batch inference failed, fall back to sequential
                    logger.warning(
                        f"Batch inference failed ({e}), falling back to sequential processing"
                    )
                    use_batch_inference = (
                        False  # Will trigger sequential processing below
                    )

            if not use_batch_inference:
                # Fall back to sequential processing for mixed-length inputs or batch failures
                logger.info(
                    f"Using sequential processing for {len(requests.inputs)} inputs"
                )
                for idx, input_text in enumerate(requests.inputs):
                    # Extract batch request parameters for config override
                    request_params = {
                        "pooling_strategy": requests.pooling_strategy,
                        "max_length": requests.max_length,
                        "chunk_logic": requests.chunk_logic,
                        "chunk_overlap": requests.chunk_overlap,
                        "chunk_size": requests.chunk_size,
                        "legacy_tokenizer": requests.legacy_tokenizer,
                        "normalize": requests.normalize,
                        "force_pooling": requests.force_pooling,
                        "lowercase": requests.lowercase,
                        "remove_emojis": requests.remove_emojis,
                        "use_optimized": requests.use_optimized,
                    }

                    result = SentenceTransformer._embed_text_local(
                        text=input_text,
                        model=requests.model,
                        projected_dimension=requests.projected_dimension,
                        join_chunks=requests.join_chunks,
                        join_by_pooling_strategy=requests.join_by_pooling_strategy,
                        output_large_text_upon_join=requests.output_large_text_upon_join,
                        request_params=request_params,
                        **kwargs,
                    )

                    if not result.success:
                        response.success = False
                        response.message = f"Error in input {idx}: {result.message}"
                        break
                    else:
                        response.results.extend(result.embedding_chunks)
                        if (
                            idx == 0
                        ):  # Set used_parameters and warnings from first result
                            response.used_parameters = getattr(
                                result, "used_parameters", {}
                            )
                            response.warnings = getattr(result, "warnings", [])

        except ModelNotFoundError as e:
            response.success = False
            response.message = str(e)
        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
        finally:
            response.time_taken = time.time() - start_time
            return response

    @staticmethod
    def _can_use_batch_inference(texts: List[str], model: str) -> bool:
        """Check if batch ONNX inference can be used for all texts.

        Returns True if all texts are short enough to avoid chunking.
        """
        try:
            model_config = SentenceTransformer._get_model_config(model)
            if not model_config:
                return False

            max_length = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)
            # Simple heuristic: if all texts are under ~75% of max_length (in chars),
            # they likely won't need chunking. 4 chars per token is a rough estimate.
            char_limit = int(max_length * 3)  # ~75% of tokens * 4 chars/token

            for text in texts:
                if len(text) > char_limit:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _embed_batch_texts(
        texts: List[str],
        model: str,
        projected_dimension: Optional[int],
        request_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> ChunkEmbeddingResult:
        """Embed multiple texts using batched ONNX inference.

        Processes all texts in a single ONNX call for 3-5x performance improvement.
        """
        try:
            model_config, tokenizer, session = (
                SentenceTransformer._prepare_embedding_resources(model)
            )

            # Override config with request parameters
            if request_params:
                model_config = SentenceTransformer._override_config_with_request(
                    model_config, request_params
                )

            # Preprocess all texts
            lowercase = getattr(model_config, "lowercase", True)
            remove_emojis = getattr(model_config, "remove_emojis", False)
            processed_texts = [
                SentenceTransformer._preprocess_text(text, lowercase, remove_emojis)
                for text in texts
            ]

            # Tokenize all texts in batch
            max_length = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)

            # Try batch tokenization; if it fails, tokenize individually
            try:
                encodings = tokenizer(
                    processed_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np",
                )
            except (TypeError, AttributeError) as e:
                # Tokenizer might not support batching; tokenize individually
                logger.debug(
                    f"Batch tokenization failed ({e}), tokenizing individually"
                )
                all_input_ids = []
                all_attention_masks = []
                for text in processed_texts:
                    enc = tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="np",
                    )
                    all_input_ids.append(enc["input_ids"])
                    all_attention_masks.append(enc.get("attention_mask"))

                input_ids = np.vstack(all_input_ids)
                attention_mask = (
                    np.vstack(all_attention_masks)
                    if all_attention_masks[0] is not None
                    else None
                )
                encodings = {"input_ids": input_ids, "attention_mask": attention_mask}

            # Prepare batched ONNX inputs
            input_ids = encodings["input_ids"]
            attention_mask = encodings.get("attention_mask")

            # Ensure correct dtypes
            if input_ids.dtype != np.int64:
                input_ids = input_ids.astype(np.int64)
            if attention_mask is not None and attention_mask.dtype != np.int64:
                attention_mask = attention_mask.astype(np.int64)

            # Build ONNX inputs with auto-detected names
            model_input_names = [inp.name for inp in session.get_inputs()]
            input_names_config = getattr(model_config, "inputnames", {})

            # Reuse the name selection logic from _prepare_onnx_inputs
            def _select_name(
                config_name: Optional[str],
                candidates: List[str],
                default: Optional[str] = None,
            ) -> Optional[str]:
                if config_name:
                    return config_name
                for cand in candidates:
                    if cand in model_input_names:
                        return cand
                model_input_names_lc = [n.lower() for n in model_input_names]
                for cand in candidates:
                    lc = cand.lower()
                    if lc in model_input_names_lc:
                        return model_input_names[model_input_names_lc.index(lc)]
                for cand in candidates:
                    lc = cand.lower()
                    for name, name_lc in zip(model_input_names, model_input_names_lc):
                        if lc in name_lc:
                            return name
                return default

            input_id_name = _select_name(
                getattr(input_names_config, "input", None),
                ["input_ids", "input"],
                "input_ids",
            )
            inputs = {input_id_name: input_ids}

            if attention_mask is not None:
                mask_name = _select_name(
                    getattr(input_names_config, "mask", None),
                    ["attention_mask", "mask"],
                    None,
                )
                if mask_name:
                    inputs[mask_name] = attention_mask

            # Run batched ONNX inference
            outputs = session.run(None, inputs)
            embeddings = outputs[
                0
            ]  # Shape: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)

            # Validate batch dimension matches input
            if embeddings.shape[0] != len(texts):
                raise ValueError(
                    f"ONNX output batch size ({embeddings.shape[0]}) doesn't match "
                    f"input batch size ({len(texts)}). Model may not support batching."
                )

            # Get prefetched projection matrix once for all texts
            prefetched_proj = SentenceTransformer._get_prefetched_proj_matrix(
                model_config, projected_dimension
            )

            # Process each embedding in the batch
            results = []
            for idx, text in enumerate(texts):
                # Extract single embedding and mask from batch
                single_embedding = embeddings[idx]  # (seq_len, hidden_dim)
                single_mask = (
                    attention_mask[idx] if attention_mask is not None else None
                )

                # Process embedding output (pooling, projection, normalization)
                local_kwargs = dict(kwargs)
                if prefetched_proj is not None:
                    local_kwargs["prefetched_proj_matrix"] = prefetched_proj

                processed = SentenceTransformer._process_embedding_output(
                    single_embedding[None, :, :],  # Add batch dim back for pooling
                    model_config,
                    single_mask[None, :] if single_mask is not None else None,
                    projected_dimension,
                    **local_kwargs,
                )

                # Flatten and convert to list
                vector = processed.flatten().tolist()
                results.append(
                    EmbededChunk(
                        vector=vector,
                        chunk=text,
                        joined_chunk=False,
                        only_vector=False,
                        item_number=idx,
                        content_as=None,
                    )
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
                "normalize": getattr(model_config, "normalize", True),
                "batch_inference": True,  # Indicator that batch inference was used
            }

            warnings = []
            if hasattr(model_config, "_dimension_warning"):
                warnings.append(model_config._dimension_warning)

            return ChunkEmbeddingResult(
                embedding_chunks=results,
                message="Batch embedding generated successfully",
                success=True,
                used_parameters=used_params,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Batch inference failed: {e}, falling back to sequential")
            raise InferenceError(f"Error in batch embedding: {e}")

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
        """Core embedding logic for text processing."""
        try:
            model_config, tokenizer, session = (
                SentenceTransformer._prepare_embedding_resources(model)
            )

            # Override config with request parameters if provided
            if request_params:
                model_config = SentenceTransformer._override_config_with_request(
                    model_config, request_params
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
                "use_optimized": getattr(model_config, "use_optimized", False),
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
            raise ModelLoadError(f"System error accessing model: {e}")
        except (ValueError, KeyError) as e:
            raise InferenceError(f"Invalid model configuration: {e}")
        except Exception as e:
            raise InferenceError(f"Error generating embedding: {e}")

    @staticmethod
    def _override_config_with_request(model_config: Any, request_params: dict) -> Any:
        """Use request parameters first, model config as fallback when parameters are None."""
        override_fields = [
            "pooling_strategy",
            "max_length",
            "chunk_logic",
            "chunk_overlap",
            "chunk_size",
            "legacy_tokenizer",
            "normalize",
            "force_pooling",
            "lowercase",
            "remove_emojis",
            "use_optimized",
        ]

        for field in override_fields:
            if field in request_params:
                if request_params[field] is not None:
                    # Use request parameter
                    setattr(model_config, field, request_params[field])
                    logger.debug(
                        f"Using request parameter for {field}: {request_params[field]}"
                    )
                # If request param is None, keep model config value (fallback)
                else:
                    logger.debug(
                        f"Using model config fallback for {field}: {getattr(model_config, field, None)}"
                    )

        return model_config

    @staticmethod
    def _prepare_embedding_resources(model: str) -> tuple[Any, Any, Any]:
        """Prepare model config, tokenizer, and session for embedding."""
        original_config = SentenceTransformer._get_model_config(model)
        if not original_config:
            raise ModelNotFoundError(f"Model '{model}' not found")

        # Resolve model path first (tests may patch this) and then validate
        # availability using the pre-fetched config and resolved path. This
        # avoids duplicate lookups and honors patched `SentenceTransformer._get_model_path`.
        model_to_use_path = SentenceTransformer._get_model_path(model, original_config)
        if not BaseNLPService._validate_model_availability(
            model,
            task="embedding",
            perform_filesystem_check=False,
            cfg=original_config,
            model_path=model_to_use_path,
        ):
            raise ModelNotFoundError(
                f"Model '{model}' not available or model files missing"
            )
        # Create a copy to avoid modifying the cached instance
        if hasattr(original_config, "model_copy"):
            model_config = original_config.model_copy()
        elif hasattr(original_config, "copy"):
            # For objects with copy method
            model_config = original_config.copy()
        else:
            # Fallback: create shallow copy using copy module
            import copy

            model_config = copy.copy(original_config)
        model_to_use_path = SentenceTransformer._get_model_path(model, model_config)
        tokenizer = SentenceTransformer._load_tokenizer(model_to_use_path, model_config)
        session = SentenceTransformer._load_session(model_to_use_path, model_config)

        cache_stats = SentenceTransformer._get_cache_stats()
        logger.info(
            f"Cache sizes - Encoder sessions: {cache_stats['encoder_sessions']}, Model configs: {cache_stats['model_configs']}"
        )
        return model_config, tokenizer, session

    @staticmethod
    def _get_model_path(model: str, model_config: Any) -> str:
        """Get validated model path."""
        # Delegate to BaseNLPService model path resolver which prefers
        # `model_folder_name` and falls back to task-based folders.
        resolved = BaseNLPService._get_model_path(model)
        if not resolved:
            raise ModelLoadError(f"Failed to resolve model path for {model}")
        return resolved

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
        """Load and validate ONNX session with fallback."""
        model_path = SentenceTransformer._get_embedding_model_path(
            model_to_use_path, model_config
        )

        try:
            session = SentenceTransformer._get_encoder_session(model_path)
        except ModelLoadError as e:
            # If optimized model fails, try fallback to regular model
            use_optimized = getattr(model_config, "use_optimized", False)
            if use_optimized and "optimized" in str(e).lower():
                logger.warning(f"Optimized model failed: {e}, trying regular model")
                # Get regular model path
                regular_filename = (
                    getattr(model_config, "encoder_onnx_model", None) or "model.onnx"
                )
                fallback_path = validate_safe_path(
                    os.path.join(model_to_use_path, regular_filename),
                    SentenceTransformer._root_path,
                )
                session = SentenceTransformer._get_encoder_session(fallback_path)
            else:
                raise e

        if not session:
            raise ModelLoadError(f"Failed to load ONNX session: {model_path}")

        # Auto-detect native dimension from ONNX model
        native_dim = SentenceTransformer._get_native_dimension_from_session(session)

        # Validate and adjust dimension if needed
        if native_dim:
            if not hasattr(model_config, "dimension") or model_config.dimension is None:
                # No dimension in config - use native
                model_config.dimension = native_dim
                logger.info(
                    f"Auto-detected native dimension from ONNX model: {native_dim}"
                )
            elif model_config.dimension > native_dim:
                # Config dimension is larger than native - use native instead
                original_dim = model_config.dimension
                model_config.dimension = native_dim
                logger.warning(
                    f"Config dimension ({original_dim}) exceeds native dimension ({native_dim}). "
                    f"Using native dimension to prevent upsampling."
                )
                # Store warning to be included in response
                if not hasattr(model_config, "_dimension_warning"):
                    model_config._dimension_warning = (
                        f"Config dimension ({original_dim}) was larger than model's native dimension ({native_dim}). "
                        f"Using native dimension {native_dim} to avoid information loss."
                    )

        # Auto-detect output names from ONNX model
        output_names_list = SentenceTransformer._get_output_names_from_session(session)
        if output_names_list and (
            not hasattr(model_config, "outputnames") or not model_config.outputnames
        ):
            # Create outputnames object with primary output
            from types import SimpleNamespace

            model_config.outputnames = SimpleNamespace(output=output_names_list[0])
            logger.info(f"Auto-detected primary output name: {output_names_list[0]}")

        # Ensure and retrieve the encoder sessions cache via registry helper
        from app.services.cache_registry import get_encoder_sessions

        encoder_sessions = get_encoder_sessions()
        logger.info(f"Encoder session cache size: {encoder_sessions.size()}")
        return session

    @staticmethod
    def _get_native_dimension_from_session(session: Any) -> Optional[int]:
        """Extract the native embedding dimension from ONNX session output shape."""
        try:
            outputs = session.get_outputs()
            if outputs and len(outputs) > 0:
                output_shape = outputs[0].shape
                # Output shape is typically ['batch_size', 'sequence_length', dimension]
                # The last dimension is the embedding dimension
                if output_shape and len(output_shape) >= 3:
                    # Handle symbolic dimensions (e.g., 'batch_size') vs numeric
                    last_dim = output_shape[-1]
                    if isinstance(last_dim, (int, np.integer)):
                        logger.debug(
                            f"Detected native dimension from ONNX output: {last_dim}"
                        )
                        return int(last_dim)
        except Exception as e:
            logger.warning(f"Could not auto-detect dimension from ONNX session: {e}")
        return None

    @staticmethod
    def _get_output_names_from_session(session: Any) -> List[str]:
        """Extract output tensor names from ONNX session."""
        try:
            outputs = session.get_outputs()
            if outputs:
                output_names = [output.name for output in outputs]
                logger.debug(
                    f"Auto-detected output names from ONNX model: {output_names}"
                )
                return output_names
        except Exception as e:
            logger.warning(f"Could not auto-detect output names from ONNX session: {e}")
        return []

    @staticmethod
    def _get_embedding_model_path(model_to_use_path: str, model_config: Any) -> str:
        """Get the path to the embedding model file."""
        if not model_to_use_path:
            raise ModelLoadError(
                "Model path is not provided to _get_embedding_model_path"
            )

        root = SentenceTransformer._root_path
        if not root:
            raise ModelLoadError("SentenceTransformer root path is not configured")

        model_filename = SentenceTransformer._get_model_filename(
            model_config, is_embedding=True
        )
        return validate_safe_path(os.path.join(model_to_use_path, model_filename), root)

    @staticmethod
    def _get_model_filename(
        model_config: Any, is_embedding: bool = True, fallback_to_regular: bool = False
    ) -> str:
        """Delegate filename selection to BaseNLPService helper."""
        from app.services.base_nlp_service import BaseNLPService

        model_type = "encoder" if is_embedding else "decoder"
        return BaseNLPService._get_model_filename(
            model_config, model_type, fallback_to_regular
        )

    @staticmethod
    def _prepare_text_chunks(text: str, tokenizer: Any, model_config: Any) -> List[str]:
        """Prepare text chunks for processing."""
        max_tokens = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)
        return ChunkingStrategies.split_text_into_chunks(
            text, tokenizer, max_tokens, model_config
        )

    @staticmethod
    def _get_prefetched_proj_matrix(
        model_config: Any, projected_dimension: Optional[int]
    ):
        """Prefetch projection matrix for given model config and target dim.

        Returns a float32 matrix or None. Kept small to keep callers concise.
        """
        try:
            native_dim = getattr(model_config, "dimension", None)
            if (
                projected_dimension is None
                or projected_dimension <= 0
                or native_dim is None
                or projected_dimension >= native_dim
            ):
                return None
            from app.services.cache_registry import get_projection_matrix_cache

            cache_key = f"proj:{native_dim}x{projected_dimension}"
            proj_cache = get_projection_matrix_cache()
            mat = proj_cache.get(cache_key)
            if mat is None:
                rng = np.random.default_rng(seed=RANDOM_SEED)
                mat = rng.uniform(-1, 1, (native_dim, projected_dimension)).astype(
                    np.float32
                )
                proj_cache.put(cache_key, mat)
            return mat
        except Exception:
            return None

    @staticmethod
    def _process_text_chunks(
        chunks: List[str],
        model_config: Any,
        tokenizer: Any,
        session: Any,
        projected_dimension: Optional[int],
        **kwargs,
    ) -> List[EmbededChunk]:
        """Process text chunks into embeddings."""
        results = []
        prefetched_proj = SentenceTransformer._get_prefetched_proj_matrix(
            model_config, projected_dimension
        )
        for chunk in chunks:
            local_kwargs = dict(kwargs)
            if prefetched_proj is not None:
                local_kwargs["prefetched_proj_matrix"] = prefetched_proj
            embedding_result = SentenceTransformer._process_single_chunk(
                chunk,
                model_config,
                tokenizer,
                session,
                projected_dimension,
                **local_kwargs,
            )
            results.append(
                EmbededChunk(
                    vector=embedding_result.vector,
                    chunk=chunk,
                    joined_chunk=False,
                    only_vector=False,
                    item_number=None,
                    content_as=None,
                )
            )
        return results

    @staticmethod
    def _process_single_chunk(
        chunk: str,
        model_config: Any,
        tokenizer: Any,
        session: Any,
        projected_dimension: Optional[int],
        **kwargs,
    ) -> Any:
        """Process a single text chunk into embedding."""
        # If projected_dimension is None, fall back to model_config.dimension or default
        pd = (
            projected_dimension
            if projected_dimension is not None
            else getattr(model_config, "dimension", DEFAULT_PROJECTED_DIMENSION)
        )

        embedding_result = SentenceTransformer._small_text_embedding(
            small_text=chunk,
            model_config=model_config,
            tokenizer=tokenizer,
            session=session,
            projected_dimension=pd,
            **kwargs,
        )
        if not embedding_result.success:
            raise InferenceError(embedding_result.message)
        return embedding_result

    @staticmethod
    def _join_chunk_embeddings(
        chunk_results: List[EmbededChunk],
        original_text: str,
        join_by_pooling_strategy: Optional[str],
        model_config: Any,
        output_large_text_upon_join: Optional[bool],
    ) -> List[EmbededChunk]:
        """Join chunk embeddings using specified pooling strategy."""
        pooling_strategy = (
            join_by_pooling_strategy
            if join_by_pooling_strategy is not None
            else getattr(model_config, "pooling_strategy", "mean")
        )

        if pooling_strategy not in ["mean", "max", "first", "last"]:
            pooling_strategy = "mean"

        merged_vector = SentenceTransformer._merge_vectors(
            chunk_results, pooling_strategy
        )

        out_large = bool(output_large_text_upon_join)
        return [
            EmbededChunk(
                vector=merged_vector,
                joined_chunk=True,
                only_vector=not out_large,
                chunk="" if not out_large else original_text,
                item_number=None,
                content_as=None,
            )
        ]
