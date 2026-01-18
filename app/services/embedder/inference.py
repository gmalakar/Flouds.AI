# =============================================================================
# File: inference.py
# Date: 2026-01-17
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Core embedding inference logic for single texts and batches."""

import logging
from typing import Any, Callable, List, Optional

import numpy as np

from app.exceptions import InferenceError
from app.logger import get_logger
from app.models.embedded_chunk import EmbededChunk
from app.services.embedder import onnx_utils, processing, text_utils
from app.services.embedder.models import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_PROJECTED_DIMENSION,
    ChunkEmbeddingResult,
    SingleEmbeddingResult,
)
from app.utils.chunking_strategies import ChunkingStrategies

logger = get_logger("embedder_service.inference")


def small_text_embedding(
    small_text: str,
    model_config: Any,
    tokenizer: Any,
    session: Any,
    projected_dimension: int = DEFAULT_PROJECTED_DIMENSION,
    **kwargs: Any,
) -> SingleEmbeddingResult:
    """Generate embedding for a single text chunk.

    Args:
        small_text: Input text (should fit within max_length)
        model_config: Model configuration
        tokenizer: Loaded tokenizer
        session: ONNX Runtime session
        projected_dimension: Target dimension for projection
        **kwargs: Additional parameters for processing

    Returns:
        SingleEmbeddingResult with vector and status
    """
    try:
        # Tokenize and prepare inputs
        processed_text = text_utils.prepare_text_for_embedding(small_text, model_config)
        inputs = onnx_utils.prepare_onnx_inputs(processed_text, tokenizer, model_config, session)

        # Run inference
        outputs = session.run(None, inputs)
        onnx_utils.log_onnx_outputs(outputs, session)

        # Process embedding
        embedding = processing.process_embedding_output(
            outputs[0],
            model_config,
            inputs.get("attention_mask"),
            projected_dimension,
            **kwargs,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Final embedding shape before flatten: %s", embedding.shape)
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


def process_text_chunks(
    chunks: List[str],
    model_config: Any,
    tokenizer: Any,
    session: Any,
    projected_dimension: Optional[int],
    **kwargs: Any,
) -> List[EmbededChunk]:
    """Process text chunks into embeddings.

    Args:
        chunks: List of text chunks to embed
        model_config: Model configuration
        tokenizer: Loaded tokenizer
        session: ONNX Runtime session
        projected_dimension: Target dimension (None uses native)
        **kwargs: Additional parameters

    Returns:
        List of EmbededChunk objects with vectors
    """
    results = []
    prefetched_proj = processing.get_prefetched_proj_matrix(model_config, projected_dimension)
    for chunk in chunks:
        local_kwargs = dict(kwargs)
        if prefetched_proj is not None:
            local_kwargs["prefetched_proj_matrix"] = prefetched_proj
        embedding_result = process_single_chunk(
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


def process_single_chunk(
    chunk: str,
    model_config: Any,
    tokenizer: Any,
    session: Any,
    projected_dimension: Optional[int],
    **kwargs: Any,
) -> SingleEmbeddingResult:
    """Process a single text chunk into embedding.

    Args:
        chunk: Text chunk to embed
        model_config: Model configuration
        tokenizer: Loaded tokenizer
        session: ONNX Runtime session
        projected_dimension: Target dimension (None uses native)
        **kwargs: Additional parameters

    Returns:
        SingleEmbeddingResult with vector
    """
    # If projected_dimension is None, fall back to model_config.dimension or default
    pd = (
        projected_dimension
        if projected_dimension is not None
        else getattr(model_config, "dimension", DEFAULT_PROJECTED_DIMENSION)
    )

    embedding_result = small_text_embedding(
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


def prepare_text_chunks(text: str, tokenizer: Any, model_config: Any) -> List[str]:
    """Prepare text chunks for processing.

    Args:
        text: Input text to chunk
        tokenizer: Tokenizer for token counting
        model_config: Configuration with chunk settings

    Returns:
        List of text chunks
    """
    max_tokens = getattr(model_config, "max_length", DEFAULT_MAX_LENGTH)
    return ChunkingStrategies.split_text_into_chunks(text, tokenizer, max_tokens, model_config)


def join_chunk_embeddings(
    chunk_results: List[EmbededChunk],
    original_text: str,
    join_by_pooling_strategy: Optional[str],
    model_config: Any,
    output_large_text_upon_join: Optional[bool],
) -> List[EmbededChunk]:
    """Join chunk embeddings using specified pooling strategy.

    Args:
        chunk_results: List of embedded chunks
        original_text: Original input text
        join_by_pooling_strategy: Pooling strategy override
        model_config: Model configuration
        output_large_text_upon_join: Whether to include full text in result

    Returns:
        Single-element list with merged embedding
    """
    pooling_strategy = (
        join_by_pooling_strategy
        if join_by_pooling_strategy is not None
        else getattr(model_config, "pooling_strategy", "mean")
    )

    if pooling_strategy not in ["mean", "max", "first", "last"]:
        pooling_strategy = "mean"

    merged_vector = text_utils.merge_vectors(chunk_results, pooling_strategy)

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


def can_use_batch_inference(
    texts: List[str], model: str, get_config_func: Callable[[str], Any]
) -> bool:
    """Check if batch ONNX inference can be used for all texts.

    Returns True if all texts are short enough to avoid chunking.

    Args:
        texts: List of input texts
        model: Model name
        get_config_func: Function to retrieve model config

    Returns:
        True if all texts can be batched, False otherwise
    """
    try:
        model_config = get_config_func(model)
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


def embed_batch_texts(
    texts: List[str],
    model_config: Any,
    tokenizer: Any,
    session: Any,
    projected_dimension: Optional[int],
    **kwargs: Any,
) -> ChunkEmbeddingResult:
    """Embed multiple texts using batched ONNX inference.

    Processes all texts in a single ONNX call for 3-5x performance improvement.

    Args:
        texts: List of texts to embed
        model_config: Model configuration
        tokenizer: Loaded tokenizer
        session: ONNX Runtime session
        projected_dimension: Target dimension (None uses native)
        **kwargs: Additional parameters

    Returns:
        ChunkEmbeddingResult with all text embeddings
    """
    try:
        # Preprocess all texts
        lowercase = getattr(model_config, "lowercase", True)
        remove_emojis = getattr(model_config, "remove_emojis", False)
        processed_texts = [
            text_utils.preprocess_text(text, lowercase, remove_emojis) for text in texts
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
            logger.debug(f"Batch tokenization failed ({e}), tokenizing individually")
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
                np.vstack(all_attention_masks) if all_attention_masks[0] is not None else None
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

        # Reuse the name selection logic from onnx_utils
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
        prefetched_proj = processing.get_prefetched_proj_matrix(model_config, projected_dimension)

        # Process each embedding in the batch
        results = []
        for idx, text in enumerate(texts):
            # Extract single embedding and mask from batch
            single_embedding = embeddings[idx]  # (seq_len, hidden_dim)
            single_mask = attention_mask[idx] if attention_mask is not None else None

            # Process embedding output (pooling, projection, normalization)
            local_kwargs = dict(kwargs)
            if prefetched_proj is not None:
                local_kwargs["prefetched_proj_matrix"] = prefetched_proj

            processed = processing.process_embedding_output(
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
