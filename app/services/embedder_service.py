# =============================================================================
# File: embedder_service.py
# Date: 2025-01-15
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import asyncio
import functools
import os
import time
from typing import Any, List

import numpy as np
from pydantic import BaseModel, Field

from app.logger import get_logger
from app.models.embedded_chunk import EmbededChunk
from app.models.embedding_request import EmbeddingBatchRequest, EmbeddingRequest
from app.models.embedding_response import EmbeddingBatchResponse, EmbeddingResponse
from app.services.base_nlp_service import BaseNLPService

logger = get_logger("embedder_service")


class _EmbeddingResults(BaseModel):
    EmbeddingResults: list[EmbededChunk]
    message: str
    success: bool = Field(default=True)


class SentenceTransformer(BaseNLPService):
    """Static class for sentence embedding using ONNX models."""

    @staticmethod
    def _merge_vectors(chunks: list[EmbededChunk], method: str = "mean") -> list[float]:
        """Merge embedding vectors using mean or max pooling."""
        vectors = [
            np.array(chunk.vector) for chunk in chunks if hasattr(chunk, "vector")
        ]
        if not vectors:
            return []

        stacked = np.stack(vectors)
        merged = (
            np.max(stacked, axis=0) if method == "max" else np.mean(stacked, axis=0)
        )
        return merged.tolist()

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Text preprocessing with line break handling."""
        import re

        # Handle line breaks and normalize whitespace
        text = re.sub(
            r"\s+", " ", text
        )  # Replace multiple whitespace with single space
        text = text.strip()

        return text

    @staticmethod
    def _pooling(embedding: np.ndarray, strategy: str = "mean") -> np.ndarray:
        """Apply pooling strategy to embeddings."""
        if embedding.ndim == 1:
            return embedding

        if strategy == "cls":
            return (
                embedding[0]
                if embedding.ndim == 2
                else embedding[(0,) * (embedding.ndim - 1)]
            )
        elif strategy == "max":
            return embedding.max(axis=tuple(range(embedding.ndim - 1)))
        else:  # mean
            return embedding.mean(axis=tuple(range(embedding.ndim - 1)))

    @staticmethod
    def _split_text_into_chunks(
        text: str, tokenizer: Any, max_tokens: int, model_config: Any
    ) -> List[str]:
        """Dynamic chunking based on model config."""

        if len(tokenizer.encode(text)) <= max_tokens:
            return [text]

        chunk_logic = getattr(model_config, "chunk_logic", "sentence")
        overlap = getattr(model_config, "chunk_overlap", 1)

        if chunk_logic == "sentence":
            return SentenceTransformer._chunk_by_sentences(
                text, tokenizer, max_tokens, overlap
            )
        elif chunk_logic == "paragraph":
            return SentenceTransformer._chunk_by_paragraphs(
                text, tokenizer, max_tokens, overlap
            )
        elif chunk_logic == "fixed":
            chunk_size = getattr(model_config, "chunk_size", max_tokens // 2)
            return SentenceTransformer._chunk_fixed_size(
                text, tokenizer, chunk_size, overlap
            )
        else:
            # Default fallback
            return SentenceTransformer._chunk_by_sentences(
                text, tokenizer, max_tokens, overlap
            )

    @staticmethod
    def _chunk_by_sentences(
        text: str, tokenizer: Any, max_tokens: int, overlap: int = 1
    ) -> List[str]:
        """Sentence-based chunking with overlap (your original logic)."""
        logger.debug("Splitting text into overlapping sentence chunks.")

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        chunks: List[str] = []
        i = 0

        while i < len(sentences):
            chunk_sentences = []
            j = i

            while j < len(sentences):
                candidate = " ".join(chunk_sentences + [sentences[j]])
                tokens = tokenizer.encode(candidate)

                if len(tokens) < max_tokens:
                    chunk_sentences.append(sentences[j])
                    j += 1
                else:
                    break

            if chunk_sentences:
                chunks.append(". ".join(chunk_sentences) + ".")

            # Move forward with overlap
            if j == i:  # Single sentence too long
                i += 1
            else:
                i = max(i + 1, j - overlap)  # Overlap sentences

        logger.debug(f"Split text into {len(chunks)} sentence chunks.")
        return chunks

    @staticmethod
    def _chunk_by_paragraphs(
        text: str, tokenizer: Any, max_tokens: int, overlap: int = 0
    ) -> List[str]:
        """Paragraph-based chunking."""
        logger.debug("Splitting text into paragraph chunks.")

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        i = 0

        while i < len(paragraphs):
            chunk_paragraphs = []
            j = i

            while j < len(paragraphs):
                candidate = "\n\n".join(chunk_paragraphs + [paragraphs[j]])
                tokens = tokenizer.encode(candidate)

                if len(tokens) < max_tokens:
                    chunk_paragraphs.append(paragraphs[j])
                    j += 1
                else:
                    break

            if chunk_paragraphs:
                chunks.append("\n\n".join(chunk_paragraphs))

            # Move forward with overlap
            if j == i:
                i += 1
            else:
                i = max(i + 1, j - overlap)

        logger.debug(f"Split text into {len(chunks)} paragraph chunks.")
        return chunks

    @staticmethod
    def _chunk_fixed_size(
        text: str, tokenizer: Any, chunk_size: int, overlap: int = 0
    ) -> List[str]:
        """Fixed-size chunking with character estimation."""
        logger.debug(f"Splitting text into fixed chunks of {chunk_size} tokens.")

        # Estimate characters per token (varies by tokenizer)
        char_per_token = 4  # Conservative estimate
        char_chunk_size = chunk_size * char_per_token
        char_overlap = overlap * char_per_token

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + char_chunk_size
            chunk = text[start:end]

            # Validate and adjust token count
            while len(tokenizer.encode(chunk)) > chunk_size and len(chunk) > 10:
                chunk = chunk[: int(len(chunk) * 0.9)]

            if chunk.strip():
                chunks.append(chunk.strip())

            # Move start position with overlap
            start = end - char_overlap if overlap > 0 else end

        logger.debug(f"Split text into {len(chunks)} fixed-size chunks.")
        return chunks

    @staticmethod
    def _small_text_embedding(
        small_text: str,
        model_config: Any,
        tokenizer: Any,
        session: Any,
        projected_dimension: int = 128,
        **kwargs,
    ) -> _EmbeddingResults:
        """Generate embedding for a single text chunk."""
        results = _EmbeddingResults(
            EmbeddingResults=[],
            message="Embedding generated successfully",
            success=True,
        )

        try:
            input_names = getattr(model_config, "inputnames", {})
            output_names = getattr(model_config, "outputnames", {})
            max_length = getattr(input_names, "max_length", 128)

            # Tokenize
            processed_text = SentenceTransformer._preprocess_text(small_text)
            encoding = tokenizer(
                processed_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )

            input_ids = encoding["input_ids"].astype(np.int64)
            attention_mask = encoding.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.astype(np.int64)

            # Prepare ONNX inputs
            inputs = {getattr(input_names, "input", "input_ids"): input_ids}

            if attention_mask is not None:
                inputs[getattr(input_names, "mask", "attention_mask")] = attention_mask

            # Add position_ids if required
            position_name = getattr(input_names, "position", None)
            if position_name:
                seq_len = input_ids.shape[1]
                position_ids = np.arange(seq_len, dtype=np.int64)[None, :]
                inputs[position_name] = position_ids

            # Add token_type_ids if required
            tokentype_name = getattr(input_names, "tokentype", None)
            if tokentype_name:
                seq_len = input_ids.shape[1]
                token_type_ids = np.zeros((1, seq_len), dtype=np.int64)
                inputs[tokentype_name] = token_type_ids

            # Add decoder_input_ids if required
            use_decoder_input = getattr(input_names, "use_decoder_input", False)
            if use_decoder_input:
                seq_len = input_ids.shape[1]
                decoder_input_ids = np.zeros((1, seq_len), dtype=np.int64)
                decoder_input_name = getattr(
                    input_names, "decoder_input_name", "decoder_input_ids"
                )
                inputs[decoder_input_name] = decoder_input_ids

            # Run inference
            outputs = session.run(None, inputs)
            SentenceTransformer._log_onnx_outputs(outputs, session)

            # Process output
            embedding = outputs[0]
            if getattr(
                output_names, "logits", False
            ) or SentenceTransformer._is_logits_output(outputs, session):
                embedding = SentenceTransformer._softmax(embedding)

            # Apply pooling
            pooling_strategy = getattr(model_config, "pooling_strategy", "mean")
            embedding = SentenceTransformer._pooling(embedding, pooling_strategy)

            # Normalize if required
            if getattr(model_config, "normalize", True):
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            # Project dimensions if needed
            if projected_dimension > 0 and embedding.shape[-1] != projected_dimension:
                embedding = SentenceTransformer._project_embedding(
                    embedding, projected_dimension
                )

            results.EmbeddingResults = embedding.flatten().tolist()

        except Exception as e:
            results.message = f"Error generating embedding: {e}"
            results.success = False
            results.EmbeddingResults = np.zeros(projected_dimension).tolist()
            logger.error(f"Embedding error: {e}")

        return results

    @staticmethod
    def _project_embedding(
        embedding: np.ndarray, projected_dimension: int
    ) -> np.ndarray:
        """Project embedding to target dimension using fixed random matrix."""
        input_dim = embedding.shape[-1]
        rng = np.random.default_rng(seed=42)
        random_matrix = rng.uniform(-1, 1, (input_dim, projected_dimension))
        return np.dot(embedding, random_matrix)

    @staticmethod
    def _truncate_text_to_token_limit(
        text: str, tokenizer: Any, max_tokens: int = 128
    ) -> str:
        """Backward compatibility for tests."""
        return SentenceTransformer._preprocess_text(text)[: max_tokens * 4]

    @staticmethod
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
            logger.exception("Unexpected error during embedding")
        finally:
            response.time_taken = time.time() - start_time
            return response

    @staticmethod
    async def embed_batch_async(
        requests: EmbeddingBatchRequest, **kwargs: Any
    ) -> EmbeddingBatchResponse:
        """Asynchronous batch embedding."""
        start_time = time.time()
        response = EmbeddingBatchResponse(
            success=True,
            message="Batch embedding generated successfully",
            model=requests.model,
            results=[],
            time_taken=0.0,
        )

        try:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    None,
                    functools.partial(
                        SentenceTransformer._embed_text_local,
                        text=input_text,
                        model=requests.model,
                        projected_dimension=requests.projected_dimension,
                        join_chunks=requests.join_chunks,
                        join_by_pooling_strategy=requests.join_by_pooling_strategy,
                        output_large_text_upon_join=requests.output_large_text_upon_join,
                        **kwargs,
                    ),
                )
                for input_text in requests.inputs
            ]

            results = await asyncio.gather(*tasks)
            for result in results:
                if result and result.success:
                    response.results.extend(result.EmbeddingResults)
                else:
                    response.success = False
                    response.message = getattr(result, "message", "Unknown error")
                    break

        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
            logger.exception("Unexpected error during batch embedding")
        finally:
            response.time_taken = time.time() - start_time
            return response

    @staticmethod
    def _embed_text_local(
        text: str,
        model: str,
        projected_dimension: int,
        join_chunks: bool = True,
        join_by_pooling_strategy: str = None,
        output_large_text_upon_join: bool = False,
        **kwargs: Any,
    ) -> _EmbeddingResults:
        """Core embedding logic for text processing."""
        results = _EmbeddingResults(
            EmbeddingResults=[],
            message="Embedding generated successfully",
            success=True,
        )

        try:
            model_config = SentenceTransformer._get_model_config(model)
            model_to_use_path = os.path.join(
                SentenceTransformer._root_path,
                "models",
                getattr(model_config, "embedder_task", "fe"),
                model,
            )

            use_legacy = getattr(model_config, "legacy_tokenizer", False)
            tokenizer = SentenceTransformer._get_tokenizer_threadsafe(
                model_to_use_path, use_legacy
            )
            if not tokenizer:
                results.success = False
                results.message = f"Failed to load tokenizer: {model_to_use_path}"
                return results

            # Choose optimized or regular model based on flag
            use_optimized = getattr(model_config, "use_optimized", False)
            if use_optimized:
                model_filename = getattr(
                    model_config, "encoder_optimized_onnx_model", "model_optimized.onnx"
                )
            else:
                model_filename = (
                    getattr(model_config, "encoder_onnx_model", None) or "model.onnx"
                )

            model_path = os.path.join(model_to_use_path, model_filename)

            session = SentenceTransformer._get_encoder_session(model_path)
            if not session:
                results.success = False
                results.message = f"Failed to load ONNX session: {model_path}"
                return results

            max_tokens = getattr(
                getattr(model_config, "inputnames", {}), "max_length", 128
            )
            chunks = SentenceTransformer._split_text_into_chunks(
                text, tokenizer, max_tokens, model_config
            )

            # Process each chunk
            for chunk in chunks:
                embedding_result = SentenceTransformer._small_text_embedding(
                    small_text=chunk,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    session=session,
                    projected_dimension=projected_dimension,
                    **kwargs,
                )

                if not embedding_result.success:
                    results.success = False
                    results.message = embedding_result.message
                    break

                results.EmbeddingResults.append(
                    EmbededChunk(vector=embedding_result.EmbeddingResults, chunk=chunk)
                )

            # Join chunks if requested
            if join_chunks and len(results.EmbeddingResults) > 1:
                pooling_strategy = join_by_pooling_strategy or getattr(
                    model_config, "pooling_strategy", "mean"
                )
                if pooling_strategy not in ["mean", "max"]:
                    pooling_strategy = "mean"

                merged_vector = SentenceTransformer._merge_vectors(
                    results.EmbeddingResults, pooling_strategy
                )
                results.EmbeddingResults = [
                    EmbededChunk(
                        vector=merged_vector,
                        joined_chunk=True,
                        only_vector=not output_large_text_upon_join,
                        chunk="" if not output_large_text_upon_join else text,
                    )
                ]

        except Exception as e:
            results.success = False
            results.message = f"Error generating embedding: {str(e)}"
            logger.exception("Unexpected error during embedding")

        return results
