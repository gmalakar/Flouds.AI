import os
import re
import time
import asyncio
from typing import Any, List, Optional

import nltk
import numpy as np
import onnxruntime as ort
from nltk.corpus import stopwords
from transformers import AutoTokenizer

from app.config.config_loader import ConfigLoader
from app.logger import get_logger
from app.models.embedded_chunk import EmbededChunk
from app.models.embedding_request import EmbeddingRequest
from app.models.embedding_response import EmbeddingResponse
from app.modules.concurrent_dict import ConcurrentDict
from app.services.base_nlp_service import BaseNLPService

logger = get_logger("sentence_transformer_wrapper")

# Example stop words set; expand as needed
STOP_WORDS = set(stopwords.words("english"))

# HINTS:
# - All static methods use type hints for clarity and editor support.
# - embed_text returns an EmbeddingResponse and is static.
# - embed_batch_async is async and returns List[EmbeddingResponse].
# - _split_text_into_chunks returns List[str].
# - _small_text_embedding returns list (of floats).
# - _get_tokenizer, _get_encoder_session, _get_model_config are static and type hinted.
# - Use debug logs for tracing.
# - This class is thread-safe and caches tokenizers and ONNX sessions for efficiency.
# - embed_batch_async uses run_in_executor to parallelize embedding for each request.

class SentenceTransformer(BaseNLPService):
    """
    Static class for sentence embedding using ONNX or HuggingFace models.
    """

    @staticmethod
    def _preprocess_text(text: str) -> str:
        logger.debug("Preprocessing text for embedding.")
        text = text.lower()
        text = re.sub(r"[^\w\s.]", "", text)
        tokens = [word for word in text.split() if word not in STOP_WORDS]
        return " ".join(tokens)

    @staticmethod
    def _pooling(embedding: np.ndarray, strategy: str = "mean") -> np.ndarray:
        """
        Generalized pooling for n-dimensional embeddings.
        - "mean": mean over all axes except the last
        - "max": max over all axes except the last
        - "cls": first vector along all axes except the last (like [0,0,...,0,:])
        """
        if embedding.ndim == 1:
            return embedding
        elif strategy == "cls":
            idx = tuple(0 for _ in range(embedding.ndim - 1)) + (slice(None),)
            return embedding[idx]
        elif strategy == "max":
            axes = tuple(range(embedding.ndim - 1))
            return embedding.max(axis=axes)
        else:
            axes = tuple(range(embedding.ndim - 1))
            return embedding.mean(axis=axes)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax for 1D, 2D, or 3D numpy arrays.
        Softmax is always applied along the last axis.
        """
        x = np.asarray(x)
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
        return e_x / sum_e_x

    @staticmethod
    def _truncate_text_to_token_limit(
        text: str, tokenizer: Any, max_tokens: int = 128
    ) -> str:
        logger.debug(f"Truncating text to max {max_tokens} tokens.")
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        current_chunk = ""
        tokenized = False

        for sentence in sentences:
            candidate = (current_chunk + sentence).strip()
            token_count = len(tokenizer.encode(candidate))
            if token_count < max_tokens:
                current_chunk = candidate + ". "
                tokenized = True
            else:
                text = current_chunk.strip()
                break

        if not tokenized and len(text) > max_tokens:
            text = text[:max_tokens]

        return text.strip()

    @staticmethod
    def _small_text_embedding(
        small_text: str,
        model_config: Any,
        tokenizer: Any,
        session: Any,
        projected_dimension: int,
    ) -> list:
        """
        Generates an embedding for a small text chunk and returns it as a Python list.
        """
        try:
            input_names = getattr(model_config, "inputnames", {})
            output_names = getattr(model_config, "outputnames", {})
            logger.debug(f"Using input name: {input_names} for ONNX session.")
            max_length = getattr(input_names, "max_length", 128)
            processed_text = SentenceTransformer._truncate_text_to_token_limit(
                SentenceTransformer._preprocess_text(small_text), tokenizer, max_length
            )

            logger.debug(f"Tokenizing processed text: {processed_text[:50]}...")
            encoding = tokenizer(
                processed_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding.get("attention_mask", None)

            # Prepare ONNX inputs
            inputs = {getattr(input_names, "input", "input_ids"): input_ids}
            if attention_mask is not None:
                inputs[getattr(input_names, "mask", "attention_mask")] = attention_mask

            # Add position_ids if required
            position_name = getattr(input_names, "position", None)
            if position_name:
                seq_len = input_ids.shape[1]
                position_ids = np.arange(seq_len)[None, :]
                inputs[position_name] = position_ids

            # Add token_type_ids if required
            tokentype_name = getattr(input_names, "tokentype", None)
            if tokentype_name:
                seq_len = input_ids.shape[1]
                token_type_ids = np.zeros((1, seq_len), dtype=np.int64)
                inputs[tokentype_name] = token_type_ids

            # Add decoder_input_ids if required
            use_decoder_input = getattr(input_names, "use_decoder_input", False)
            logger.debug(f"Use decoder inputs: {use_decoder_input}")
            if use_decoder_input:
                seq_len = input_ids.shape[1]
                decoder_input_ids = np.zeros((1, seq_len), dtype=np.int64)
                inputs[
                    getattr(input_names, "decoder_input_name", "decoder_input_ids")
                ] = decoder_input_ids

            logger.debug(
                f"Running ONNX session for embedding for input: {inputs.keys()}..."
            )

            outputs = session.run(None, inputs)
            SentenceTransformer._log_onnx_outputs(outputs, session)

            logits = getattr(output_names, "logits", False)
            if logits or SentenceTransformer._is_logits_output(outputs, session):
                logger.debug("Output is logits, applying softmax.")
                embedding = SentenceTransformer._softmax(outputs[0])
            else:
                embedding = outputs[0]
            logger.debug(
                f"Generated embedding with shape: {embedding.shape}, dtype: {embedding.dtype}, dimension: {embedding.ndim}"
            )

            pooling_strategy = getattr(model_config, "pooling_strategy", "mean")
            embedding = SentenceTransformer._pooling(embedding, pooling_strategy)
            logger.debug(
                f"Pooling strategy: {pooling_strategy}, resulting embedding shape: {embedding.shape}"
            )
            if getattr(model_config, "normalize", True):
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    logger.debug("Normalized embedding vector.")
            logger.debug(
                f"Projecting embedding from {embedding.shape[-1]} to {projected_dimension}"
            )
            if projected_dimension > 0 and embedding.shape[-1] != projected_dimension:
                logger.debug(
                    f"Projecting embedding from {embedding.shape[-1]} to {projected_dimension} dimensions."
                )
                embedding = SentenceTransformer._project_embedding(
                    embedding, projected_dimension
                )

            return embedding.flatten().tolist()
        except Exception:
            logger.exception(f"Error generating embedding")
            return [0.0] * (projected_dimension if projected_dimension > 0 else 128)

    @staticmethod
    def _project_embedding(
        embedding: np.ndarray, projected_dimension: int
    ) -> np.ndarray:
        """
        Projects the embedding to the desired dimension using a random matrix (fixed seed for reproducibility).
        """
        input_dim = embedding.shape[-1]
        rng = np.random.default_rng(seed=42)
        random_matrix = rng.uniform(-1, 1, (input_dim, projected_dimension))
        logger.debug(
            f"Projecting embedding with shape {embedding.shape} to {projected_dimension} dimensions."
        )
        return np.dot(embedding, random_matrix)

    @staticmethod
    def _split_text_into_chunks(
        large_text: str, tokenizer: Any, max_tokens: int, overlap_sentences: int = 1
    ) -> List[str]:
        logger.debug("Splitting large text into overlapping chunks for embedding.")
        sentences = [s.strip() for s in large_text.split(".") if s.strip()]
        chunks: List[str] = []
        i = 0
        while i < len(sentences):
            chunk_sentences = []
            token_count = 0
            j = i
            while j < len(sentences):
                candidate = " ".join(chunk_sentences + [sentences[j]])
                tokens = tokenizer.encode(candidate)
                if len(tokens) < max_tokens:
                    chunk_sentences.append(sentences[j])
                    token_count = len(tokens)
                    j += 1
                else:
                    break
            if chunk_sentences:
                chunks.append(". ".join(chunk_sentences) + ".")
            if j == i:
                i += 1
            else:
                i = j - overlap_sentences if (j - overlap_sentences) > i else j
        logger.debug(f"Split text into {len(chunks)} overlapping chunk(s).")
        return chunks

    @staticmethod
    async def embed_batch_async(requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """
        Asynchronously embed a batch of texts using run_in_executor for concurrency.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                SentenceTransformer.embed_text,
                req.input,
                req.model,
                req.projected_dimension,
            )
            for req in requests
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    def embed_text(
        text: str, model_to_use: str, projected_dimension: int
    ) -> EmbeddingResponse:
        """
        Splits text into chunks and generates embeddings for each chunk.
        Returns a list of dicts: [{"Embedding": embedding, "TextChunk": chunk}, ...]
        """
        response = EmbeddingResponse(
            success=True,
            message="Embedding generated successfully",
            model=model_to_use,
            results=[],
            time_taken=0.0,
        )
        start_time = time.time()
        try:
            model_config = SentenceTransformer._get_model_config(model_to_use)
            logger.debug(
                f"Embedding text using model: {model_to_use}, projected_dimension: {projected_dimension} and task: {getattr(model_config, 'embedder_task', 'fe')}..."
            )
            model_to_use_path = os.path.join(
                SentenceTransformer._root_path,
                "models",
                getattr(model_config, "embedder_task", "fe"),
                model_to_use,
            )
            logger.debug(f"Using model path: {model_to_use_path}")
            tokenizer = SentenceTransformer._get_tokenizer(model_to_use_path)

            model_path = os.path.join(
                model_to_use_path,
                getattr(model_config, "encoder_onnx_model", None) or "model.onnx",
            )
            logger.debug(f"Using ONNX model path: {model_path}")
            session = SentenceTransformer._get_encoder_session(model_path)
            if not session:
                logger.error(
                    f"Failed to get ONNX session for model: {model_to_use} and path: {model_path}"
                )
                return None
            max_tokens = getattr(
                getattr(model_config, "inputnames", {}), "max_length", 128
            )

            logger.debug(f"Splitting text into chunks with max_tokens={max_tokens}")
            chunks = SentenceTransformer._split_text_into_chunks(
                text, tokenizer, max_tokens
            )
            for chunk in chunks:
                logger.debug(f"Generating embedding for chunk: {chunk[:50]}...")
                embedding = SentenceTransformer._small_text_embedding(
                    small_text=chunk,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    session=session,
                    projected_dimension=projected_dimension,
                )
                response.results.append(EmbededChunk(vector=embedding, chunk=chunk))
            return response
        except Exception as e:
            response.success = False
            response.message = f"Error generating embedding: {str(e)}"
            logger.exception("Unexpected error during embedding")
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"Embedding completed in {elapsed:.2f} seconds.")
            response.time_taken = elapsed
            return response


# HINT:
# - Logging is added at key steps for tracing and debugging.
# - Use debug logs for internal steps and info logs for high-level process.
# - This class is thread-safe and caches tokenizers and ONNX sessions for efficiency.
# - Make sure your model config and tokenizer paths are correct for your deployment.
