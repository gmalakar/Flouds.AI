# =============================================================================
# File: chunking_strategies.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import re
from typing import Any, List

import nltk

from app.exceptions import InvalidInputError, TokenizerError
from app.logger import get_logger

logger = get_logger("chunking_strategies")


class ChunkingStrategies:
    """Text chunking strategies for different use cases."""

    @staticmethod
    def split_text_into_chunks(
        text: str, tokenizer: Any, max_tokens: int, model_config: Any
    ) -> List[str]:
        """Dynamic chunking based on model config."""

        try:
            if len(tokenizer.encode(text)) <= max_tokens:
                return [text]
        except Exception:
            # If tokenizer fails, return original text
            return [text]

        try:
            chunk_logic = getattr(model_config, "chunk_logic", "sentence")
            overlap = getattr(model_config, "chunk_overlap", 1)

            if chunk_logic == "sentence":
                return ChunkingStrategies.chunk_by_sentences(text, tokenizer, max_tokens, overlap)
            elif chunk_logic == "paragraph":
                return ChunkingStrategies.chunk_by_paragraphs(text, tokenizer, max_tokens, overlap)
            elif chunk_logic == "fixed":
                chunk_size = getattr(model_config, "chunk_size", max_tokens // 2)
                return ChunkingStrategies.chunk_fixed_size(text, tokenizer, chunk_size, overlap)
            else:
                # Default fallback
                return ChunkingStrategies.chunk_by_sentences(text, tokenizer, max_tokens, overlap)
        except AttributeError as e:
            logger.error("Model config attribute error: %s", str(e))
            return [text]
        except (ValueError, TypeError) as e:
            logger.error("Invalid chunking parameters: %s", str(e))
            return [text]
        except Exception as e:
            logger.error("Chunking failed: %s", str(e))
            raise InvalidInputError(f"Text chunking failed: {e}")

    @staticmethod
    def chunk_by_sentences(
        text: str, tokenizer: Any, max_tokens: int, overlap: int = 1
    ) -> List[str]:
        """Efficient sentence-based chunking with smart overlap and token-aware joining."""
        logger.debug("Splitting text into optimized overlapping sentence chunks (NLTK).")

        # Sentence segmentation
        sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        sentence_tokens = [len(tokenizer.encode(s)) for s in sentences]

        chunks: List[str] = []
        i = 0

        while i < len(sentences):
            chunk_sentences = []
            chunk_token_count = 0
            j = i

            while j < len(sentences) and (chunk_token_count + sentence_tokens[j]) < max_tokens:
                chunk_sentences.append(sentences[j])
                chunk_token_count += sentence_tokens[j]
                j += 1

            if chunk_sentences:
                # Smart join preserving punctuation spacing
                chunk = " ".join(chunk_sentences)
                chunks.append(chunk)

            # Move forward with overlap
            if j == i:  # Handles overly long single sentence
                i += 1
            else:
                i = max(i + 1, j - overlap)

        logger.debug(f"Generated {len(chunks)} semantic chunks.")
        return chunks

    @staticmethod
    def chunk_by_paragraphs(
        text: str, tokenizer: Any, max_tokens: int, overlap: int = 0
    ) -> List[str]:
        """Paragraph-based chunking with overlap and robust splitting."""
        logger.debug("Splitting text into paragraph chunks.")

        # Split paragraphs by double newline, fallback to single newline if needed
        paragraphs = [p.strip() for p in re.split(r"\n{2,}|\r{2,}", text) if p.strip()]
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        chunks: List[str] = []
        i = 0

        while i < len(paragraphs):
            chunk_paragraphs: List[str] = []
            j = i

            while j < len(paragraphs):
                candidate = "\n\n".join(chunk_paragraphs + [paragraphs[j]])
                try:
                    tokens = tokenizer.encode(candidate)
                except (AttributeError, TypeError) as e:
                    logger.error("Tokenizer attribute error: %s", str(e))
                    break
                except (ValueError, KeyError) as e:
                    logger.error("Invalid tokenizer parameters: %s", str(e))
                    break
                except Exception as e:
                    logger.error("Tokenizer error in paragraph chunking: %s", str(e))
                    raise TokenizerError(f"Paragraph chunking tokenizer error: {e}")

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
    def chunk_fixed_size(text: str, tokenizer: Any, chunk_size: int, overlap: int = 0) -> List[str]:
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
            try:
                while len(tokenizer.encode(chunk)) > chunk_size and len(chunk) > 10:
                    chunk = chunk[: int(len(chunk) * 0.9)]
            except (AttributeError, TypeError) as e:
                logger.error("Tokenizer attribute error: %s", str(e))
                break
            except (ValueError, KeyError) as e:
                logger.error("Invalid tokenizer parameters: %s", str(e))
                break
            except Exception as e:
                logger.error("Tokenizer error in fixed chunking: %s", str(e))
                raise TokenizerError(f"Fixed chunking tokenizer error: {e}")

            if chunk.strip():
                chunks.append(chunk.strip())

            # Move start position with overlap
            start = end - char_overlap if overlap > 0 else end

        logger.debug(f"Split text into {len(chunks)} fixed-size chunks.")
        return chunks
