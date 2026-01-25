# =============================================================================
# File: extract_embed.py
# Date: 2025-12-22
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================


import base64
from typing import Annotated, Any, Dict, List, Optional, Union

from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.models.embedding_request import EmbeddingBatchRequest
from app.models.embedding_response import EmbeddingBatchResponse
from app.models.extract_embed_request import ExtractEmbedRequest
from app.models.extracted_file_content import ExtractedFileContent
from app.models.file_request import FileRequest
from app.services.embedder import SentenceTransformer
from app.services.extractor_service import ExtractorService
from app.utils.error_handler import ErrorHandler
from app.utils.log_sanitizer import sanitize_for_log

router = APIRouter()
logger = get_logger("router")

# Module-level Form instance to avoid function calls in defaults (B008)
DEFAULT_FORM = Form()


def _extract_text_from_file(
    file_content: Union[str, bytes], extension: str
) -> List[ExtractedFileContent]:
    """
    Extract text from a file with the given content and extension.

    Args:
        file_content: Base64 string or raw bytes file content
        extension: File extension

    Returns:
        List of ExtractedFileContent with text, item_number, and content_as

    Raises:
        HTTPException: If extraction fails or no content found
    """
    file_request = FileRequest(file_content=file_content, extention=extension)
    extraction_response = ExtractorService.extract_text(file_request)

    # Check if extraction was successful
    if not extraction_response.success or not extraction_response.results:
        raise HTTPException(
            status_code=400,
            detail="Failed to extract text from file or no content found",
        )

    return extraction_response.results


def _create_batch_embedding_request(
    extracted_contents: List[ExtractedFileContent],
    model: Optional[str] = None,
    projected_dimension: Optional[int] = None,
    join_chunks: bool = False,
    join_by_pooling_strategy: Optional[str] = None,
    output_large_text_upon_join: bool = False,
    pooling_strategy: Optional[str] = None,
    max_length: Optional[int] = None,
    chunk_logic: Optional[str] = None,
    chunk_overlap: Optional[int] = None,
    chunk_size: Optional[int] = None,
    legacy_tokenizer: Optional[bool] = None,
    normalize: Optional[bool] = None,
    force_pooling: Optional[bool] = None,
    lowercase: Optional[bool] = None,
    remove_emojis: Optional[bool] = None,
) -> EmbeddingBatchRequest:
    """
    Create a batch embedding request from extracted file contents.

    Args:
        extracted_contents: List of extracted file contents
        **kwargs: All embedding configuration parameters

    Returns:
        EmbeddingBatchRequest instance
    """
    texts = [content.content for content in extracted_contents]

    # Build request dict with only non-None values
    request_data: Dict[str, Any] = {
        "inputs": texts,
        "model": model,
        "join_chunks": join_chunks,
        "output_large_text_upon_join": output_large_text_upon_join,
    }

    # Add optional parameters only if they are not None
    if projected_dimension is not None:
        request_data["projected_dimension"] = projected_dimension
    if join_by_pooling_strategy is not None:
        request_data["join_by_pooling_strategy"] = join_by_pooling_strategy
    if pooling_strategy is not None:
        request_data["pooling_strategy"] = pooling_strategy
    if max_length is not None:
        request_data["max_length"] = max_length
    if chunk_logic is not None:
        request_data["chunk_logic"] = chunk_logic
    if chunk_overlap is not None:
        request_data["chunk_overlap"] = chunk_overlap
    if chunk_size is not None:
        request_data["chunk_size"] = chunk_size
    if legacy_tokenizer is not None:
        request_data["legacy_tokenizer"] = legacy_tokenizer
    if normalize is not None:
        request_data["normalize"] = normalize
    if force_pooling is not None:
        request_data["force_pooling"] = force_pooling
    if lowercase is not None:
        request_data["lowercase"] = lowercase
    if remove_emojis is not None:
        request_data["remove_emojis"] = remove_emojis

    return EmbeddingBatchRequest(
        inputs=texts,
        model=model or "",
        tenant_code=None,
        projected_dimension=projected_dimension,
        join_chunks=join_chunks,
        join_by_pooling_strategy=join_by_pooling_strategy,
        output_large_text_upon_join=output_large_text_upon_join,
        pooling_strategy=pooling_strategy,
        max_length=max_length,
        chunk_logic=chunk_logic,
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        legacy_tokenizer=legacy_tokenizer,
        normalize=normalize,
        force_pooling=force_pooling,
        lowercase=lowercase,
        remove_emojis=remove_emojis,
    )


@router.post("/extract_and_embed", response_model=EmbeddingBatchResponse)
async def extract_and_embed(request: ExtractEmbedRequest) -> EmbeddingBatchResponse:
    """
    Combined endpoint that extracts text from a file and then generates embeddings.

    This endpoint:
    1. Extracts text content from the provided file (base64 encoded)
    2. Embeds the extracted text using the specified model and parameters
    3. Returns the embedding batch response with item_number and content_as metadata
    """
    logger.debug(
        "Extract and embed request for file type: %s with model: %s",
        sanitize_for_log(request.extention),
        sanitize_for_log(str(request.model)),
    )
    try:
        # Extract text from file
        extracted_contents = _extract_text_from_file(request.file_content, request.extention)

        # Create batch embedding request with extracted contents
        batch_request = _create_batch_embedding_request(
            extracted_contents=extracted_contents,
            model=request.model,
            projected_dimension=request.projected_dimension,
            join_chunks=(request.join_chunks if request.join_chunks is not None else False),
            join_by_pooling_strategy=request.join_by_pooling_strategy,
            output_large_text_upon_join=(
                request.output_large_text_upon_join
                if request.output_large_text_upon_join is not None
                else False
            ),
            pooling_strategy=request.pooling_strategy,
            max_length=request.max_length,
            chunk_logic=request.chunk_logic,
            chunk_overlap=request.chunk_overlap,
            chunk_size=request.chunk_size,
            legacy_tokenizer=request.legacy_tokenizer,
            normalize=request.normalize,
            force_pooling=request.force_pooling,
            lowercase=request.lowercase,
            remove_emojis=request.remove_emojis,
        )

        # Generate embeddings for all extracted contents
        response: EmbeddingBatchResponse = SentenceTransformer.embed_batch_async(batch_request)

        # Add metadata (item_number and content_as) to each embedding chunk
        for i, chunk in enumerate(response.results):
            if i < len(extracted_contents):
                chunk.item_number = extracted_contents[i].item_number
                chunk.content_as = extracted_contents[i].content_as

        return response

    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in extract file and embed endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/extract_file_and_embed", response_model=EmbeddingBatchResponse)
async def extract_file_and_embed(
    file: UploadFile,
    extension: Annotated[Optional[str], DEFAULT_FORM] = None,
    model: Annotated[Optional[str], DEFAULT_FORM] = None,
    projected_dimension: Annotated[Optional[int], DEFAULT_FORM] = None,
    join_chunks: Annotated[bool, DEFAULT_FORM] = False,
    join_by_pooling_strategy: Annotated[Optional[str], DEFAULT_FORM] = None,
    output_large_text_upon_join: Annotated[bool, DEFAULT_FORM] = False,
    pooling_strategy: Annotated[Optional[str], DEFAULT_FORM] = None,
    max_length: Annotated[Optional[int], DEFAULT_FORM] = None,
    chunk_logic: Annotated[Optional[str], DEFAULT_FORM] = None,
    chunk_overlap: Annotated[Optional[int], DEFAULT_FORM] = None,
    chunk_size: Annotated[Optional[int], DEFAULT_FORM] = None,
    legacy_tokenizer: Annotated[Optional[bool], DEFAULT_FORM] = None,
    normalize: Annotated[Optional[bool], DEFAULT_FORM] = None,
    force_pooling: Annotated[Optional[bool], DEFAULT_FORM] = None,
    lowercase: Annotated[Optional[bool], DEFAULT_FORM] = None,
    remove_emojis: Annotated[Optional[bool], DEFAULT_FORM] = None,
    # removed `use_optimized` parameter
) -> EmbeddingBatchResponse:
    """
    Combined endpoint that extracts text from an uploaded file and then generates embeddings.

    This endpoint:
    1. Accepts a file upload (multipart/form-data)
    2. Extracts text content from the uploaded file
    3. Embeds the extracted text using the specified model and parameters
    4. Returns the embedding batch response with item_number and content_as metadata
    """
    try:
        # Validate file is provided
        if not file or not file.filename:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "missing",
                        "loc": ["body", "file"],
                        "msg": "Field required",
                        "input": None,
                    }
                ],
            )

        file_bytes = await file.read()

        # Validate file content is not empty
        if not file_bytes:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "value_error",
                        "loc": ["body", "file"],
                        "msg": "Uploaded file is empty",
                        "input": None,
                    }
                ],
            )
        file_content = base64.b64encode(file_bytes).decode()

        # Auto-detect extension if not provided
        if not extension:
            extension = (
                file.filename.split(".")[-1] if file.filename and "." in file.filename else "txt"
            )

        logger.debug(
            "Extract file and embed request for file: %s (type: %s) with model: %s",
            sanitize_for_log(file.filename or "unknown"),
            sanitize_for_log(extension),
            sanitize_for_log(str(model)),
        )

        # Extract text from file
        extracted_contents = _extract_text_from_file(file_content, extension)

        # Create batch embedding request with extracted contents
        batch_request = _create_batch_embedding_request(
            extracted_contents=extracted_contents,
            model=model,
            projected_dimension=projected_dimension,
            join_chunks=join_chunks,
            join_by_pooling_strategy=join_by_pooling_strategy,
            output_large_text_upon_join=output_large_text_upon_join,
            pooling_strategy=pooling_strategy,
            max_length=max_length,
            chunk_logic=chunk_logic,
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
            legacy_tokenizer=legacy_tokenizer,
            normalize=normalize,
            force_pooling=force_pooling,
            lowercase=lowercase,
            remove_emojis=remove_emojis,
        )

        # Generate embeddings for all extracted contents
        response: EmbeddingBatchResponse = SentenceTransformer.embed_batch_async(batch_request)

        # Add metadata (item_number and content_as) to each embedding chunk
        for i, chunk in enumerate(response.results):
            if i < len(extracted_contents):
                chunk.item_number = extracted_contents[i].item_number
                chunk.content_as = extracted_contents[i].content_as

        return response

    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in extract file and embed endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
