# =============================================================================
# File: test_extract_embed.py
# Date: 2025-12-23
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_extract_embed.py
# Date: 2025-12-22
# Copyright (c) 2025 Goutam Malakar. All rights reserved.
# =============================================================================

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from app.models.embedding_response import EmbeddingBatchResponse
from app.models.extract_embed_request import ExtractEmbedRequest
from app.models.extracted_file_content import ExtractedFileContent
from app.models.extracted_response import ExtractedResponse
from app.routers.extract_embed import (
    _create_batch_embedding_request,
    _extract_text_from_file,
    extract_and_embed,
    extract_file_and_embed,
)


@pytest.fixture(autouse=True)
def isolate_tests():
    import logging

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


# Helper function tests
def test_extract_text_from_file_success():
    """Test successful text extraction from file."""
    mock_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Page 1 content", item_number=1, content_as="pages"
            ),
            ExtractedFileContent(
                content="Page 2 content", item_number=2, content_as="pages"
            ),
        ],
        time_taken=0.1,
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_response,
    ):
        file_content = base64.b64encode(b"dummy pdf content").decode()
        result = _extract_text_from_file(file_content, "pdf")

        assert len(result) == 2
        assert result[0].content == "Page 1 content"
        assert result[0].item_number == 1
        assert result[0].content_as == "pages"
        assert result[1].content == "Page 2 content"
        assert result[1].item_number == 2
        assert result[1].content_as == "pages"


def test_extract_text_from_file_accepts_bytes():
    """Ensure raw bytes content is accepted and passed through."""
    mock_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Page 1 content", item_number=1, content_as="pages"
            ),
        ],
        time_taken=0.1,
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_response,
    ) as mock_extract:
        file_content = b"dummy pdf content"
        result = _extract_text_from_file(file_content, "pdf")

        assert len(result) == 1
        assert result[0].content == "Page 1 content"
        # Verify bytes were preserved on the request model
        called_request = mock_extract.call_args[0][0]
        assert called_request.file_content == file_content
        assert isinstance(called_request.file_content, bytes)


def test_extract_text_from_file_accepts_base64_string():
    """Ensure base64 string content is accepted without coercion to bytes."""
    mock_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Page 1 content", item_number=1, content_as="pages"
            ),
        ],
        time_taken=0.1,
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_response,
    ) as mock_extract:
        file_content = base64.b64encode(b"dummy pdf content").decode()
        result = _extract_text_from_file(file_content, "pdf")

        assert len(result) == 1
        assert result[0].content == "Page 1 content"
        called_request = mock_extract.call_args[0][0]
        assert called_request.file_content == file_content
        assert isinstance(called_request.file_content, str)


def test_extract_text_from_file_no_results():
    """Test extraction when no content is found."""
    mock_response = ExtractedResponse(
        success=True, message="Extraction successful", results=[], time_taken=0.1
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_response,
    ):
        file_content = base64.b64encode(b"dummy pdf content").decode()

        with pytest.raises(HTTPException) as exc_info:
            _extract_text_from_file(file_content, "pdf")

        assert exc_info.value.status_code == 400
        assert "Failed to extract text" in exc_info.value.detail


def test_extract_text_from_file_extraction_failed():
    """Test extraction when extraction fails."""
    mock_response = ExtractedResponse(
        success=False, message="Extraction failed", results=[], time_taken=0.1
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_response,
    ):
        file_content = base64.b64encode(b"dummy pdf content").decode()

        with pytest.raises(HTTPException) as exc_info:
            _extract_text_from_file(file_content, "pdf")

        assert exc_info.value.status_code == 400
        assert "Failed to extract text" in exc_info.value.detail


def test_create_batch_embedding_request_all_params():
    """Test creating batch embedding request with all parameters."""
    extracted_contents = [
        ExtractedFileContent(content="Test text 1", item_number=1, content_as="pages"),
        ExtractedFileContent(content="Test text 2", item_number=2, content_as="pages"),
    ]

    result = _create_batch_embedding_request(
        extracted_contents=extracted_contents,
        model="test-model",
        projected_dimension=256,
        join_chunks=True,
        join_by_pooling_strategy="mean",
        output_large_text_upon_join=True,
        pooling_strategy="max",
        max_length=512,
        chunk_logic="sentence",
        chunk_overlap=10,
        chunk_size=100,
        legacy_tokenizer=False,
        normalize=True,
        force_pooling=False,
        lowercase=True,
        remove_emojis=True,
        use_optimized=True,
    )

    assert result.inputs == ["Test text 1", "Test text 2"]
    assert result.model == "test-model"
    assert result.projected_dimension == 256
    assert result.join_chunks is True
    assert result.pooling_strategy == "max"
    assert result.max_length == 512
    assert result.chunk_logic == "sentence"
    assert result.normalize is True
    assert result.lowercase is True
    assert result.use_optimized is True


def test_create_batch_embedding_request_minimal_params():
    """Test creating batch embedding request with minimal parameters."""
    extracted_contents = [
        ExtractedFileContent(content="Test text", item_number=1, content_as="text"),
    ]

    result = _create_batch_embedding_request(
        extracted_contents=extracted_contents, model="test-model"
    )

    assert result.inputs == ["Test text"]
    assert result.model == "test-model"
    assert result.projected_dimension is None
    assert result.join_chunks is False


# Endpoint tests
@pytest.mark.asyncio
async def test_extract_and_embed_success():
    """Test successful extract and embed operation."""
    from app.models.embedded_chunk import EmbededChunk

    mock_extraction_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Test content", item_number=1, content_as="pages"
            )
        ],
        time_taken=0.1,
    )

    mock_embedding_response = EmbeddingBatchResponse(
        success=True,
        message="Embedding successful",
        results=[EmbededChunk(chunk="Test content", vector=[0.1, 0.2])],
        used_parameters={"model": "test-model"},
    )

    file_content = base64.b64encode(b"dummy pdf content").decode()
    request = ExtractEmbedRequest(
        file_content=file_content, extention="pdf", model="test-model"
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ), patch(
        "app.routers.extract_embed.SentenceTransformer.embed_batch_async",
        return_value=mock_embedding_response,
    ):
        response = await extract_and_embed(request)

        assert response.success is True
        assert response.message == "Embedding successful"
        assert response.results[0].item_number == 1
        assert response.results[0].content_as == "pages"


@pytest.mark.asyncio
async def test_extract_and_embed_extraction_fails():
    """Test extract and embed when extraction fails."""
    mock_extraction_response = ExtractedResponse(
        success=False, message="Extraction failed", results=[], time_taken=0.1
    )

    file_content = base64.b64encode(b"dummy pdf content").decode()
    request = ExtractEmbedRequest(
        file_content=file_content, extention="pdf", model="test-model"
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await extract_and_embed(request)

        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_extract_and_embed_with_all_parameters():
    """Test extract and embed with all embedding parameters."""
    from app.models.embedded_chunk import EmbededChunk

    mock_extraction_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Test content", item_number=1, content_as="pages"
            )
        ],
        time_taken=0.1,
    )

    mock_embedding_response = EmbeddingBatchResponse(
        success=True,
        message="Embedding successful",
        results=[EmbededChunk(chunk="Test content", vector=[0.1, 0.2])],
        used_parameters={"model": "test-model"},
    )

    file_content = base64.b64encode(b"dummy pdf content").decode()
    request = ExtractEmbedRequest(
        file_content=file_content,
        extention="pdf",
        model="test-model",
        projected_dimension=256,
        join_chunks=True,
        pooling_strategy="mean",
        max_length=512,
        chunk_logic="sentence",
        normalize=True,
        use_optimized=True,
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ), patch(
        "app.routers.extract_embed.SentenceTransformer.embed_batch_async",
        return_value=mock_embedding_response,
    ) as mock_embed:
        response = await extract_and_embed(request)

        assert response.success is True
        # Verify embedding was called with correct parameters
        called_request = mock_embed.call_args[0][0]
        assert called_request.inputs == ["Test content"]
        assert called_request.model == "test-model"
        assert called_request.projected_dimension == 256
        assert called_request.normalize is True


@pytest.mark.asyncio
async def test_extract_file_and_embed_success():
    """Test successful file upload extract and embed."""
    from app.models.embedded_chunk import EmbededChunk

    mock_extraction_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Uploaded file content", item_number=1, content_as="pages"
            )
        ],
        time_taken=0.1,
    )

    mock_embedding_response = EmbeddingBatchResponse(
        success=True,
        message="Embedding successful",
        results=[EmbededChunk(chunk="Uploaded file content", vector=[0.1, 0.2])],
        used_parameters={"model": "test-model"},
    )

    file_content = b"dummy pdf content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)
    mock_file.filename = "test.pdf"

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ), patch(
        "app.routers.extract_embed.SentenceTransformer.embed_batch_async",
        return_value=mock_embedding_response,
    ):
        response = await extract_file_and_embed(
            file=mock_file, extension="pdf", model="test-model"
        )

        assert response.success is True
        assert response.message == "Embedding successful"
        assert response.results[0].item_number == 1
        assert response.results[0].content_as == "pages"


@pytest.mark.asyncio
async def test_extract_file_and_embed_auto_detect_extension():
    """Test file upload with automatic extension detection."""
    from app.models.embedded_chunk import EmbededChunk

    mock_extraction_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Test content", item_number=1, content_as="pages"
            )
        ],
        time_taken=0.1,
    )

    mock_embedding_response = EmbeddingBatchResponse(
        success=True,
        message="Embedding successful",
        results=[EmbededChunk(chunk="Test content", vector=[0.1, 0.2])],
        used_parameters={"model": "test-model"},
    )

    file_content = b"dummy docx content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)
    mock_file.filename = "document.docx"

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ) as mock_extract, patch(
        "app.routers.extract_embed.SentenceTransformer.embed_batch_async",
        return_value=mock_embedding_response,
    ):
        response = await extract_file_and_embed(file=mock_file, model="test-model")

        assert response.success is True
        # Verify the extension was auto-detected
        called_request = mock_extract.call_args[0][0]
        assert called_request.extention == "docx"


@pytest.mark.asyncio
async def test_extract_file_and_embed_no_filename_extension():
    """Test file upload with no filename extension defaults to txt."""
    from app.models.embedded_chunk import EmbededChunk

    mock_extraction_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Test content", item_number=1, content_as="text"
            )
        ],
        time_taken=0.1,
    )

    mock_embedding_response = EmbeddingBatchResponse(
        success=True,
        message="Embedding successful",
        results=[EmbededChunk(chunk="Test content", vector=[0.1, 0.2])],
        used_parameters={"model": "test-model"},
    )

    file_content = b"plain text content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)
    mock_file.filename = "document"

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ) as mock_extract, patch(
        "app.routers.extract_embed.SentenceTransformer.embed_batch_async",
        return_value=mock_embedding_response,
    ):
        response = await extract_file_and_embed(file=mock_file, model="test-model")

        assert response.success is True
        # Verify default extension is txt
        called_request = mock_extract.call_args[0][0]
        assert called_request.extention == "txt"


@pytest.mark.asyncio
async def test_extract_file_and_embed_with_form_parameters():
    """Test file upload with all form parameters."""
    from app.models.embedded_chunk import EmbededChunk

    mock_extraction_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Test content", item_number=1, content_as="pages"
            )
        ],
        time_taken=0.1,
    )

    mock_embedding_response = EmbeddingBatchResponse(
        success=True,
        message="Embedding successful",
        results=[EmbededChunk(chunk="Test content", vector=[0.1, 0.2])],
        used_parameters={"model": "test-model"},
    )

    file_content = b"dummy pdf content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)
    mock_file.filename = "test.pdf"

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ), patch(
        "app.routers.extract_embed.SentenceTransformer.embed_batch_async",
        return_value=mock_embedding_response,
    ) as mock_embed:
        response = await extract_file_and_embed(
            file=mock_file,
            extension="pdf",
            model="test-model",
            projected_dimension=128,
            join_chunks=False,
            pooling_strategy="cls",
            max_length=256,
            normalize=True,
            lowercase=False,
            use_optimized=True,
        )

        assert response.success is True
        # Verify all parameters were passed correctly
        called_request = mock_embed.call_args[0][0]
        assert called_request.model == "test-model"
        assert called_request.projected_dimension == 128
        assert called_request.pooling_strategy == "cls"
        assert called_request.max_length == 256
        assert called_request.normalize is True
        assert called_request.lowercase is False
        assert called_request.use_optimized is True


@pytest.mark.asyncio
async def test_extract_file_and_embed_extraction_fails():
    """Test file upload when extraction fails."""
    mock_extraction_response = ExtractedResponse(
        success=False, message="Extraction failed", results=[], time_taken=0.1
    )

    file_content = b"dummy pdf content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)
    mock_file.filename = "test.pdf"

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await extract_file_and_embed(file=mock_file, extension="pdf")

        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_extract_and_embed_multiple_pages():
    """Test extract and embed with multiple pages of content."""
    from app.models.embedded_chunk import EmbededChunk

    mock_extraction_response = ExtractedResponse(
        success=True,
        message="Extraction successful",
        results=[
            ExtractedFileContent(
                content="Page 1 content", item_number=1, content_as="pages"
            ),
            ExtractedFileContent(
                content="Page 2 content", item_number=2, content_as="pages"
            ),
            ExtractedFileContent(
                content="Page 3 content", item_number=3, content_as="pages"
            ),
        ],
        time_taken=0.1,
    )

    mock_embedding_response = EmbeddingBatchResponse(
        success=True,
        message="Embedding successful",
        results=[
            EmbededChunk(chunk="Page 1 content", vector=[0.1, 0.2]),
            EmbededChunk(chunk="Page 2 content", vector=[0.3, 0.4]),
            EmbededChunk(chunk="Page 3 content", vector=[0.5, 0.6]),
        ],
        used_parameters={"model": "test-model"},
    )

    file_content = base64.b64encode(b"dummy pdf content").decode()
    request = ExtractEmbedRequest(
        file_content=file_content, extention="pdf", model="test-model"
    )

    with patch(
        "app.routers.extract_embed.ExtractorService.extract_text",
        return_value=mock_extraction_response,
    ), patch(
        "app.routers.extract_embed.SentenceTransformer.embed_batch_async",
        return_value=mock_embedding_response,
    ) as mock_embed:
        response = await extract_and_embed(request)

        assert response.success is True
        # Verify batch request contains all pages
        called_request = mock_embed.call_args[0][0]
        assert called_request.inputs == [
            "Page 1 content",
            "Page 2 content",
            "Page 3 content",
        ]
        # Verify metadata was added to each chunk
        assert len(response.results) == 3
        assert response.results[0].item_number == 1
        assert response.results[0].content_as == "pages"
        assert response.results[1].item_number == 2
        assert response.results[1].content_as == "pages"
        assert response.results[2].item_number == 3
        assert response.results[2].content_as == "pages"
