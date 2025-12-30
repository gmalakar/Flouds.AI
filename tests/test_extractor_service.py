# =============================================================================
# File: test_extractor_service.py
# Date: 2025-12-23
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

# =============================================================================
# File: test_extractor_service.py
# Date: 2025-01-27
# Copyright (c) 2025 Goutam Malakar. All rights reserved.
# =============================================================================

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from app.models.file_request import FileRequest
from app.routers.extractor import extract_file
from app.services.extractor_service import ExtractorService


@pytest.fixture(autouse=True)
def isolate_tests():
    import logging

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


# Service Tests
def test_extract_text_pdf():
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page 1 content"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("app.services.extractor_service.pdfplumber.open", return_value=mock_pdf):
        req = FileRequest(
            file_content=base64.b64encode(b"dummy pdf content").decode(),
            extention="pdf",
        )
        response = ExtractorService.extract_text(req)

        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].content == "Page 1 content"
        assert response.results[0].item_number == 1
        assert response.results[0].content_as == "pages"


def test_extract_text_docx():
    mock_doc = MagicMock()
    mock_paragraph = MagicMock()
    mock_paragraph.text = "Paragraph content"
    mock_doc.paragraphs = [mock_paragraph]

    with patch("app.services.extractor_service.Document", return_value=mock_doc):
        req = FileRequest(
            file_content=base64.b64encode(b"dummy docx content").decode(),
            extention="docx",
        )
        response = ExtractorService.extract_text(req)

        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].content == "Paragraph content"
        assert response.results[0].item_number == 1
        assert response.results[0].content_as == "paragraphs"


def test_extract_text_doc():
    mock_doc = MagicMock()
    mock_paragraph = MagicMock()
    mock_paragraph.text = "DOC paragraph content"
    mock_doc.paragraphs = [mock_paragraph]

    with patch("app.services.extractor_service.Document", return_value=mock_doc):
        req = FileRequest(
            file_content=base64.b64encode(b"dummy doc content").decode(),
            extention="doc",
        )
        response = ExtractorService.extract_text(req)

        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].content == "DOC paragraph content"
        assert response.results[0].content_as == "paragraphs"


def test_extract_text_pptx():
    mock_prs = MagicMock()
    mock_slide = MagicMock()
    mock_shape = MagicMock()
    mock_shape.text = "Slide 1 content"
    mock_slide.shapes = [mock_shape]
    mock_prs.slides = [mock_slide]

    with patch("app.services.extractor_service.Presentation", return_value=mock_prs):
        req = FileRequest(
            file_content=base64.b64encode(b"dummy pptx content").decode(),
            extention="pptx",
        )
        response = ExtractorService.extract_text(req)

        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].content == "Slide 1 content"
        assert response.results[0].content_as == "slides"


def test_extract_text_xlsx():
    mock_wb = MagicMock()
    mock_ws = MagicMock()
    mock_ws.iter_rows.return_value = [("A1", "B1"), ("A2", "B2")]
    mock_wb.__getitem__.return_value = mock_ws
    mock_wb.sheetnames = ["Sheet1"]

    with patch("app.services.extractor_service.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = mock_wb
        req = FileRequest(
            file_content=base64.b64encode(b"dummy xlsx content").decode(),
            extention="xlsx",
        )
        response = ExtractorService.extract_text(req)

        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].content_as == "sheets"


def test_extract_text_txt():
    text_content = "This is plain text content"
    req = FileRequest(
        file_content=base64.b64encode(text_content.encode()).decode(), extention="txt"
    )
    response = ExtractorService.extract_text(req)

    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].content == text_content
    assert response.results[0].content_as == "text"


def test_extract_text_txt_bytes():
    text_content = b"This is plain text content"
    req = FileRequest(file_content=text_content, extention="txt")
    response = ExtractorService.extract_text(req)

    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].content == text_content.decode()
    assert response.results[0].content_as == "text"


def test_extract_text_html():
    html_content = "<html><body><p>HTML content</p></body></html>"
    req = FileRequest(
        file_content=base64.b64encode(html_content.encode()).decode(), extention="html"
    )
    response = ExtractorService.extract_text(req)

    assert response.success is True
    assert len(response.results) == 1
    assert "HTML content" in response.results[0].content
    assert response.results[0].content_as == "text"


def test_extract_text_csv():
    req = FileRequest(
        file_content=base64.b64encode(b"col1,col2\nval1,val2").decode(),
        extention="csv",
    )
    response = ExtractorService.extract_text(req)

    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].content_as == "rows"


def test_extract_text_unsupported_format():
    req = FileRequest(
        file_content=base64.b64encode(b"dummy content").decode(), extention="xyz"
    )
    response = ExtractorService.extract_text(req)

    assert response.success is False
    assert "Unsupported file type" in response.message


def test_extract_text_doc_missing_library():
    with patch(
        "app.services.extractor_service.Document", side_effect=Exception("no docx")
    ):
        req = FileRequest(
            file_content=base64.b64encode(b"dummy doc content").decode(),
            extention="doc",
        )
        response = ExtractorService.extract_text(req)

        assert response.success is False
        assert "Failed to parse" in response.message


def test_extract_text_pptx_missing_library():
    with patch("app.services.extractor_service.Presentation", None):
        req = FileRequest(
            file_content=base64.b64encode(b"dummy pptx content").decode(),
            extention="pptx",
        )
        response = ExtractorService.extract_text(req)

        assert response.success is False
        assert "Install python-pptx" in response.message


# Router Tests
@pytest.mark.asyncio
async def test_extract_file_pdf():
    file_content = b"dummy pdf content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF page content"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("app.services.extractor_service.pdfplumber.open", return_value=mock_pdf):
        response = await extract_file(mock_file, "pdf")

        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].content == "PDF page content"
        assert response.results[0].content_as == "pages"


@pytest.mark.asyncio
async def test_extract_file_txt():
    file_content = b"Plain text content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)

    response = await extract_file(mock_file, "txt")

    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].content == "Plain text content"
    assert response.results[0].content_as == "text"


@pytest.mark.asyncio
async def test_extract_file_unsupported():
    file_content = b"unsupported content"
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=file_content)

    response = await extract_file(mock_file, "xyz")

    assert response.success is False
    assert "Unsupported file type" in response.message


@pytest.mark.asyncio
async def test_extract_file_read_error():
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(side_effect=Exception("File read error"))

    with pytest.raises(HTTPException) as exc_info:
        await extract_file(mock_file, "txt")

    assert exc_info.value.status_code == 500
    assert "Internal server error" in str(exc_info.value.detail)
