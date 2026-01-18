# =============================================================================
# File: extract.py
# Date: 2025-12-20
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================


import base64

from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.models.extracted_response import ExtractedResponse
from app.models.file_request import FileRequest
from app.services.extractor_service import ExtractorService
from app.utils.error_handler import ErrorHandler

router = APIRouter()
logger = get_logger("router")

# Move Form() default to module-level constant to avoid B008
DEFAULT_EXTENSION_FORM = Form(None)


@router.post("/extract", response_model=ExtractedResponse)
async def extract(request: FileRequest) -> ExtractedResponse:
    try:
        response: ExtractedResponse = ExtractorService.extract_text(request)
        return response
    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception:
        logger.exception("Unexpected error in extract endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/extract_file", response_model=ExtractedResponse)
async def extract_file(
    file: UploadFile, extension: str = DEFAULT_EXTENSION_FORM
) -> ExtractedResponse:
    try:
        file_bytes = await file.read()
        file_content = base64.b64encode(file_bytes).decode()

        # Auto-detect extension if not provided
        if not extension:
            extension = (
                file.filename.split(".")[-1] if file.filename and "." in file.filename else "txt"
            )

        request = FileRequest(file_content=file_content, extention=extension)
        response: ExtractedResponse = ExtractorService.extract_text(request)
        return response
    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception:
        logger.exception("Unexpected error in extract_file endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")
