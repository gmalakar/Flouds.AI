# =============================================================================
# File: sendprompt.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from fastapi import APIRouter, HTTPException

from app.exceptions import FloudsBaseException
from app.logger import get_logger
from app.models.prompt_request import PromptRequest
from app.models.prompt_response import PromptResponse
from app.services.prompt_service import PromptProcessor
from app.utils.error_handler import ErrorHandler

router = APIRouter()
logger = get_logger("sendprompt")


@router.post("/sendprompt", response_model=PromptResponse)
async def send_prompt(request: PromptRequest) -> PromptResponse:
    """Process input string prompt using PromptProcessor."""
    try:
        logger.info("Processing prompt for model %s: %s", request.model, request.input[:100])

        # Use PromptProcessor to process the prompt
        response = PromptProcessor.process_prompt(request)
        return response

    except FloudsBaseException as e:
        status_code = ErrorHandler.get_http_status(e)
        raise HTTPException(status_code=status_code, detail=e.message)
    except Exception as e:
        logger.error("Error processing prompt: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process prompt")
