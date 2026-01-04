# =============================================================================
# File: delete_config_request.py
# Simple model for delete config payload
# =============================================================================
from typing import Any, cast

from pydantic import Field

from app.models.base_request import BaseRequest


class DeleteConfigRequest(BaseRequest):
    """Request model for deleting a config entry.

    Fields:
      - key: config key (required)
      - tenant_code: optional tenant code (empty string for default tenant)
    """

    key: str = Field(cast(Any, ...), description="The config key.")  # type: ignore
