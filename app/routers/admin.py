# =============================================================================
# File: admin.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.logger import get_logger
from app.utils.key_manager import key_manager
from app.utils.log_sanitizer import sanitize_for_log

logger = get_logger("admin")
router = APIRouter(prefix="/admin")


def verify_admin_access(request: Request):
    """Verify client has admin access using optimized checks."""
    try:
        # Fast path: check client_id directly with admin cache
        client_id = getattr(request.state, "client_id", None)
        if not client_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        # Use optimized admin check
        if not key_manager.is_admin(client_id):
            logger.warning(
                "Non-admin client attempted admin access: %s",
                sanitize_for_log(client_id),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    return True


class ClientKeyRequest(BaseModel):
    client_id: str


class ClientKeyResponse(BaseModel):
    client_id: str
    api_key: str


class ClientListResponse(BaseModel):
    clients: List[str]


@router.post("/generate-key", response_model=ClientKeyResponse)
async def generate_client_key(
    req: ClientKeyRequest, request: Request, _: bool = Depends(verify_admin_access)
):
    """Generate API key for a client."""
    try:
        # For now, return message since we're using JSON file approach
        return {"message": "Use JSON file to manage clients"}
    except (ValueError, TypeError) as e:
        logger.error("Invalid client key parameters: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid request parameters")
    except (PermissionError, OSError) as e:
        logger.error("Database access error during key generation: %s", str(e))
        raise HTTPException(status_code=500, detail="Database access denied")
    except Exception as e:
        logger.error("Key generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to generate client key")


@router.delete("/remove-client/{client_id}")
async def remove_client(
    client_id: str, request: Request, _: bool = Depends(verify_admin_access)
):
    """Remove client from database."""
    if key_manager.remove_client(client_id):
        return {"message": f"Client removed: {client_id}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Client not found"
        )


@router.get("/clients", response_model=ClientListResponse)
async def list_clients(request: Request, _: bool = Depends(verify_admin_access)):
    """List all clients with API keys."""
    clients = list(key_manager.clients.keys())
    return ClientListResponse(clients=clients)
