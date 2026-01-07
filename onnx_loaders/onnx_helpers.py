import logging
import time
from typing import Optional

try:
    import onnxruntime as ort
except Exception:
    ort = None


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_preferred_provider(default: str = "CPUExecutionProvider") -> str:
    # Allow override via environment variable
    import os

    prov = os.getenv("FLOUDS_ORT_PROVIDER")
    if prov:
        return prov
    return default


def get_default_opset(default: int = 18) -> int:
    import os

    try:
        val = os.getenv("FLOUDS_ONNX_OPSET")
        if val:
            return int(val)
    except Exception:
        pass
    return default


def create_ort_session(
    path: str, provider: Optional[str] = None, retries: int = 2, backoff: float = 1.0
):
    """Create an ONNX Runtime InferenceSession with simple retry/backoff.

    Raises the last exception if all attempts fail.
    """
    logger = get_logger("onnx_helpers")
    if ort is None:
        raise RuntimeError("onnxruntime is not available in the environment")

    providers = ort.get_available_providers()
    if provider is None or provider not in providers:
        provider = get_preferred_provider()
        if provider not in providers:
            provider = "CPUExecutionProvider"

    last_exc = None
    for attempt in range(retries + 1):
        try:
            logger.debug(
                "Creating ORT session for %s with provider=%s (attempt=%d)",
                path,
                provider,
                attempt + 1,
            )
            sess = ort.InferenceSession(path, providers=[provider])
            return sess
        except Exception as e:
            last_exc = e
            logger.warning(
                "ORT session creation failed (attempt %d): %s", attempt + 1, e
            )
            time.sleep(backoff * (1 + attempt))

    # If we reach here, re-raise the last exception
    raise last_exc
