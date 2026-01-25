# =============================================================================
# File: path_validator.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
import re
from pathlib import Path
from typing import IO, Any, Union

from app.exceptions import ResourceException
from app.logger import get_logger

logger = get_logger("path_validator")

# Dangerous path patterns
DANGEROUS_PATTERNS = [
    r"\.\.",  # Parent directory traversal
    r"~",  # Home directory
    r"\$",  # Environment variables
    r"%",  # Windows environment variables
    r"\\\\",  # UNC paths
]

# Compile patterns for performance
COMPILED_PATTERNS = [re.compile(pattern) for pattern in DANGEROUS_PATTERNS]


def validate_safe_path(file_path: Union[str, Path], base_dir: Union[str, Path]) -> str:
    """
    Validate that file_path is within base_dir to prevent path traversal attacks.

    Args:
        file_path: The file path to validate
        base_dir: The base directory that file_path must be within

    Returns:
        str: The resolved absolute path if safe

    Raises:
        ResourceException: If path traversal is detected or path is unsafe
    """
    try:
        # Convert to strings for initial validation
        file_str = str(file_path)

        # Check for dangerous patterns in raw path
        for pattern in COMPILED_PATTERNS:
            if pattern.search(file_str):
                raise ResourceException(f"Dangerous path pattern detected: {file_str}")

        # Convert to Path objects and resolve
        file_path = Path(file_path).resolve()
        base_dir = Path(base_dir).resolve()

        # Additional validation for resolved paths
        if not base_dir.exists():
            raise ResourceException(f"Base directory does not exist: {base_dir}")

        # Check if file_path is within base_dir
        try:
            file_path.relative_to(base_dir)
        except ValueError:
            raise ResourceException(f"Path traversal detected: {file_path} is outside {base_dir}")

        # Validate path length (prevent extremely long paths)
        if len(str(file_path)) > 4096:
            raise ResourceException("Path too long")

        return str(file_path)

    except OSError as e:
        raise ResourceException(f"Cannot access path: {e}")
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        raise ResourceException(f"Path validation failed: {e}")


def safe_join(base_dir: Union[str, Path], *paths: str) -> str:
    """
    Safely join paths and validate against path traversal.

    Args:
        base_dir: The base directory
        *paths: Path components to join

    Returns:
        str: The safe joined path

    Raises:
        ResourceException: If path traversal is detected
    """
    # Validate each path component
    for path_component in paths:
        if not path_component or path_component.strip() != path_component:
            raise ResourceException(f"Invalid path component: '{path_component}'")

        # Check for dangerous patterns in components
        for pattern in COMPILED_PATTERNS:
            if pattern.search(path_component):
                raise ResourceException(f"Dangerous pattern in path component: {path_component}")

    joined_path = os.path.join(base_dir, *paths)
    return validate_safe_path(joined_path, base_dir)


def safe_open(
    file_path: Union[str, Path],
    base_dir: Union[str, Path],
    mode: str = "r",
    **kwargs: Any,
) -> IO[Any]:
    """
    Safely open a file with path validation.

    Args:
        file_path: The file path to open
        base_dir: The base directory that file_path must be within
        mode: File open mode
        **kwargs: Additional arguments for open()

    Returns:
        File object

    Raises:
        ResourceException: If path is unsafe or file cannot be opened
    """
    safe_path = validate_safe_path(file_path, base_dir)

    # Additional validation for write modes
    if "w" in mode or "a" in mode or "+" in mode:
        # Ensure parent directory exists for write operations
        parent_dir = Path(safe_path).parent
        if not parent_dir.exists():
            raise ResourceException(f"Parent directory does not exist: {parent_dir}")

        # Check write permissions
        if not os.access(parent_dir, os.W_OK):
            raise ResourceException(f"No write permission for directory: {parent_dir}")

    try:
        return open(safe_path, mode, **kwargs)
    except OSError as e:
        raise ResourceException(f"Cannot open file {safe_path}: {e}")


def validate_filename(filename: str) -> str:
    """
    Validate filename for safety.

    Args:
        filename: The filename to validate

    Returns:
        str: The validated filename

    Raises:
        ResourceException: If filename is unsafe
    """
    if not filename or not filename.strip():
        raise ResourceException("Empty filename")

    filename = filename.strip()

    # Check for dangerous patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(filename):
            raise ResourceException(f"Dangerous pattern in filename: {filename}")

    # Check for reserved names (Windows)
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    if filename.upper().split(".")[0] in reserved_names:
        raise ResourceException(f"Reserved filename: {filename}")

    # Check for invalid characters (excluding valid ones like hyphens)
    invalid_chars = '<>:"|?*'
    # Add control characters (0x00-0x1f)
    invalid_chars += "".join(chr(i) for i in range(32))
    if any(char in filename for char in invalid_chars):
        raise ResourceException(f"Invalid characters in filename: {filename}")

    # Check length
    if len(filename) > 255:
        raise ResourceException("Filename too long")

    return filename
