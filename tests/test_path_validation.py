# =============================================================================
# File: test_path_validation.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Tests for enhanced path validation system."""

import os
import tempfile
from pathlib import Path

import pytest

from app.exceptions import ResourceException
from app.utils.path_validator import (
    safe_join,
    safe_open,
    validate_filename,
    validate_safe_path,
)


class TestPathValidation:
    """Test path validation functions."""

    def test_validate_safe_path_valid(self):
        """Test valid path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            Path(test_file).touch()

            result = validate_safe_path(test_file, temp_dir)
            assert os.path.abspath(test_file) == result

    def test_validate_safe_path_traversal_attack(self):
        """Test path traversal attack prevention."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test various path traversal attempts
            dangerous_paths = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32",
                temp_dir + "/../../../etc/passwd",
                os.path.join(temp_dir, "..", "..", "etc", "passwd"),
            ]

            for dangerous_path in dangerous_paths:
                with pytest.raises(ResourceException):
                    validate_safe_path(dangerous_path, temp_dir)

    def test_validate_safe_path_dangerous_patterns(self):
        """Test dangerous pattern detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dangerous_patterns = [
                "~/malicious",
                "$HOME/malicious",
                "%USERPROFILE%/malicious",
                "\\\\server\\share\\malicious",
            ]

            for pattern in dangerous_patterns:
                with pytest.raises(ResourceException):
                    validate_safe_path(pattern, temp_dir)

    def test_validate_safe_path_nonexistent_base(self):
        """Test validation with non-existent base directory."""
        with pytest.raises(ResourceException):
            validate_safe_path("test.txt", "/nonexistent/directory")

    def test_safe_join_valid(self):
        """Test safe path joining."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = safe_join(temp_dir, "subdir", "file.txt")
            expected = os.path.join(temp_dir, "subdir", "file.txt")
            assert os.path.abspath(expected) == result

    def test_safe_join_dangerous_components(self):
        """Test safe join with dangerous path components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dangerous_components = [
                ["../", "malicious"],
                ["subdir", "..\\..\\malicious"],
                ["~", "malicious"],
                ["$HOME", "malicious"],
            ]

            for components in dangerous_components:
                with pytest.raises(ResourceException):
                    safe_join(temp_dir, *components)

    def test_safe_open_read(self):
        """Test safe file opening for reading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            with safe_open(test_file, temp_dir, "r") as f:
                content = f.read()
                assert content == "test content"

    def test_safe_open_write(self):
        """Test safe file opening for writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")

            with safe_open(test_file, temp_dir, "w") as f:
                f.write("test content")

            with open(test_file, "r") as f:
                assert f.read() == "test content"

    def test_safe_open_traversal_attack(self):
        """Test safe open prevents path traversal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ResourceException):
                safe_open("../../../etc/passwd", temp_dir, "r")

    def test_validate_filename_valid(self):
        """Test valid filename validation."""
        valid_names = [
            "test.txt",
            "document.pdf",
            "image_001.jpg",
            "data-file.json",
        ]

        for name in valid_names:
            result = validate_filename(name)
            assert result == name

    def test_validate_filename_dangerous(self):
        """Test dangerous filename detection."""
        dangerous_names = [
            "../malicious.txt",
            "file<script>.txt",
            "CON.txt",  # Windows reserved name
            "file|pipe.txt",
            "file?.txt",
            "file*.txt",
            "",  # Empty filename
            "   ",  # Whitespace only
            "a" * 300,  # Too long
        ]

        for name in dangerous_names:
            with pytest.raises(ResourceException):
                validate_filename(name)

    def test_validate_filename_reserved_windows(self):
        """Test Windows reserved filename detection."""
        reserved_names = [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "LPT1",
            "LPT2",
            "con.txt",
            "prn.log",  # Case insensitive
        ]

        for name in reserved_names:
            with pytest.raises(ResourceException):
                validate_filename(name)


class TestPathValidationIntegration:
    """Integration tests for path validation."""

    def test_config_loading_security(self):
        """Test that config loading uses safe path validation."""
        # This would be tested with actual config loading
        # but requires proper setup of config files
        pass

    def test_model_loading_security(self):
        """Test that model loading uses safe path validation."""
        # This would be tested with actual model loading
        # but requires proper setup of model files
        pass


if __name__ == "__main__":
    pytest.main([__file__])
