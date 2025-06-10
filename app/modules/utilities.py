# =============================================================================
# File: utilities.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================


class Utilities:
    @staticmethod
    def add_missing_from_other(target: dict, source: dict) -> dict:
        """
        Adds only missing key-value pairs from source to target dict.
        Existing keys in target are not overwritten.
        Returns the updated target dict.
        """
        for key, value in source.items():
            if key not in target:
                target[key] = value
        return target
