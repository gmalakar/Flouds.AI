# =============================================================================
# File: test_config_caching.py
# Date: 2025-01-27
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

"""Tests for configuration caching performance improvements."""

import time
from unittest.mock import Mock, patch

import pytest

from app.config.config_loader import ConfigLoader
from app.services.base_nlp_service import BaseNLPService
from app.utils.cache_manager import CacheManager


class TestConfigCaching:
    """Test configuration caching functionality."""

    def test_onnx_config_caching(self):
        """Test that ONNX configs are cached properly."""
        with (
            patch.object(ConfigLoader, "_load_config_data") as mock_load,
            patch("os.path.getmtime") as mock_mtime,
        ):

            mock_load.return_value = {
                "test-model": {
                    "dimension": 512,
                    "inputnames": {"input": "input_ids"},
                    "outputnames": {"output": "last_hidden_state"},
                }
            }
            mock_mtime.return_value = time.time()

            # Clear cache first
            ConfigLoader.clear_cache()

            # First call should load from file
            config1 = ConfigLoader.get_onnx_config("test-model")
            assert mock_load.call_count == 1

            # Second call should use cache
            config2 = ConfigLoader.get_onnx_config("test-model")
            assert mock_load.call_count == 1  # No additional calls

            # Configs should be the same object
            assert config1 is config2

    def test_cache_invalidation_on_file_change(self):
        """Test that cache is invalidated when config file changes."""
        with (
            patch.object(ConfigLoader, "_load_config_data") as mock_load,
            patch("os.path.getmtime") as mock_mtime,
        ):

            mock_load.return_value = {
                "test-model": {
                    "dimension": 512,
                    "inputnames": {"input": "input_ids"},
                    "outputnames": {"output": "last_hidden_state"},
                }
            }

            # Clear cache first
            ConfigLoader.clear_cache()

            # First call with mtime = 1000
            mock_mtime.return_value = 1000
            ConfigLoader.get_onnx_config("test-model")
            assert mock_load.call_count == 1

            # Second call with same mtime should use cache
            ConfigLoader.get_onnx_config("test-model")
            assert mock_load.call_count == 1

            # Third call with different mtime should reload
            mock_mtime.return_value = 2000
            ConfigLoader.get_onnx_config("test-model")
            assert mock_load.call_count == 2

    def test_model_config_caching_in_base_service(self):
        """Test model configuration caching in BaseNLPService."""
        with patch.object(ConfigLoader, "get_onnx_config") as mock_get_config:
            mock_config = Mock()
            mock_config.inputnames = Mock()
            mock_config.outputnames = Mock()
            mock_get_config.return_value = mock_config

            # Clear cache first
            BaseNLPService._clear_model_config_cache()

            # First call should load from ConfigLoader
            config1 = BaseNLPService._get_model_config("test-model")
            assert mock_get_config.call_count == 1

            # Second call should use cache
            config2 = BaseNLPService._get_model_config("test-model")
            assert mock_get_config.call_count == 1  # No additional calls

            # Configs should be the same object
            assert config1 is config2

    def test_cache_stats(self):
        """Test cache statistics reporting."""
        stats = ConfigLoader.get_cache_stats()

        assert "onnx_configs_cached" in stats
        assert "cache_file_mtime" in stats
        assert "cache_loaded" in stats
        assert isinstance(stats["onnx_configs_cached"], int)

    def test_cache_manager_functionality(self):
        """Test CacheManager operations."""
        # Test getting cache health
        health = CacheManager.get_cache_health()
        assert "status" in health
        assert "total_cached_items" in health
        assert "cache_efficiency" in health
        assert "details" in health

        # Test getting all cache stats
        all_stats = CacheManager.get_all_cache_stats()
        assert "config_loader" in all_stats
        assert "base_nlp_service" in all_stats
        assert "timestamp" in all_stats

    def test_cache_warm_up(self):
        """Test cache warm-up functionality."""
        with patch.object(ConfigLoader, "get_onnx_config") as mock_get_config:
            mock_get_config.return_value = Mock()

            # Test warm-up with specific models
            test_models = ["model1", "model2"]
            CacheManager.warm_up_caches(test_models)

            assert mock_get_config.call_count == len(test_models)
            for model in test_models:
                mock_get_config.assert_any_call(model)

    def test_cache_clear_all(self):
        """Test clearing all caches."""
        with (
            patch.object(ConfigLoader, "clear_cache") as mock_clear_config,
            patch("app.utils.cache_manager.clear_encoder_sessions") as mock_clear_sessions,
            patch("app.utils.cache_manager.clear_thread_tokenizers") as mock_clear_tokenizers,
            patch("app.utils.cache_manager.clear_model_config_cache") as mock_clear_model_config,
        ):

            CacheManager.clear_all_caches()

            mock_clear_config.assert_called_once()
            mock_clear_sessions.assert_called_once()
            mock_clear_tokenizers.assert_called_once()
            mock_clear_model_config.assert_called_once()


class TestCachePerformance:
    """Test cache performance improvements."""

    def test_config_loading_performance(self):
        """Test that caching improves config loading performance."""
        with (
            patch.object(ConfigLoader, "_load_config_data") as mock_load,
            patch("os.path.getmtime") as mock_mtime,
        ):

            # Simulate slow file loading
            def slow_load(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                return {
                    "test-model": {
                        "dimension": 512,
                        "inputnames": {"input": "input_ids"},
                        "outputnames": {"output": "last_hidden_state"},
                    }
                }

            mock_load.side_effect = slow_load
            mock_mtime.return_value = time.time()

            # Clear cache first
            ConfigLoader.clear_cache()

            # First call (should be slow)
            start_time = time.time()
            ConfigLoader.get_onnx_config("test-model")
            first_call_time = time.time() - start_time

            # Second call (should be fast due to caching)
            start_time = time.time()
            ConfigLoader.get_onnx_config("test-model")
            second_call_time = time.time() - start_time

            # Second call should be significantly faster
            assert second_call_time < first_call_time / 2
            assert mock_load.call_count == 1  # Only called once


if __name__ == "__main__":
    pytest.main([__file__])
