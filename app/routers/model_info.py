# =============================================================================
# File: model_info.py
# Date: 2025-12-30
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.logger import get_logger
from app.models.model_info_response import ModelInfoResponse
from app.services.base_nlp_service import BaseNLPService
from app.utils.path_validator import validate_safe_path

logger = get_logger("model_info")

router = APIRouter()


class ModelInfoService:
    """Service for retrieving model information and availability."""

    @staticmethod
    def _get_model_type(config: Any) -> str:
        """Determine model type from configuration."""
        if hasattr(config, "embedder_task"):
            return "embedding"
        elif hasattr(config, "summarization_task"):
            task = getattr(config, "summarization_task", "")
            if task == "llm":
                return "language_model"
            elif task == "s2s":
                return "summarization"
        return "unknown"

    @staticmethod
    def _get_required_params(model_type: str) -> List[str]:
        """Get list of required parameters based on model type."""
        common_required = ["max_length", "chunk_logic", "encoder_onnx_model"]

        if model_type == "embedding":
            return common_required + ["embedder_task", "normalize", "pooling_strategy"]
        elif model_type in ["summarization", "language_model"]:
            return common_required + [
                "summarization_task",
                "pad_token_id",
                "eos_token_id",
            ]
        return common_required

    @staticmethod
    def _get_default_params(model_type: str) -> Dict[str, Any]:
        """Get default parameter values for specific model type."""
        # Common defaults for all models
        common_defaults = {
            "legacy_tokenizer": False,
            "lowercase": False,
            "remove_emojis": False,
            "chunk_overlap": 0,
            "use_optimized": False,
        }

        # Embedding-specific defaults
        if model_type == "embedding":
            return {**common_defaults, "force_pooling": False}

        # Summarization and language model defaults
        elif model_type in ["summarization", "language_model"]:
            return {
                **common_defaults,
                "use_seq2seqlm": False,
                "min_length": 0,
                "num_beams": 1,
                "early_stopping": False,
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            }

        return common_defaults

    @staticmethod
    def _filter_config_params(
        config_params: Dict[str, Any], model_type: str
    ) -> Dict[str, Any]:
        """Filter configuration parameters to only include relevant ones for the model type."""
        # Parameters relevant to all models
        common_params = [
            "max_length",
            "chunk_logic",
            "chunk_overlap",
            "chunk_size",
            "encoder_onnx_model",
            "encoder_optimized_onnx_model",
            "legacy_tokenizer",
            "use_optimized",
            "lowercase",
            "remove_emojis",
        ]

        # Embedding-specific parameters
        embedding_params = [
            "embedder_task",
            "normalize",
            "pooling_strategy",
            "force_pooling",
            "inputnames",
            "outputnames",
            "dimension",
        ]

        # Summarization/Language model specific parameters
        generation_params = [
            "summarization_task",
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "decoder_start_token_id",
            "decoder_onnx_model",
            "decoder_optimized_onnx_model",
            "decoder_inputnames",
            "use_seq2seqlm",
            "vocab_size",
            "special_tokens_map_path",
            "generation_config_path",
            "min_length",
            "num_beams",
            "early_stopping",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "prepend_text",
            "encoder_only",
            "use_t5_encoder",
        ]

        # Determine which parameters to include
        if model_type == "embedding":
            allowed_params = common_params + embedding_params
        elif model_type in ["summarization", "language_model"]:
            allowed_params = common_params + generation_params
        else:
            allowed_params = common_params

        # Filter the config params
        return {k: v for k, v in config_params.items() if k in allowed_params}

    @staticmethod
    def _check_onnx_files(model_path: str, config: Any) -> Dict[str, Any]:
        """Check which ONNX model files exist."""
        files_info = {
            "encoder_file": None,
            "encoder_exists": False,
            "encoder_optimized_file": None,
            "encoder_optimized_exists": False,
            "decoder_file": None,
            "decoder_exists": False,
            "decoder_optimized_file": None,
            "decoder_optimized_exists": False,
        }

        # Check encoder file
        encoder_file = getattr(config, "encoder_onnx_model", "model.onnx")
        encoder_path = os.path.join(model_path, encoder_file)
        files_info["encoder_file"] = encoder_file
        files_info["encoder_exists"] = os.path.exists(encoder_path)

        # Check encoder optimized file
        if hasattr(config, "encoder_optimized_onnx_model"):
            encoder_opt_file = config.encoder_optimized_onnx_model
            encoder_opt_path = os.path.join(model_path, encoder_opt_file)
            files_info["encoder_optimized_file"] = encoder_opt_file
            files_info["encoder_optimized_exists"] = os.path.exists(encoder_opt_path)

        # Check decoder file
        if hasattr(config, "decoder_onnx_model"):
            decoder_file = config.decoder_onnx_model
            decoder_path = os.path.join(model_path, decoder_file)
            files_info["decoder_file"] = decoder_file
            files_info["decoder_exists"] = os.path.exists(decoder_path)

        # Check decoder optimized file
        if hasattr(config, "decoder_optimized_onnx_model"):
            decoder_opt_file = config.decoder_optimized_onnx_model
            decoder_opt_path = os.path.join(model_path, decoder_opt_file)
            files_info["decoder_optimized_file"] = decoder_opt_file
            files_info["decoder_optimized_exists"] = os.path.exists(decoder_opt_path)

        return files_info

    @staticmethod
    def _get_auto_detected_params(model_path: str, config: Any) -> Dict[str, Any]:
        """Get auto-detected parameters from ONNX model."""
        auto_detected = {}

        try:
            # Get encoder session to detect parameters
            encoder_file = getattr(config, "encoder_onnx_model", "model.onnx")
            encoder_path = os.path.join(model_path, encoder_file)

            if not os.path.exists(encoder_path):
                return auto_detected

            # Load session
            from app.services.embedder_service import SentenceTransformer

            session = SentenceTransformer._get_encoder_session(encoder_path)

            if session:
                # Detect dimension
                native_dim = SentenceTransformer._get_native_dimension_from_session(
                    session
                )
                if native_dim:
                    auto_detected["dimension"] = native_dim

                # Detect output names
                output_names = SentenceTransformer._get_output_names_from_session(
                    session
                )
                if output_names:
                    auto_detected["outputnames"] = output_names
                    auto_detected["primary_output"] = (
                        output_names[0] if output_names else None
                    )

                # Detect input names
                input_names = [inp.name for inp in session.get_inputs()]
                if input_names:
                    auto_detected["inputnames"] = input_names

                # Detect vocab_size (for language models)
                from app.services.prompt_service import PromptProcessor

                vocab_size = PromptProcessor._get_vocab_size_from_session(session)
                if vocab_size:
                    auto_detected["vocab_size"] = vocab_size

        except Exception as e:
            logger.warning(f"Error auto-detecting parameters: {e}")

        return auto_detected

    @staticmethod
    def get_model_info(model_name: str) -> ModelInfoResponse:
        """Get comprehensive model information including availability and parameters."""
        from app.models.model_info_response import ModelDetails

        response = ModelInfoResponse(
            success=True,
            message="Model information retrieved successfully",
            model=model_name,
            details=ModelDetails(),
        )

        # Check if model exists in config
        config = BaseNLPService._get_model_config(model_name)
        if not config:
            response.success = False
            response.message = f"Model '{model_name}' not found in configuration"
            response.details.model_available = False
            return response

        response.details.model_available = True

        # Get model type
        model_type = ModelInfoService._get_model_type(config)
        response.details.model_type = model_type

        # Get model path (internal use only, not returned)
        task_folder = getattr(
            config, "embedder_task", "llm" if model_type == "language_model" else "s2s"
        )
        try:
            model_path = validate_safe_path(
                os.path.join(
                    BaseNLPService._root_path, "models", task_folder, model_name
                ),
                BaseNLPService._root_path,
            )
        except Exception as e:
            response.success = False
            response.message = f"Invalid model path: {e}"
            return response

        # Check if ONNX files exist
        files_info = ModelInfoService._check_onnx_files(model_path, config)
        response.details.files_info = files_info
        response.details.onnx_file_available = files_info["encoder_exists"]

        # Get configuration parameters (use Pydantic's model_dump for proper serialization)
        if hasattr(config, "model_dump"):
            # Pydantic v2
            config_params = config.model_dump(exclude_none=True, exclude_unset=True)
        elif hasattr(config, "dict"):
            # Pydantic v1
            config_params = config.dict(exclude_none=True, exclude_unset=True)
        else:
            # Fallback for non-Pydantic objects
            config_params = {
                k: v
                for k, v in vars(config).items()
                if not k.startswith("_") and v is not None
            }

        # Filter config params based on model type
        config_params = ModelInfoService._filter_config_params(
            config_params, model_type
        )
        response.details.config_params = config_params

        # Get auto-detected parameters
        if response.details.onnx_file_available:
            auto_detected = ModelInfoService._get_auto_detected_params(
                model_path, config
            )
            response.details.auto_detected_params = auto_detected

        # Get default parameters for this model type
        response.details.default_params = ModelInfoService._get_default_params(
            model_type
        )

        # Get required parameters
        response.details.required_params = ModelInfoService._get_required_params(
            model_type
        )

        # Update message based on availability
        if not response.details.onnx_file_available:
            response.warnings = [
                f"ONNX model file '{files_info['encoder_file']}' not found"
            ]
            response.message = "Model found in config but ONNX files are missing"

        return response


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model Information"])
async def get_model_info(
    model: str = Query(..., description="Name of the model to check")
) -> ModelInfoResponse:
    """
    Get comprehensive model information including availability and parameters.

    Returns:
    - Model availability in configuration
    - ONNX file existence
    - Auto-detected parameters (dimension, inputnames, outputnames, vocab_size)
    - Configuration parameters
    - Default parameter values
    - Required parameters list
    - File information (encoder, decoder, optimized versions)
    """
    try:
        return ModelInfoService.get_model_info(model)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/list", tags=["Model Information"])
@router.post("/models/list", tags=["Model Information"])
async def list_models():
    """
    List all available models in the configuration.

    Returns a list of model names and their types.
    """
    try:
        from app.config.config_loader import ConfigLoader

        # Load one config to ensure cache is populated
        config_file = ConfigLoader.get_app_settings().onnx.config_file

        # Force cache refresh by checking if it should be refreshed
        if ConfigLoader._should_refresh_cache(config_file):
            ConfigLoader._refresh_onnx_cache(config_file)

        # Access the cached configs directly
        all_configs = ConfigLoader._ConfigLoader__onnx_config_cache

        models_list = []
        for model_name, config in all_configs.items():
            if not model_name.startswith("_"):
                model_type = ModelInfoService._get_model_type(config)
                models_list.append({"name": model_name, "type": model_type})

        return {
            "success": True,
            "message": f"Found {len(models_list)} models",
            "models": models_list,
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
