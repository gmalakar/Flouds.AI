# =============================================================================
# File: model_info.py
# Date: 2025-12-30
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import os
from typing import Any, Dict, List, Optional, Union, cast

from fastapi import APIRouter, HTTPException, Query

from app.logger import get_logger
from app.models.model_info_response import ModelInfoResponse
from app.services.base_nlp_service import BaseNLPService

# validate_safe_path not required in this module

logger = get_logger("model_info")

router = APIRouter()

# Module-level Query defaults to avoid function-call defaults (flake8 B008)
MODEL_QUERY = Query(..., description="Name of the model to check")
PROPERTY_NAME_QUERY = Query(
    None,
    description="Optional property name from `details` to return (returns its value or null).",
)
FOR_TASK_QUERY = Query(None, description="Optional task filter (e.g. 'embedding', 'prompt')")


class ModelInfoService:
    """Service for retrieving model information and availability."""

    @staticmethod
    def _get_model_type(config: Any) -> List[str]:
        """Determine model tasks from configuration and return ALL declared tasks.

        Returns a list of lower-cased task names (e.g. ['embedding', 'prompt']).
        Only the explicit `tasks` list in the model configuration is considered.
        """
        tasks: List[str] = []

        # Prefer explicit `tasks` list in config.
        if hasattr(config, "tasks") and config.tasks:
            try:
                # Access attribute directly after hasattr check to satisfy bugbear
                tasks = [str(t).lower() for t in config.tasks]
            except Exception:
                tasks = []

        # Ensure unique, normalized ordering
        normalized = []
        for t in tasks:
            lt = str(t).lower()
            if lt not in normalized:
                normalized.append(lt)
        return normalized

    @staticmethod
    def get_model_type(config: Any) -> List[str]:
        """Public wrapper for `_get_model_type` to avoid protected-member warnings.

        Returns a list of tasks.
        """
        return ModelInfoService._get_model_type(config)

    @staticmethod
    def _get_required_params(model_type: Union[str, List[str]]) -> List[str]:
        """Get list of required parameters based on one or more model types.

        If a list is provided, returns the union of required params for all types.
        """
        common_required = {"max_length", "chunk_logic", "encoder_onnx_model"}

        types: List[str] = [model_type] if isinstance(model_type, str) else list(model_type or [])

        required = set(common_required)
        if "embedding" in types:
            required.update({"normalize", "pooling_strategy"})
        if any(t in types for t in ["summarization", "language_model"]):
            required.update({"pad_token_id", "eos_token_id"})

        return list(required)

    @staticmethod
    def _get_default_params(model_type: Union[str, List[str]]) -> Dict[str, Any]:
        """Get default parameter values for specific model type or combination of types.

        When multiple types are provided, defaults are merged with embedding-specific
        defaults taking effect when `embedding` is present, and generation defaults
        taking effect when `summarization` or `language_model` is present.
        """
        # Common defaults for all models
        common_defaults: Dict[str, Any] = {
            "legacy_tokenizer": False,
            "lowercase": False,
            "remove_emojis": False,
            "chunk_overlap": 0,
        }

        types: List[str] = [model_type] if isinstance(model_type, str) else list(model_type or [])

        defaults: Dict[str, Any] = dict(common_defaults)

        if "embedding" in types:
            defaults.update({"force_pooling": False})

        if any(t in types for t in ["summarization", "language_model"]):
            gen_defaults = {
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
            defaults.update(gen_defaults)

        return defaults

    @staticmethod
    def _filter_config_params(
        config_params: Dict[str, Any], model_type: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """Filter configuration parameters to only include relevant ones for the model type(s).

        When multiple types are provided, the allowed params are the union for all
        relevant types.
        """
        # Parameters relevant to all models
        common_params = {
            "max_length",
            "chunk_logic",
            "chunk_overlap",
            "chunk_size",
            "encoder_onnx_model",
            "legacy_tokenizer",
            "lowercase",
            "remove_emojis",
        }

        # Embedding-specific parameters
        embedding_params = {
            "normalize",
            "pooling_strategy",
            "force_pooling",
            "inputnames",
            "outputnames",
            "dimension",
        }

        # Summarization/Language model specific parameters
        generation_params = {
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "decoder_start_token_id",
            "decoder_onnx_model",
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
        }

        types: List[str] = [model_type] if isinstance(model_type, str) else list(model_type or [])

        allowed = set(common_params)
        if "embedding" in types:
            allowed.update(embedding_params)
        if any(t in types for t in ["summarization", "language_model"]):
            allowed.update(generation_params)

        # Filter the config params with explicit typing to help static checkers
        filtered: Dict[str, Any] = {}
        for k, v in config_params.items():
            if k in allowed:
                filtered[str(k)] = v
        return filtered

    @staticmethod
    def _check_onnx_files(model_path: str, config: Any) -> Dict[str, Any]:
        """Check which ONNX model files exist."""
        files_info: Dict[str, Any] = {
            "encoder_file": None,
            "encoder_exists": False,
            "decoder_file": None,
            "decoder_exists": False,
        }

        # Check encoder file
        encoder_file = getattr(config, "encoder_onnx_model", "model.onnx")
        encoder_path = os.path.join(model_path, encoder_file)
        files_info["encoder_file"] = encoder_file
        files_info["encoder_exists"] = os.path.exists(encoder_path)

        # Check decoder file
        if hasattr(config, "decoder_onnx_model"):
            decoder_file = config.decoder_onnx_model
            decoder_path = os.path.join(model_path, decoder_file)
            files_info["decoder_file"] = decoder_file
            files_info["decoder_exists"] = os.path.exists(decoder_path)

        # (Optimized artifact checks removed - project no longer uses optimized filenames)

        return files_info

    @staticmethod
    def _get_auto_detected_params(model_path: str, config: Any) -> Dict[str, Any]:
        """Get auto-detected parameters from ONNX model."""
        auto_detected: Dict[str, Any] = {}

        try:
            # Get encoder session to detect parameters
            encoder_file = getattr(config, "encoder_onnx_model", "model.onnx")
            encoder_path = os.path.join(model_path, encoder_file)

            if not os.path.exists(encoder_path):
                return auto_detected

            # Load session (use cast to Any for protected helpers)
            from typing import cast

            from app.services.embedder_service import SentenceTransformer

            session = cast(Any, SentenceTransformer)._get_encoder_session(encoder_path)

            if session:
                # Detect dimension
                native_dim = cast(Any, SentenceTransformer)._get_native_dimension_from_session(
                    session
                )
                if native_dim:
                    auto_detected["dimension"] = native_dim

                # Detect output names
                output_names = cast(Any, SentenceTransformer)._get_output_names_from_session(
                    session
                )
                if output_names:
                    auto_detected["outputnames"] = output_names
                    auto_detected["primary_output"] = output_names[0] if output_names else None

                # Detect input names
                input_names: List[str] = [getattr(inp, "name", "") for inp in session.get_inputs()]
                input_names = [n for n in input_names if n]
                if input_names:
                    auto_detected["inputnames"] = input_names

                # Detect vocab_size (for language models)
                from app.services.prompt_service import PromptProcessor

                vocab_size = cast(Any, PromptProcessor)._get_vocab_size_from_session(session)
                if vocab_size:
                    auto_detected["vocab_size"] = vocab_size

        except Exception as e:
            logger.warning(f"Error auto-detecting parameters: {e}")

        return auto_detected

    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get comprehensive model information including availability and parameters.

        Returns a plain dict to avoid static-checker issues with Pydantic constructor
        signatures. The router will return this dict as a `ModelInfoResponse`.
        """

        # Build initial dict-backed response and details
        response: Dict[str, Any] = {
            "success": True,
            "message": "Model information retrieved successfully",
            "model": model_name,
            "time_taken": 0.0,
            "warnings": [],
            "details": {
                "model_name": model_name,
                "model_available": False,
                "onnx_file_available": False,
                "model_type": None,
                "auto_detected_params": {},
                "config_params": {},
                "default_params": {},
                "required_params": [],
                "files_info": {},
            },
        }

        details: Dict[str, Any] = cast(Dict[str, Any], response["details"])

        # Check if model exists in config (call protected helper via Any cast to silence linter)
        config = cast(Any, BaseNLPService)._get_model_config(model_name)
        if not config:
            response["success"] = False
            response["message"] = f"Model '{model_name}' not found in configuration"
            details["model_available"] = False
            return response

        details["model_available"] = True

        # Get model type
        model_type = ModelInfoService._get_model_type(config)
        details["model_type"] = model_type

        # Resolve model path using BaseNLPService resolver which respects
        # `model_folder_name` and falls back to task-based folders.
        try:
            model_path = BaseNLPService._get_model_path(model_name)
            if not model_path:
                raise Exception("Could not resolve model path")
        except Exception as e:
            response["success"] = False
            response["message"] = f"Invalid model path: {e}"
            return response

        # Check if ONNX files exist
        files_info: Dict[str, Any] = ModelInfoService._check_onnx_files(model_path, config)
        details["files_info"] = files_info
        details["onnx_file_available"] = files_info["encoder_exists"]

        # Get configuration parameters (use Pydantic's model_dump for proper serialization)
        config_params: Dict[str, Any]
        if hasattr(config, "model_dump"):
            # Pydantic v2
            config_params = config.model_dump(exclude_none=True, exclude_unset=True)
        elif hasattr(config, "dict"):
            # Pydantic v1
            config_params = config.dict(exclude_none=True, exclude_unset=True)
        else:
            # Fallback for non-Pydantic objects - build typed dict explicitly
            config_params = {}
            raw_vars: Dict[str, Any] = cast(Dict[str, Any], vars(config))
            for k_raw, v in raw_vars.items():
                # Make key explicitly a string and value typed for the static checker
                k: str = str(k_raw)
                if not k.startswith("_") and v is not None:
                    config_params[k] = v

        # Filter config params based on model type
        config_params = ModelInfoService._filter_config_params(config_params, model_type)
        details["config_params"] = config_params

        # Get auto-detected parameters (call protected helpers via Any cast)
        if details.get("onnx_file_available"):
            auto_detected: Dict[str, Any] = ModelInfoService._get_auto_detected_params(
                model_path, config
            )
            details["auto_detected_params"] = auto_detected

        # Get default parameters for this model type
        details["default_params"] = ModelInfoService._get_default_params(model_type)

        # Get required parameters
        details["required_params"] = ModelInfoService._get_required_params(model_type)

        # Update message based on availability
        if not details.get("onnx_file_available"):
            response["warnings"] = [f"ONNX model file '{files_info['encoder_file']}' not found"]
            response["message"] = "Model found in config but ONNX files are missing"

        return response


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    response_model_exclude_none=True,
    tags=["Model Information"],
)
async def get_model_info(
    model: str = MODEL_QUERY, property_name: Optional[str] = PROPERTY_NAME_QUERY
) -> Any:
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
        response = ModelInfoService.get_model_info(model)

        # If caller requested a specific property, return only that property's value
        if property_name:
            # `response` is a dict produced by the service; get `details` as a dict
            details_dict: Dict[str, Any] = response.get("details", {}) or {}

            # Support nested lookups using dot notation, e.g. 'auto_detected_params.dimension'
            parts = property_name.split(".") if property_name else []

            # Traverse details_dict to get the deepest value, return None if any key missing
            cur: Any = details_dict
            value: Any = None
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    cur = None
                    break
            value = cur

            # Normalize empty containers to None so callers receive a null for missing values
            if isinstance(value, (dict, list)) and len(value) == 0:
                value = None

            # Build nested payload (not used directly) and prepare explicit JSON response
            from typing import Any as _Any
            from typing import List as _List

            def build_nested(keys: _List[str], val: _Any) -> _Any:
                if not keys:
                    return val
                return {keys[0]: build_nested(keys[1:], val)}

            # Prepare explicit JSON response so we can include JSON `null` for missing values
            from fastapi.responses import JSONResponse

            prop_value: Any = value if value is not None else None

            # Update message to reflect property retrieval or absence
            if value is not None:
                msg = f"Property '{property_name}' retrieved"
                success_flag = True
            else:
                msg = f"Property '{property_name}' not found"
                success_flag = False

            json_payload = {
                "success": success_flag,
                "message": msg,
                "model": response.get("model"),
                "time_taken": response.get("time_taken", 0.0),
                "warnings": response.get("warnings", []),
                "results": {
                    "property_name": property_name,
                    "property_value": prop_value,
                },
            }

            return JSONResponse(content=json_payload)

        return response
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/list", tags=["Model Information"])
@router.post("/models/list", tags=["Model Information"])
async def list_models(for_task: Optional[str] = FOR_TASK_QUERY) -> Dict[str, Any]:
    """
    List all available models in the configuration.

    Returns a list of model names and their types.
    """
    try:
        from app.config.config_loader import ConfigLoader

        # Load one config to ensure cache is populated
        cfg = ConfigLoader.get_app_settings()
        config_file = getattr(cfg, "onnx", None)
        config_file = getattr(config_file, "config_file", None) if config_file else None

        # Force cache refresh by checking if it should be refreshed (use Any cast for protected helpers)
        if config_file and cast(Any, ConfigLoader)._should_refresh_cache(config_file):
            cast(Any, ConfigLoader)._refresh_onnx_cache(config_file)

        # Access the cached configs directly (use getattr to avoid name-mangled attribute access)
        all_configs: Dict[str, Any] = getattr(
            cast(Any, ConfigLoader), "_ConfigLoader__onnx_config_cache", {}
        )

        models_list: List[Dict[str, str]] = []
        filter_task = str(for_task).lower() if for_task else None
        for mn, cfg_obj in all_configs.items():
            # Normalize model name to str for safe string ops
            model_name = str(mn)
            if model_name.startswith("_"):
                continue
            model_type = cast(Any, ModelInfoService)._get_model_type(cfg_obj)
            # Normalize model_type into a list of lower-cased tokens (handle
            # lists, comma-separated strings, and other unexpected shapes).
            try:
                if isinstance(model_type, (list, tuple)):
                    type_list = [str(t).lower() for t in model_type]
                elif isinstance(model_type, str):
                    type_list = [p.strip().lower() for p in model_type.split(",") if p.strip()]
                else:
                    type_list = []
            except Exception:
                type_list = []

            # If a task filter was provided, only include models that declare it.
            # For backward compatibility, also check legacy fields like
            # `embedder_task` / `summarization_task` when `tasks` is absent.
            if filter_task:
                # Only include models that explicitly declare the requested task
                if filter_task not in type_list:
                    continue
            # Format the type for the short listing: join multiple tasks if present
            try:
                if isinstance(model_type, (list, tuple)):
                    type_str = ",".join([str(t) for t in model_type])
                else:
                    type_str = str(model_type)
            except Exception:
                type_str = str(model_type)
            models_list.append({"name": model_name, "type": type_str})

        return {
            "success": True,
            "message": f"Found {len(models_list)} models",
            "models": models_list,
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
