# =============================================================================
# File: model_info_response.py
# Date: 2025-12-30
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.base_response import BaseResponse


class ModelDetails(BaseModel):
    """
    Detailed model information nested under the 'details' field.
    """

    model_name: str = Field(
        None,
        description="Name of the model as specified in the configuration file.",
    )

    model_available: bool = Field(
        False, description="Whether the model is available in the configuration."
    )
    onnx_file_available: bool = Field(
        False, description="Whether the ONNX model file exists on disk."
    )
    model_type: Optional[str] = Field(
        None,
        description="Type of model: 'embedding' (fe), 'summarization' (s2s), or 'language_model' (llm).",
    )

    # Auto-detected parameters
    auto_detected_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters auto-detected from the ONNX model (dimension, inputnames, outputnames, vocab_size).",
    )

    # Configuration parameters
    config_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters from the configuration file, filtered by model type relevance.",
    )

    # Default values
    default_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default values for this model type that will be used if not specified in config.",
    )

    # Required parameters
    required_params: List[str] = Field(
        default_factory=list,
        description="List of required parameter names that must be specified.",
    )

    # File information
    files_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about model files (encoder, decoder, optimized versions).",
    )


class ModelInfoResponse(BaseResponse):
    """
    Response model for model information and availability check.
    All model details are nested under the 'details' field.
    """

    details: Optional[ModelDetails] = Field(
        default=None,
        description="Detailed information about the model.",
    )

    class ResultsModel(BaseModel):
        property_name: str = Field(
            ..., description="Requested property name (dot notation allowed)"
        )
        property_value: Optional[Any] = Field(
            None, description="Value of the requested property (nullable)"
        )

    results: Optional[ResultsModel] = Field(
        default=None,
        description="If `property_name` query param was provided, contains the requested property's name and value.",
    )
