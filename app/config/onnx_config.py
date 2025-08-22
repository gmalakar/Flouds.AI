# =============================================================================
# File: onnx_config.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import Optional

from pydantic import BaseModel, Field


class InputNames(BaseModel):
    input: str = Field(default="input_ids")
    mask: str = Field(default="attention_mask")
    position: Optional[str] = None
    tokentype: Optional[str] = None
    use_decoder_input: bool = Field(default=False)
    decoder_input_name: str = Field(default="decoder_input_ids")


class OutputNames(BaseModel):
    output: str = Field(default="last_hidden_state")
    outputnext: Optional[str] = None
    outmask: Optional[str] = None


class DecoderInputNames(BaseModel):
    input: str = Field(default="input_ids")
    mask: str = Field(default="encoder_attention_mask")
    encoder_output: str = Field(default="encoder_hidden_states")


class OnnxConfig(BaseModel):
    dimension: int = 128
    inputnames: InputNames = Field(default_factory=InputNames)
    max_length: int = 256
    min_length: int = 0
    normalize: bool = True
    logits: bool = Field(default=False)
    eos_token_id: int = Field(default=1)
    summarization_task: str = Field(default="s2s")
    embedder_task: str = Field(default="fe")
    outputnames: OutputNames = Field(default_factory=OutputNames)
    decoder_inputnames: DecoderInputNames = Field(default_factory=DecoderInputNames)
    use_generation_config: bool = Field(default=False)
    pad_token_id: int = 0
    pooling_strategy: str = Field(default="mean")
    projected_dimension: int = Field(default=256)
    encoder_onnx_model: str = Field(default="encoder_model.onnx")
    decoder_onnx_model: str = Field(default="decoder_model.onnx")
    encoder_optimized_onnx_model: str = Field(default="model_optimized.onnx")
    decoder_optimized_onnx_model: str = Field(default="decoder_model_optimized.onnx")
    use_optimized: bool = Field(default=False)
    special_tokens_map_path: str = Field(default="special_tokens_map.json")
    generation_config_path: str = Field(default="generation_config.json")
    num_beams: int = 4
    temperature: float = 0.0
    early_stopping: bool = True
    use_seq2seqlm: bool = Field(default=False)
    prepend_text: str = Field(default="summarize: ")
    chunk_logic: str = Field(default="sentence")
    chunk_overlap: int = Field(default=1)
    chunk_size: int = Field(default=None)  # For fixed chunking
    legacy_tokenizer: bool = Field(
        default=False
    )  # Use legacy tokenizer for older models
    lowercase: bool = Field(default=False)  # Convert text to lowercase
    remove_emojis: bool = Field(default=False)  # Remove emojis and non-ASCII characters
    force_pooling: bool = Field(default=False)  # Force pooling for embeddings
