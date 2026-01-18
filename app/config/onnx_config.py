# =============================================================================
# File: onnx_config.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

from typing import List, Optional

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
    eos_token_id: int = Field(default=1)
    bos_token_id: Optional[int] = Field(default=None)
    # Legacy fields `summarization_task` and `embedder_task` removed.
    # Use `tasks: List[str]` to declare supported capabilities (e.g. ['embedding','prompt']).
    # Optional explicit folder name for the model under the `models/` directory.
    # When set, this value should be used as the relative folder name to locate
    # the model files instead of deriving the folder from task/type.
    model_folder_name: Optional[str] = Field(default=None)
    # Explicit list of supported tasks for this model. Examples: ["embedding", "prompt", "llm", "summarization"]
    tasks: List[str] = Field(default_factory=list)
    outputnames: OutputNames = Field(default_factory=OutputNames)
    decoder_inputnames: DecoderInputNames = Field(default_factory=DecoderInputNames)
    pad_token_id: int = 0
    pooling_strategy: str = Field(default="mean")
    encoder_onnx_model: str = Field(default="encoder_model.onnx")
    decoder_onnx_model: str = Field(default="decoder_model.onnx")
    # Note: Existence flags are stored in the runtime model metadata cache
    # (see `BaseNLPService._set_model_metadata`) rather than on the
    # configuration object itself.
    special_tokens_map_path: str = Field(default="special_tokens_map.json")
    generation_config_path: str = Field(default="generation_config.json")
    num_beams: int = 4
    temperature: float = 0.0
    top_k: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    repetition_penalty: Optional[float] = Field(default=None)
    early_stopping: bool = True
    use_seq2seqlm: bool = Field(default=False)
    encoder_only: bool = Field(default=False)  # For GPT-style models that use only encoder
    prepend_text: str = Field(default="summarize: ")
    chunk_logic: str = Field(default="sentence")
    chunk_overlap: int = Field(default=1)
    chunk_size: Optional[int] = Field(default=None)  # For fixed chunking
    legacy_tokenizer: bool = Field(default=False)  # Use legacy tokenizer for older models
    lowercase: bool = Field(default=False)  # Convert text to lowercase
    remove_emojis: bool = Field(default=False)  # Remove emojis and non-ASCII characters
    force_pooling: bool = Field(default=False)  # Force pooling for embeddings
    vocab_size: Optional[int] = Field(default=None)
    quantize: bool = Field(default=False)  # Enable output quantization
    quantize_type: str = Field(default="int8")  # Quantization type: int8, uint8, binary
    forced_bos_token_id: Optional[int] = Field(default=None)  # Forced BOS token ID for generation
    # Preferred limit on newly generated tokens (recommended over total `max_length`)
    max_new_tokens: Optional[int] = Field(default=None)
