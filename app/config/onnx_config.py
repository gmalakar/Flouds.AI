from typing import Optional

from pydantic import BaseModel, Field


class InputNames(BaseModel):
    input: str = Field(default="input_ids")
    mask: str = Field(default="attention_mask")
    eos_token_id: int = Field(default=1)
    position: Optional[str] = None
    tokentype: Optional[str] = None
    use_decoder_input: bool = Field(default=False)
    decoder_input_name: str = Field(default="decoder_input_ids")


class OutputNames(BaseModel):
    output: str = Field(default="last_hidden_state")
    outputnext: Optional[str] = None
    outmask: Optional[str] = None
    logits: bool = Field(default=False)


class DecoderInputNames(BaseModel):
    input: str = Field(default="input_ids")
    mask: str = Field(default="encoder_attention_mask")
    input_ids: int = Field(default=0)
    encoder_output: str = Field(default="encoder_hidden_states")


class OnnxConfig(BaseModel):
    dimension: int = 128
    inputnames: InputNames = Field(default_factory=InputNames)
    max_length: int = 256
    normalize: bool = True
    summarization_task: str = Field(default="seq2seq-lm")
    embedder_task: str = Field(default="feature-extraction")
    outputnames: OutputNames = Field(default_factory=OutputNames)
    decoder_inputnames: DecoderInputNames = Field(default_factory=DecoderInputNames)
    use_generation_config: bool = Field(default=False)
    padid: int = 0
    pooling_strategy: str = Field(default="mean")
    projected_dimension: int = Field(default=256)
    encoder_onnx_model: str = Field(default="encoder_model.onnx")
    decoder_onnx_model: str = Field(default="decoder_model.onnx")
    special_tokens_map_path: str = Field(default="special_tokens_map.json")
    num_beams: int = 0
    early_stopping: bool = True
    use_seq2seqlm: bool = Field(default=False)
    prepend_text: str = Field(default="summarize: ")
