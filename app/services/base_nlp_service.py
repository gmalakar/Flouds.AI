import os
from typing import Any, Optional
import onnxruntime as ort
from transformers import AutoTokenizer
from app.modules.concurrent_dict import ConcurrentDict
from app.config.config_loader import ConfigLoader
from app.logger import get_logger
from app.setup import APP_SETTINGS

logger = get_logger("base_nlp_service")

class BaseNLPService:
    _root_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "onnx"))
    _tokenizers: ConcurrentDict = ConcurrentDict()
    _encoder_sessions: ConcurrentDict = ConcurrentDict()

    @staticmethod
    def _get_model_config(model_to_use: str) -> Any:
        logger.debug(f"Loading model config for: {model_to_use}")
        return ConfigLoader.get_onnx_config(model_to_use)

    @staticmethod
    def _get_tokenizer(tokenizer_path: str) -> Any:
        logger.debug(f"Getting tokenizer for path: {tokenizer_path}")
        return BaseNLPService._tokenizers.get_or_add(
            tokenizer_path, lambda: AutoTokenizer.from_pretrained(tokenizer_path)
        )

    @staticmethod
    def _get_encoder_session(encoder_model_path: str) -> ort.InferenceSession:
        provider = APP_SETTINGS.server.model_session_provider or "CPUExecutionProvider"
        logger.debug(
            f"Getting ONNX encoder session for path: {encoder_model_path} with provider: {provider}"
        )
        providers = [provider]
        return BaseNLPService._encoder_sessions.get_or_add(
            (encoder_model_path, provider),
            lambda: ort.InferenceSession(encoder_model_path, providers=providers),
        )

    @staticmethod
    def _preprocess_text(text: str, prepend_text: Optional[str] = None) -> str:
        return f"{prepend_text}{text}" if prepend_text else text

    @staticmethod
    def _log_onnx_outputs(outputs: Any, session: Optional[Any]) -> None:
        if APP_SETTINGS.app.debug:
            if session is not None:
                output_names = [o.name for o in session.get_outputs()]
            else:
                output_names = [f"output_{i}" for i in range(len(outputs))]
            for name, arr in zip(output_names, outputs):
                logger.debug(
                    f"ONNX output: {name}, shape: {arr.shape}, dtype: {arr.dtype}"
                )

    @staticmethod
    def _is_logits_output(outputs: Any, session: Optional[Any] = None) -> bool:
        if session is not None:
            output_names = [o.name.lower() for o in session.get_outputs()]
            for name in output_names:
                if "logit" in name or "score" in name or "prob" in name:
                    return True
        arr = outputs[0]
        shape = arr.shape
        if len(shape) == 2:
            if shape[1] <= 10:
                return True
        elif len(shape) == 3:
            if shape[2] <= 10:
                return True
        return False