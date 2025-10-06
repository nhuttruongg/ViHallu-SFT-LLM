"""
Enhanced configuration for VinAllama-7B Hallucination Detection
Optimized for 3-class classification: NO, INTRINSIC, EXTRINSIC
"""
from dataclasses import dataclass, field
import os
from typing import Optional


def _getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


def _getenv_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


@dataclass
class Config:
    model_id: str = os.getenv("MODEL_ID", "vilm/vinallama-7b")
    tokenizer_id: str = os.getenv("TOKENIZER_ID", "")
    train_csv: str = os.getenv("TRAIN_CSV", "data/vihallu-train.csv")
    test_csv: str = os.getenv("TEST_CSV", "data/vihallu-public-test.csv")
    id_column: str = os.getenv("ID_COLUMN", "id")
    context_column: str = os.getenv("CONTEXT_COLUMN", "context")
    prompt_column: str = os.getenv("PROMPT_COLUMN", "prompt")
    response_column: str = os.getenv("RESPONSE_COLUMN", "response")
    label_column: str = os.getenv("LABEL_COLUMN", "label")
    prompt_type_column: str = os.getenv("PROMPT_TYPE_COLUMN", "prompt_type")
    text_column: str = os.getenv("TEXT_COLUMN", "text")

    num_labels: int = _getenv_int("NUM_LABELS", 3)
    max_length: int = _getenv_int("MAX_LEN", 1024)

    k_sentences: int = _getenv_int("K_SENTENCES", 10)
    use_prompt_type_tag: bool = _getenv_bool("USE_PROMPT_TYPE_TAG", True)
    use_keyword_extraction: bool = _getenv_bool("USE_KEYWORD_EXTRACTION", True)

    output_dir: str = os.getenv("OUTPUT_DIR", "outputs/default")
    epochs: int = _getenv_int("EPOCHS", 4)
    train_bs: int = _getenv_int("TRAIN_BS", 2)
    eval_bs: int = _getenv_int("EVAL_BS", 4)
    grad_accum: int = _getenv_int("GRAD_ACCUM", 4)

    lr: float = _getenv_float("LR", 3e-5)
    weight_decay: float = _getenv_float("WEIGHT_DECAY", 0.01)
    warmup_ratio: float = _getenv_float("WARMUP_RATIO", 0.05)

    lora_r: int = _getenv_int("LORA_R", 32)
    lora_alpha: int = _getenv_int("LORA_ALPHA", 64)  #
    lora_dropout: float = _getenv_float("LORA_DROPOUT", 0.05)

    valid_size: float = _getenv_float("VALID_SIZE", 0.15)
    random_state: int = _getenv_int("RANDOM_STATE", 42)

    use_class_weights: bool = _getenv_bool("USE_CLASS_WEIGHTS", True)
    label_smoothing: float = _getenv_float("LABEL_SMOOTHING", 0.1)

    logging_steps: int = _getenv_int("LOG_STEPS", 25)
    eval_steps: int = _getenv_int("EVAL_STEPS", 100)
    save_steps: int = _getenv_int("SAVE_STEPS", 100)
    save_total_limit: int = _getenv_int("SAVE_TOTAL_LIMIT", 3)

    gradient_checkpointing: bool = _getenv_bool("GRADIENT_CHECKPOINTING", True)
    bf16: bool = _getenv_bool("BF16", True)
    fp16: bool = _getenv_bool("FP16", False)

    max_grad_norm: float = _getenv_float("MAX_GRAD_NORM", 0.3)

    compute_metrics = None
    class_weights: Optional[list] = None

    @property
    def effective_tokenizer_id(self) -> str:
        return self.tokenizer_id or self.model_id

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size with gradient accumulation"""
        return self.train_bs * self.grad_accum

    def __post_init__(self):
        """Validate configuration"""
        assert self.num_labels == 3, "Must be 3-class: NO, INTRINSIC, EXTRINSIC"
        assert self.max_length <= 2048, "Max length should be <= 2048 for LLaMA"
        assert 0 < self.valid_size < 0.5, "Valid size should be between 0 and 0.5"
