# src/infer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import Config
from src.preprocessing import build_text, normalize_light_vi

ID2LABEL = {0: "NO", 1: "INTRINSIC", 2: "EXTRINSIC"}

_model_cache = None


def _load_model(C: Config):
    tok = AutoTokenizer.from_pretrained(
        C.output_dir, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        C.output_dir, trust_remote_code=True)
    return model, tok


def generate(sample: dict, return_prob: bool | None = None):
    """Trả về nhãn (và optional xác suất) cho 1 sample có keys: context, prompt, response (optional prompt_type)."""
    C = Config()
    global _model_cache
    if _model_cache is None:
        _model_cache = _load_model(C)
    model, tok = _model_cache

    text = build_text(
        normalize_light_vi(sample.get(C.context_column, "")),
        normalize_light_vi(sample.get(C.prompt_column, "")),
        normalize_light_vi(sample.get(C.response_column, "")),
        sample.get("prompt_type", None),
        k_sent=7
    )
    inputs = tok(text, return_tensors="pt", truncation=True,
                 max_length=C.max_length).to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = logits.softmax(-1)[0].tolist()
    label = ID2LABEL[int(logits.argmax(-1).item())]
    if return_prob:
        return label, probs
    return label
