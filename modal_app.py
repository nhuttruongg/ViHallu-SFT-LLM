"""
Self-contained Modal app for fine-tuning on the ViHallu dataset.
This version incorporates all necessary code from the project's `src` directory
to function as a standalone script, removing the need for local imports or mounts.
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import re
import math
import unicodedata
from collections import Counter
import pandas as pd                                                                       
import numpy as np

# Modal-specific imports
from modal import App, Image, Secret, Volume, gpu

# #############################################################################
# COPIED FROM: src/config.py
# #############################################################################
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
    lora_alpha: int = _getenv_int("LORA_ALPHA", 64)
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
        return self.train_bs * self.grad_accum
    def __post_init__(self):
        assert self.num_labels == 3, "Must be 3-class: NO, INTRINSIC, EXTRINSIC"
        assert self.max_length <= 2048, "Max length should be <= 2048 for LLaMA"
        assert 0 < self.valid_size < 0.5, "Valid size should be between 0 and 0.5"

# #############################################################################
# COPIED FROM: src/preprocessing.py
# #############################################################################
_ZW = re.compile(r'[\u200B-\u200D\uFEFF]')
_MULTI_PUNC = re.compile(r'([!?.,;:])\1{2,}')
_MULTI_WS = re.compile(r'\s+')
_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+|\n+')

def normalize_light_vi(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = _ZW.sub("", s)
    s = _MULTI_PUNC.sub(r"\1\1", s)
    s = _MULTI_WS.sub(" ", s).strip()
    return s

def _tf(text: str) -> Counter:
    tokens = _WORD.findall(text.lower())
    return Counter(tokens)

def _cosine_sim(tf1: Counter, tf2: Counter) -> float:
    norm1 = math.sqrt(sum(v*v for v in tf1.values())) or 1e-9
    norm2 = math.sqrt(sum(v*v for v in tf2.values())) or 1e-9
    dot = sum(tf1[t] * tf2.get(t, 0) for t in tf1)
    return dot / (norm1 * norm2)

def _sent_split(text: str) -> List[str]:
    if not text:
        return []
    sents = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sents if s and len(s.strip()) > 10]

def select_sentences_mmr(context: str, query: str, k: int = 10, lambda_param: float = 0.7) -> str:
    sents = _sent_split(context or "")
    if len(sents) <= k:
        return " ".join(sents)
    query_tf = _tf(query or "")
    sent_tfs = [_tf(s) for s in sents]
    relevance = [_cosine_sim(stf, query_tf) for stf in sent_tfs]
    selected_idx = []
    remaining_idx = list(range(len(sents)))
    first_idx = max(remaining_idx, key=lambda i: relevance[i])
    selected_idx.append(first_idx)
    remaining_idx.remove(first_idx)
    while len(selected_idx) < k and remaining_idx:
        mmr_scores = []
        for i in remaining_idx:
            rel_score = relevance[i]
            max_sim = max(_cosine_sim(sent_tfs[i], sent_tfs[j]) for j in selected_idx) if selected_idx else 0
            mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim
            mmr_scores.append((mmr, i))
        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)
    selected_idx.sort()
    return " ".join(sents[i] for i in selected_idx)

def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    tf = _tf(text)
    stopwords = {"của", "và", "là", "có", "được", "trong", "cho", "từ", "với", "này", "đó", "các", "những", "để", "một", "không"}
    filtered = {w: c for w, c in tf.items() if len(w) > 2 and w not in stopwords}
    top = sorted(filtered.items(), key=lambda x: -x[1])[:top_k]
    return [w for w, _ in top]

PROMPT_TYPE_TAGS = {"factual": "<FACTUAL>", "noisy": "<NOISY>", "adversarial": "<ADVERSARIAL>"}

def add_prompt_type_tag(prompt: str, prompt_type: Optional[str]) -> str:
    if not prompt_type:
        return prompt
    pt_lower = str(prompt_type).strip().lower()
    tag = PROMPT_TYPE_TAGS.get(pt_lower, "")
    if tag:
        return f"{tag}\n{prompt}"
    return prompt

def build_text(context: str, prompt: str, response: str, prompt_type: Optional[str] = None, k_sent: int = 10, use_prompt_type_tag: bool = True, use_keywords: bool = True) -> str:

    context = normalize_light_vi(context)

    prompt = normalize_light_vi(prompt)

    response = normalize_light_vi(response)

    query = f"{prompt} {response}"

    selected_context = select_sentences_mmr(context, query, k=k_sent)

    sections = [f"[CONTEXT]\n{selected_context}"]

    if use_keywords:

        keywords = extract_keywords(selected_context, top_k=5)

        if keywords:

            sections.append(f"[KEYWORDS]\n{', '.join(keywords)}")

    if use_prompt_type_tag and prompt_type:

        prompt = add_prompt_type_tag(prompt, prompt_type)

    sections.append(f"[QUESTION]\n{prompt}")

    sections.append(f"[RESPONSE]\n{response}")

    return "\n\n".join(sections)



# #############################################################################

# COPIED FROM: src/utils.py

# #############################################################################

import warnings

def should_trust_remote_code(model_id: str) -> bool:

    """

    Decide whether to enable `trust_remote_code` for a given model identifier.



    Controls (environment variables):

      - ALLOW_TRUST_REMOTE_CODE: set to true/1 to allow enabling the flag (default: false)

      - TRUSTED_MODELS: optional comma-separated list of model id prefixes allowed when ALLOW_TRUST_REMOTE_CODE is true



    Rationale:

      `trust_remote_code=True` causes Transformers to execute model-specific code

      from the downloaded repository. This can run arbitrary Python and is a

      security risk if model sources are untrusted. Make enabling explicit and

      auditable via env vars.

    """



    if not model_id:

        return False



    allow = os.getenv("ALLOW_TRUST_REMOTE_CODE", "false").lower() in ("1", "true", "yes", "on")

    if not allow:

        return False



    trusted_models = os.getenv("TRUSTED_MODELS", "").strip()

    if trusted_models:

        prefixes = [p.strip() for p in trusted_models.split(",") if p.strip()]

        for p in prefixes:

            if model_id.startswith(p):

                # allowed for this specific model prefix

                warnings.warn(f"Enabling trust_remote_code for model '{model_id}' (matched prefix '{p}'). Ensure this model is audited.", stacklevel=2)

                return True

        # ALLOW set but no matching prefix

        warnings.warn(

            f"ALLOW_TRUST_REMOTE_CODE=True but '{model_id}' is not listed in TRUSTED_MODELS; refusing to enable trust_remote_code.",

            stacklevel=2,

        )

        return False



    # No whitelist provided; allow because ALLOW_TRUST_REMOTE_CODE explicitly enabled

    warnings.warn(f"Enabling trust_remote_code for model '{model_id}' because ALLOW_TRUST_REMOTE_CODE=True (no TRUSTED_MODELS whitelist).", stacklevel=2)

    return True



# #############################################################################

# COPIED FROM: src/peft_model.py and main.py

# #############################################################################

import torch

import torch.nn as nn

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset

from transformers import (

    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,

    BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding,

    EarlyStoppingCallback

)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel, PeftConfig

LABEL2ID = {"no": 0, "intrinsic": 1, "extrinsic": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def to_label_id_robust(x):
    if x is None or pd.isna(x): return None
    s = str(x).strip().lower()
    if not s: return None
    if s.isdigit():
        i = int(s)
        return i if i in ID2LABEL else None
    aliases = {
        "no": 0, "none": 0, "non-hallucination": 0, "không": 0,
        "intrinsic": 1, "internal": 1, "nội tại": 1, "intrinsic hallucination": 1,
        "extrinsic": 2, "external": 2, "ngoại tại": 2, "extrinsic hallucination": 2,
    }
    return aliases.get(s, LABEL2ID.get(s))

def ensure_text_column(df: pd.DataFrame, C: Config) -> pd.DataFrame:
    df = df.copy()
    for col in [C.context_column, C.prompt_column, C.response_column]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].astype(str).fillna("").apply(normalize_light_vi)
    prompt_type_col = getattr(C, 'prompt_type_column', None)
    if prompt_type_col and prompt_type_col in df.columns:
        df[prompt_type_col] = df[prompt_type_col].fillna("")
    else:
        df[prompt_type_col] = None
    if C.text_column not in df.columns:
        df[C.text_column] = df.apply(
            lambda row: build_text(
                context=row[C.context_column], prompt=row[C.prompt_column], response=row[C.response_column],
                prompt_type=row.get(prompt_type_col) if prompt_type_col else None,
                k_sent=C.k_sentences, use_prompt_type_tag=C.use_prompt_type_tag, use_keywords=C.use_keyword_extraction
            ),
            axis=1
        )
    return df

def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, C: Config):
    train_df = ensure_text_column(train_df, C)
    val_df = ensure_text_column(val_df, C)
    if C.label_column not in train_df.columns:
        raise ValueError(f"Missing label column '{C.label_column}' in train_df")
    train_df["labels"] = train_df[C.label_column].apply(to_label_id_robust)
    val_df["labels"] = val_df[C.label_column].apply(to_label_id_robust)
    train_df = train_df.dropna(subset=["labels"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["labels"]).reset_index(drop=True)
    if C.use_class_weights:
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=train_df["labels"].values)
        C.class_weights = class_weights.tolist()
    tok = AutoTokenizer.from_pretrained(
        C.effective_tokenizer_id,
        use_fast=True,
        trust_remote_code=should_trust_remote_code(C.effective_tokenizer_id),
    )
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    def tokenize_function(batch):
        return tok(batch[C.text_column], max_length=C.max_length, truncation=True, padding=False)
    cols = [C.text_column, "labels"]
    train_ds = Dataset.from_pandas(train_df[cols])
    val_ds = Dataset.from_pandas(val_df[cols])
    remove_cols = [C.text_column]
    if "__index_level_0__" in train_ds.column_names: remove_cols.append("__index_level_0__")
    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=remove_cols)
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=remove_cols)
    return tok, train_ds, val_ds

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def get_trainer(train_ds, val_ds, tok, C: Config):
    cfg = AutoConfig.from_pretrained(
        C.model_id,
        num_labels=C.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=should_trust_remote_code(C.model_id),
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if C.bf16 else torch.float16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        C.model_id,
        config=cfg,
        quantization_config=bnb_cfg,
        trust_remote_code=should_trust_remote_code(C.model_id),
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if C.bf16 else torch.float16,
    )
    if C.gradient_checkpointing: model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=C.lora_r, lora_alpha=C.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=C.lora_dropout, bias="none", task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_cfg)
    data_collator = DataCollatorWithPadding(tokenizer=tok, padding=True)
    args = TrainingArguments(
        output_dir=C.output_dir, per_device_train_batch_size=C.train_bs, per_device_eval_batch_size=C.eval_bs,
        gradient_accumulation_steps=C.grad_accum, num_train_epochs=C.epochs, learning_rate=C.lr,
        weight_decay=C.weight_decay, max_grad_norm=C.max_grad_norm, lr_scheduler_type="cosine",
        warmup_ratio=C.warmup_ratio, eval_strategy="steps", eval_steps=C.eval_steps,
        logging_steps=C.logging_steps, save_steps=C.save_steps, save_total_limit=C.save_total_limit,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        bf16=C.bf16, fp16=C.fp16, report_to="none", seed=C.random_state,
        dataloader_pin_memory=True, dataloader_num_workers=2,
    )
    trainer = WeightedLossTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok,
        data_collator=data_collator, compute_metrics=C.compute_metrics,
        class_weights=C.class_weights if C.use_class_weights else None,
        label_smoothing=C.label_smoothing, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer

def compute_metrics_detailed(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_per_class = f1_score(labels, predictions, average=None)
    return {
        'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'f1_no': f1_per_class[0],
        'f1_intrinsic': f1_per_class[1], 'f1_extrinsic': f1_per_class[2],
        'precision_macro': precision_score(labels, predictions, average='macro'),
        'recall_macro': recall_score(labels, predictions, average='macro'),
        'accuracy': accuracy_score(labels, predictions),
    }

def prepare_data_splits(df: pd.DataFrame, C: Config):
    df['label_id'] = df[C.label_column].apply(to_label_id_robust)
    df = df.dropna(subset=['label_id']).reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=C.valid_size, random_state=C.random_state)
    train_idx, val_idx = next(sss.split(df, df['label_id']))
    select_cols = [C.id_column, C.context_column, C.prompt_column, C.response_column, C.label_column]
    if hasattr(C, 'prompt_type_column') and C.prompt_type_column in df.columns:
        select_cols.append(C.prompt_type_column)
    return df.iloc[train_idx][select_cols].reset_index(drop=True), df.iloc[val_idx][select_cols].reset_index(drop=True)

def prepare_inference_dataset(df: pd.DataFrame, tok, C: Config):
    def tokenize_fn(batch):
        return tok(batch[C.text_column], truncation=True, max_length=C.max_length, padding=False)
    ds = Dataset.from_pandas(df[[C.id_column, C.text_column]].reset_index(drop=True))
    remove_cols = [c for c in ds.column_names if c != C.id_column]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    return ds

def save_predictions_csv(output_path: str, ids, pred_ids, id2label):
    submission = pd.DataFrame({"id": ids, "predict_label": [id2label[int(p)] for p in pred_ids]})
    submission.to_csv(output_path, index=False, encoding="utf-8")

# #############################################################################
# MODAL APP DEFINITION
# #############################################################################
APP_NAME = "vihallu-finetune-standalone"
app = App(APP_NAME)

try:
    HF_SECRET = Secret.from_name("huggingface-token")
except Exception:
    HF_SECRET = None

DATASET_VOLUME = Volume.from_name("vihallu-dataset", create_if_missing=True)
ARTIFACT_VOLUME = Volume.from_name("vihallu-artifacts", create_if_missing=True)
HF_CACHE = Volume.from_name("hf-cache", create_if_missing=True)

image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.43.3", "accelerate==0.30.1", "peft==0.11.1", "bitsandbytes==0.43.1",
        "datasets==2.19.1", "tokenizers==0.19.1", "numpy==1.25.2", "pandas==2.0.3", "scikit-learn==1.3.0",
        "tqdm==4.66.1", "sentencepiece==0.1.99", "protobuf==3.20.3", "safetensors", "huggingface_hub",
    )
)

@app.local_entrypoint()
def upload_checkpoint():
    volume = ARTIFACT_VOLUME
    checkpoint_dir = "./checkpoint"
    with volume.batch_upload() as batch:
        for file in os.listdir(checkpoint_dir):
            local_path = os.path.join(checkpoint_dir, file)
            remote_path = f"/outputs/finetuned-model/{file}"
            if os.path.isfile(local_path):
                batch.put_file(local_path, remote_path)
                print(f"Uploaded {local_path} to {remote_path}")
    print("Finished uploading checkpoint files.")

@app.local_entrypoint()
def upload_dataset():
    volume = DATASET_VOLUME
    train_path = "data/vihallu-train.csv"
    test_path = "data/vihallu-public-test.csv"
    with volume.batch_upload() as batch:
        if os.path.exists(train_path):
            batch.put_file(train_path, "vihallu-train.csv")
            print(f"Uploaded {train_path} to volume.")
        if os.path.exists(test_path):
            batch.put_file(test_path, "vihallu-public-test.csv")
            print(f"Uploaded {test_path} to volume.")
    print("Finished uploading dataset files.")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=6 * 3600,
    volumes={"/data": DATASET_VOLUME, "/outputs": ARTIFACT_VOLUME, "/root/.cache/huggingface": HF_CACHE},
    secrets=[HF_SECRET] if HF_SECRET else [],
)
def train():
    C = Config()
    C.train_csv = "/data/vihallu-train.csv"
    C.test_csv = "/data/vihallu-public-test.csv"
    C.output_dir = "/outputs/finetuned-model"
    
    df = pd.read_csv(C.train_csv)
    train_df, val_df = prepare_data_splits(df, C)
    tok, train_ds, val_ds = build_datasets(train_df, val_df, C)
    C.compute_metrics = compute_metrics_detailed
    trainer = get_trainer(train_ds, val_ds, tok, C)
    trainer.train()
    trainer.save_model(C.output_dir)
    tok.save_pretrained(C.output_dir)
    print(f"Model and tokenizer saved to: {C.output_dir}")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=2 * 3600,
    volumes={"/data": DATASET_VOLUME, "/outputs": ARTIFACT_VOLUME, "/root/.cache/huggingface": HF_CACHE},
    secrets=[HF_SECRET] if HF_SECRET else [],
)
def predict():
    C = Config()
    C.output_dir = "/outputs/finetuned-model"
    C.test_csv = "/data/vihallu-public-test.csv"

    # Load the PEFT config
    peft_config = PeftConfig.from_pretrained(C.output_dir)

    # Load the base model
    cfg = AutoConfig.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=C.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=should_trust_remote_code(peft_config.base_model_name_or_path),
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if C.bf16 else torch.float16,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        config=cfg,
        quantization_config=bnb_cfg,
        trust_remote_code=should_trust_remote_code(peft_config.base_model_name_or_path),
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if C.bf16 else torch.float16,
    )

    # Load the PEFT adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, C.output_dir)
    model.eval() # Set to eval mode
    tok = AutoTokenizer.from_pretrained(C.output_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tok, padding=True)
    trainer = Trainer(model=model, data_collator=data_collator)
    test_df = pd.read_csv(C.test_csv)
    
    test_df = ensure_text_column(test_df, C)
    test_inf_ds = prepare_inference_dataset(test_df, tok, C)
    
    test_predictions = trainer.predict(test_inf_ds)
    test_pred_ids = np.argmax(test_predictions.predictions, axis=1)
    
    output_path = "/outputs/preds_test.csv"
    save_predictions_csv(output_path, test_df[C.id_column].values, test_pred_ids, ID2LABEL)
    print(f"Predictions saved to {output_path} in the volume.")

@app.local_entrypoint()
def download():
    target_dir = "./modal-outputs"
    os.makedirs(target_dir, exist_ok=True)
    for entry in ARTIFACT_VOLUME.listdir("/"):
        if entry.type == "file":
            target_path = os.path.join(target_dir, entry.path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(ARTIFACT_VOLUME.read_file(entry.path))
            print(f"Downloaded {entry.path}")
    print("Download complete.")

@app.function(
    image=image,
    volumes={"/outputs": ARTIFACT_VOLUME},
)
def clear_outputs():
    import os
    import shutil
    output_dir = "/outputs"
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print("Cleared the /outputs directory.")
