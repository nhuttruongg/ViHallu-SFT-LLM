"""
Enhanced PEFT model for hallucination detection
- Class-weighted loss for imbalanced data
- Better LoRA configuration
- Optimized training settings
"""
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.utils.class_weight import compute_class_weight
from src.config import Config
from src.preprocessing import build_text, normalize_light_vi
from src.utils import should_trust_remote_code

LABEL2ID = {"no": 0, "intrinsic": 1, "extrinsic": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def to_label_id_robust(x):
    """Convert various label formats to integer ID"""
    if x is None or pd.isna(x):
        return None

    s = str(x).strip().lower()
    if not s:
        return None

    if s.isdigit():
        i = int(s)
        return i if i in ID2LABEL else None

    aliases = {
        "no": 0, "none": 0, "non-hallucination": 0, "không": 0,
        "intrinsic": 1, "internal": 1, "ná»™i táº¡i": 1, "intrinsic hallucination": 1,
        "extrinsic": 2, "external": 2, "ngoáº¡i táº¡i": 2, "extrinsic hallucination": 2,
    }
    return aliases.get(s, LABEL2ID.get(s))


def ensure_text_column(df: pd.DataFrame, C: Config) -> pd.DataFrame:
    """Ensure text column exists with proper preprocessing"""
    df = df.copy()

    for col in [C.context_column, C.prompt_column, C.response_column]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("").apply(normalize_light_vi)

    prompt_type_col = getattr(C, 'prompt_type_column', None)
    if prompt_type_col and prompt_type_col in df.columns:
        df[prompt_type_col] = df[prompt_type_col].fillna("")
    else:
        df[prompt_type_col] = None

    if C.text_column not in df.columns:
        print(
            f"[peft_model] Building text column with k_sent={C.k_sentences}...")
        df[C.text_column] = df.apply(
            lambda row: build_text(
                context=row[C.context_column],
                prompt=row[C.prompt_column],
                response=row[C.response_column],
                prompt_type=row.get(
                    prompt_type_col) if prompt_type_col else None,
                k_sent=C.k_sentences,
                use_prompt_type_tag=C.use_prompt_type_tag,
                use_keywords=C.use_keyword_extraction
            ),
            axis=1
        )

    return df


def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, C: Config):
    """Build HuggingFace datasets with proper tokenization"""

    train_df = ensure_text_column(train_df, C)
    val_df = ensure_text_column(val_df, C)
    if C.label_column not in train_df.columns:
        raise ValueError(
            f"Missing label column '{C.label_column}' in train_df")

    print("\n[peft_model] Label distribution (train - raw):")
    print(train_df[C.label_column].value_counts(dropna=False))
    print("\n[peft_model] Label distribution (val - raw):")
    print(val_df[C.label_column].value_counts(dropna=False))

    train_df["labels"] = train_df[C.label_column].apply(to_label_id_robust)
    val_df["labels"] = val_df[C.label_column].apply(to_label_id_robust)

    print("\n[peft_model] Label distribution (train - mapped):")
    print(train_df["labels"].value_counts(dropna=False))
    print("\n[peft_model] Label distribution (val - mapped):")
    print(val_df["labels"].value_counts(dropna=False))

    before_train, before_val = len(train_df), len(val_df)
    train_df = train_df.dropna(subset=["labels"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["labels"]).reset_index(drop=True)
    after_train, after_val = len(train_df), len(val_df)

    print(
        f"\n[peft_model] Dropped {before_train - after_train} train, {before_val - after_val} val samples")

    if after_train == 0 or after_val == 0:
        raise ValueError(
            f"Empty dataset after label mapping: train={after_train}, val={after_val}"
        )

    if C.use_class_weights:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array([0, 1, 2]),
            y=train_df["labels"].values
        )
        C.class_weights = class_weights.tolist()
        print(f"\n[peft_model] Class weights: {C.class_weights}")

    tok = AutoTokenizer.from_pretrained(
        C.effective_tokenizer_id,
        use_fast=True,
        trust_remote_code=should_trust_remote_code(C.effective_tokenizer_id)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_function(batch):
        return tok(
            batch[C.text_column],
            max_length=C.max_length,
            truncation=True,
            padding=False,
        )

    cols = [C.text_column, "labels"]
    train_ds = Dataset.from_pandas(train_df[cols])
    val_ds = Dataset.from_pandas(val_df[cols])

    remove_cols = [C.text_column]
    if "__index_level_0__" in train_ds.column_names:
        remove_cols.append("__index_level_0__")

    train_ds = train_ds.map(
        tokenize_function, batched=True, remove_columns=remove_cols)
    val_ds = val_ds.map(tokenize_function, batched=True,
                        remove_columns=remove_cols)

    return tok, train_ds, val_ds


class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted loss"""

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights,
                                  dtype=logits.dtype, device=logits.device)
            loss_fct = nn.CrossEntropyLoss(
                weight=weight, label_smoothing=self.label_smoothing)
        else:
            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing)

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def get_trainer(train_ds, val_ds, tok, C: Config):
    """Setup QLoRA trainer with optimized settings"""

    print(f"\n[peft_model] Loading base model: {C.model_id}")

    cfg = AutoConfig.from_pretrained(
        C.model_id,
        num_labels=C.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=should_trust_remote_code(C.model_id),
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
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

    if C.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=C.lora_r,
        lora_alpha=C.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=C.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_cfg)

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[peft_model] Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    data_collator = DataCollatorWithPadding(tokenizer=tok, padding=True)

    args = TrainingArguments(
        output_dir=C.output_dir,
        per_device_train_batch_size=C.train_bs,
        per_device_eval_batch_size=C.eval_bs,
        gradient_accumulation_steps=C.grad_accum,
        num_train_epochs=C.epochs,
        learning_rate=C.lr,
        weight_decay=C.weight_decay,
        max_grad_norm=C.max_grad_norm,
        lr_scheduler_type="cosine",
        warmup_ratio=C.warmup_ratio,
        eval_strategy="steps",
        eval_steps=C.eval_steps,
        logging_steps=C.logging_steps,
        save_steps=C.save_steps,
        save_total_limit=C.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=C.bf16,
        fp16=C.fp16,
        report_to="none",
        seed=C.random_state,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=C.compute_metrics,
        class_weights=C.class_weights if C.use_class_weights else None,
        label_smoothing=C.label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    return trainer
