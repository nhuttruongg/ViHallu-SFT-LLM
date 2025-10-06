# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset
from src.config import Config, ID2LABEL
from src.peft_model import build_datasets, get_trainer, to_label_id_robust
from src.preprocessing import normalize_light_vi, build_text


def compute_metrics_detailed(eval_pred):
    preds, labels = eval_pred
    logits = preds
    y_hat = logits.argmax(axis=-1)
    return {
        "f1_macro": f1_score(labels, y_hat, average="macro"),
        "f1_weighted": f1_score(labels, y_hat, average="weighted"),
        "precision_macro": precision_score(labels, y_hat, average="macro"),
        "recall_macro": recall_score(labels, y_hat, average="macro"),
        "accuracy": accuracy_score(labels, y_hat),
    }


def prepare_data_splits(df: pd.DataFrame, C: Config):
    req = {C.id_column, C.context_column, C.prompt_column,
           C.response_column, C.label_column}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")
    for col in [C.context_column, C.prompt_column, C.response_column]:
        df[col] = df[col].astype(str).fillna("").apply(normalize_light_vi)
    df["label_id"] = df[C.label_column].apply(to_label_id_robust)
    df = df.dropna(subset=["label_id"]).reset_index(drop=True)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=C.valid_size, random_state=C.random_state)
    tr_idx, va_idx = next(sss.split(df, df["label_id"]))
    keep = [C.id_column, C.context_column,
            C.prompt_column, C.response_column, C.label_column]
    if C.prompt_type_column in df.columns:
        keep.append(C.prompt_type_column)
    return df.iloc[tr_idx][keep].reset_index(drop=True), df.iloc[va_idx][keep].reset_index(drop=True)


def main():
    C = Config()
    print("="*70, "\nLoRA FINETUNE - TRAINING PIPELINE\n", "="*70, sep="")
    print(
        f"Model: {C.model_id} | Out: {C.output_dir} | Epochs: {C.epochs} | Eff-BS: {C.effective_batch_size}")

    df = pd.read_csv(C.train_csv)
    train_df, val_df = prepare_data_splits(df, C)
    tok, train_ds, val_ds = build_datasets(train_df, val_df, C)
    C.compute_metrics = compute_metrics_detailed
    trainer = get_trainer(train_ds, val_ds, tok, C)

    print("\n== START TRAIN ==")
    train_result = trainer.train()

    os.makedirs(C.output_dir, exist_ok=True)
    trainer.save_model(C.output_dir)
    tok.save_pretrained(C.output_dir)
    with open(f"{C.output_dir}/train_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_id": C.model_id, "num_labels": C.num_labels, "max_length": C.max_length,
            "epochs": C.epochs, "learning_rate": C.lr, "train_samples": len(train_ds),
            "val_samples": len(val_ds), "train_loss": train_result.training_loss
        }, f, indent=2, ensure_ascii=False)

    print("\n== FINAL EVAL ==")
    eval_res = trainer.evaluate()
    for k, v in sorted(eval_res.items()):
        if k.startswith("eval_"):
            print(f"{k}: {v:.4f}")
    if os.path.exists(C.test_csv):
        test_df = pd.read_csv(C.test_csv)
        for col in [C.context_column, C.prompt_column, C.response_column]:
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(
                    str).fillna("").apply(normalize_light_vi)
            else:
                test_df[col] = ""
        if C.text_column not in test_df.columns:
            pt = C.prompt_type_column if C.prompt_type_column in test_df.columns else None
            test_df[C.text_column] = test_df.apply(lambda r: build_text(
                r[C.context_column], r[C.prompt_column], r[C.response_column],
                r.get(pt) if pt else None, C.k_sentences, C.use_prompt_type_tag,
                C.use_keyword_extraction, selector_mode=C.selector
            ), axis=1)

        def tok_fn(b): return tok(
            b[C.text_column], truncation=True, max_length=C.max_length, padding=False)
        ds = Dataset.from_pandas(test_df[[C.id_column, C.text_column]]).map(
            tok_fn, batched=True, remove_columns=[C.text_column])
        pred = trainer.predict(ds).predictions.argmax(axis=-1)
        sub = pd.DataFrame({"id": test_df[C.id_column].values, "predict_label": [
                           ID2LABEL[int(p)] for p in pred]})
        out = f"{C.output_dir}/preds_test.csv"
        sub.to_csv(out, index=False, encoding="utf-8")
        print(f"Saved test predictions → {out}")

    print("\n== DONE ==")


if __name__ == "__main__":
    main()
