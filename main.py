"""
Main training pipeline for hallucination detection
Enhanced with better metrics and comprehensive evaluation
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from datasets import Dataset
from src.config import Config
from src.peft_model import build_datasets, get_trainer, ID2LABEL, to_label_id_robust
from src.preprocessing import normalize_light_vi


def compute_metrics_detailed(eval_pred):
    """Comprehensive metrics for 3-class classification"""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_per_class = f1_score(labels, predictions, average=None)

    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_no': f1_per_class[0],
        'f1_intrinsic': f1_per_class[1],
        'f1_extrinsic': f1_per_class[2],
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'accuracy': accuracy,
    }


def print_detailed_evaluation(labels, predictions, label_names=None):
    """Print comprehensive evaluation report"""
    if label_names is None:
        label_names = ['NO', 'INTRINSIC', 'EXTRINSIC']

    print("\n" + "="*70)
    print("DETAILED EVALUATION REPORT")
    print("="*70)

    print("\nClassification Report:")
    print(classification_report(labels, predictions,
          target_names=label_names, digits=4))

    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(f"{'':15} " + " ".join(f"{ln:>12}" for ln in label_names))
    for i, row in enumerate(cm):
        print(f"{label_names[i]:15} " + " ".join(f"{val:>12}" for val in row))

    print("\nPer-class Accuracy:")
    for i, label in enumerate(label_names):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            print(f"  {label:15}: {acc:.4f} ({cm[i, i]}/{cm[i].sum()})")

    print("="*70 + "\n")


def prepare_data_splits(df: pd.DataFrame, C: Config):
    """Prepare stratified train/val split with data validation"""

    required_cols = {
        C.id_column, C.context_column, C.prompt_column,
        C.response_column, C.label_column
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print("[main] Normalizing text columns...")
    for col in [C.context_column, C.prompt_column, C.response_column]:
        df[col] = df[col].astype(str).fillna("").apply(normalize_light_vi)

    df['label_id'] = df[C.label_column].apply(to_label_id_robust)

    print("\n[main] Label distribution (before filtering):")
    print(df[C.label_column].value_counts())
    print("\n[main] Mapped label distribution:")
    print(df['label_id'].value_counts(dropna=False))

    before_len = len(df)
    df = df.dropna(subset=['label_id']).reset_index(drop=True)
    after_len = len(df)
    if after_len < before_len:
        print(
            f"\n[main] Removed {before_len - after_len} samples with invalid labels")

    if len(df) == 0:
        raise ValueError("No valid samples after label mapping!")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=C.valid_size,
        random_state=C.random_state
    )

    train_idx, val_idx = next(sss.split(df, df['label_id']))

    select_cols = [
        C.id_column, C.context_column, C.prompt_column,
        C.response_column, C.label_column
    ]
    if hasattr(C, 'prompt_type_column') and C.prompt_type_column in df.columns:
        select_cols.append(C.prompt_type_column)

    train_df = df.iloc[train_idx][select_cols].reset_index(drop=True)
    val_df = df.iloc[val_idx][select_cols].reset_index(drop=True)

    print(f"\n[main] Split sizes: train={len(train_df)}, val={len(val_df)}")
    print("\n[main] Train label distribution:")
    print(train_df[C.label_column].value_counts())
    print("\n[main] Val label distribution:")
    print(val_df[C.label_column].value_counts())

    return train_df, val_df


def prepare_inference_dataset(df: pd.DataFrame, tok, C: Config):
    """Prepare dataset for inference"""

    def tokenize_fn(batch):
        return tok(
            batch[C.text_column],
            truncation=True,
            max_length=C.max_length,
            padding=False
        )

    ds = Dataset.from_pandas(
        df[[C.id_column, C.text_column]].reset_index(drop=True))

    remove_cols = [c for c in ds.column_names if c != C.id_column]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    return ds


def save_predictions_csv(output_path: str, ids, pred_ids, id2label):
    """Save predictions to CSV"""
    submission = pd.DataFrame({
        "id": ids,
        "predict_label": [id2label[int(p)] for p in pred_ids]
    })

    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nâœ… Predictions saved to: {output_path}")

    print("\nPrediction distribution:")
    dist = submission["predict_label"].value_counts()
    print(dist)
    print("\nPercentages:")
    print((dist / len(submission) * 100).round(2))

    return submission


def main():
    print("\n" + "="*70)
    print("HALLUCINATION DETECTION - TRAINING PIPELINE")
    print("="*70 + "\n")

    C = Config()

    print(f"Model: {C.model_id}")
    print(f"Output dir: {C.output_dir}")
    print(f"Epochs: {C.epochs}")
    print(f"Effective batch size: {C.effective_batch_size}")
    print(f"Learning rate: {C.lr}")
    print(f"LoRA r={C.lora_r}, alpha={C.lora_alpha}")
    print(f"Max length: {C.max_length}")
    print(f"K sentences: {C.k_sentences}")

    print(f"\n[main] Loading training data from: {C.train_csv}")
    df = pd.read_csv(C.train_csv)
    print(f"Loaded {len(df)} samples")

    train_df, val_df = prepare_data_splits(df, C)
    print("\n[main] Building HuggingFace datasets...")
    tok, train_ds, val_ds = build_datasets(train_df, val_df, C)

    print(f"\nDataset sizes: train={len(train_ds)}, val={len(val_ds)}")
    print(f"Features: {train_ds.features}")

    C.compute_metrics = compute_metrics_detailed

    print("\n[main] Initializing trainer...")
    trainer = get_trainer(train_ds, val_ds, tok, C)

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    train_result = trainer.train()

    print("\n[main] Saving model...")
    os.makedirs(C.output_dir, exist_ok=True)
    trainer.save_model(C.output_dir)
    tok.save_pretrained(C.output_dir)

    train_info = {
        'model_id': C.model_id,
        'num_labels': C.num_labels,
        'max_length': C.max_length,
        'epochs': C.epochs,
        'learning_rate': C.lr,
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'train_loss': train_result.training_loss,
    }

    with open(f"{C.output_dir}/train_info.json", 'w') as f:
        json.dump(train_info, f, indent=2)

    print(f"âœ… Model and tokenizer saved to: {C.output_dir}")

    print("\n" + "="*70)
    print("FINAL VALIDATION EVALUATION")
    print("="*70)

    eval_result = trainer.evaluate()
    print("\nValidation metrics:")
    for k, v in eval_result.items():
        if not k.startswith('eval_'):
            continue
        print(f"  {k}: {v:.4f}")

    val_predictions = trainer.predict(val_ds)
    val_pred_labels = val_predictions.predictions.argmax(axis=-1)
    val_true_labels = val_predictions.label_ids

    print_detailed_evaluation(val_true_labels, val_pred_labels)

    print("\n" + "="*70)
    print("INFERENCE ON TEST SET")
    print("="*70 + "\n")

    if not os.path.exists(C.test_csv):
        print(f"Test file not found: {C.test_csv}")
        print("Skipping test inference.")
    else:
        print(f"Loading test data from: {C.test_csv}")
        test_df = pd.read_csv(C.test_csv)
        print(f"Test samples: {len(test_df)}")

        for col in [C.context_column, C.prompt_column, C.response_column]:
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(
                    str).fillna("").apply(normalize_light_vi)
            else:
                test_df[col] = ""

        if C.text_column not in test_df.columns:
            from src.preprocessing import build_text

            print("Building text representations...")

            prompt_type_col = getattr(C, 'prompt_type_column', None)
            test_df[C.text_column] = test_df.apply(
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

        test_inf_ds = prepare_inference_dataset(test_df, tok, C)

        print("Running inference...")
        test_predictions = trainer.predict(test_inf_ds)
        test_logits = test_predictions.predictions
        if test_logits.ndim == 2:
            test_pred_ids = test_logits.argmax(axis=1)
        else:
            # Binary case fallback
            test_pred_ids = (
                1 / (1 + np.exp(-test_logits.reshape(-1))) >= 0.5).astype(int)

        # Save predictions
        output_path = "preds_test.csv"
        save_predictions_csv(
            output_path,
            test_df[C.id_column].values,
            test_pred_ids,
            ID2LABEL
        )

        # Also save to output dir
        save_predictions_csv(
            f"{C.output_dir}/preds_test.csv",
            test_df[C.id_column].values,
            test_pred_ids,
            ID2LABEL
        )

    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
