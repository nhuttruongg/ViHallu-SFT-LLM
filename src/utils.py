"""
Utility functions for data analysis, visualization, and advanced metrics
"""
import numpy as np
import pandas as pd
import os
import warnings
from typing import Dict, List, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import json


def analyze_dataset(df: pd.DataFrame, label_col: str = "label") -> Dict:
    """
    Comprehensive dataset analysis

    Returns statistics about:
    - Label distribution
    - Text length statistics
    - Missing values
    - Data quality issues
    """

    stats = {
        "total_samples": len(df),
        "label_distribution": {},
        "text_lengths": {},
        "missing_values": {},
        "quality_issues": [],
    }

    if label_col in df.columns:
        label_counts = df[label_col].value_counts().to_dict()
        stats["label_distribution"] = label_counts

        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / \
            min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 3:
            stats["quality_issues"].append(
                f"Class imbalance detected: ratio {imbalance_ratio:.2f}:1"
            )

    for col in ["context", "prompt", "response"]:
        if col in df.columns:
            lengths = df[col].astype(str).str.len()
            stats["text_lengths"][col] = {
                "mean": float(lengths.mean()),
                "median": float(lengths.median()),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "std": float(lengths.std()),
            }

            if lengths.min() < 10:
                stats["quality_issues"].append(
                    f"{col}: Has very short texts (min={lengths.min()})"
                )
            if lengths.max() > 5000:
                stats["quality_issues"].append(
                    f"{col}: Has very long texts (max={lengths.max()})"
                )

    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            stats["missing_values"][col] = int(null_count)
            stats["quality_issues"].append(
                f"{col}: {null_count} missing values ({100*null_count/len(df):.1f}%)"
            )

    return stats


def print_dataset_analysis(stats: Dict):
    """Pretty print dataset analysis"""

    print("\n" + "="*70)
    print("DATASET ANALYSIS")
    print("="*70)

    print(f"\nTotal samples: {stats['total_samples']:,}")

    if stats['label_distribution']:
        print("\nLabel Distribution:")
        for label, count in sorted(stats['label_distribution'].items()):
            pct = 100 * count / stats['total_samples']
            print(f"  {label:15s}: {count:6,} ({pct:5.2f}%)")

    if stats['text_lengths']:
        print("\nText Length Statistics:")
        for col, lengths in stats['text_lengths'].items():
            print(f"\n  {col}:")
            print(f"    Mean:   {lengths['mean']:8.1f}")
            print(f"    Median: {lengths['median']:8.1f}")
            print(f"    Min:    {lengths['min']:8,}")
            print(f"    Max:    {lengths['max']:8,}")
            print(f"    Std:    {lengths['std']:8.1f}")

    if stats['missing_values']:
        print("\nMissing Values:")
        for col, count in stats['missing_values'].items():
            print(f"  {col}: {count}")

    if stats['quality_issues']:
        print("\nâš ï¸  Quality Issues:")
        for issue in stats['quality_issues']:
            print(f"  - {issue}")

    print("="*70 + "\n")


def compute_per_class_metrics(y_true, y_pred, labels=None) -> pd.DataFrame:
    """
    Compute detailed per-class metrics

    Returns DataFrame with precision, recall, f1, support for each class
    """

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )

    df = pd.DataFrame({
        'class': labels,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
    })

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracies = []
    for i, label in enumerate(labels):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
        else:
            acc = 0.0
        accuracies.append(acc)

    df['accuracy'] = accuracies

    return df


def analyze_errors(
    df: pd.DataFrame,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    id_col: str = "id",
    text_col: str = "text",
    max_examples: int = 10
) -> Dict:
    """
    Analyze prediction errors

    Returns dictionary with:
    - Confusion pairs (true -> predicted)
    - Example errors for each confusion type
    """

    errors = {
        "total_errors": int((true_labels != pred_labels).sum()),
        "error_rate": float((true_labels != pred_labels).mean()),
        "confusion_pairs": {},
        "examples": {},
    }

    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            pair = f"{true_labels[i]} -> {pred_labels[i]}"

            if pair not in errors["confusion_pairs"]:
                errors["confusion_pairs"][pair] = 0
                errors["examples"][pair] = []

            errors["confusion_pairs"][pair] += 1
            if len(errors["examples"][pair]) < max_examples:
                example = {
                    "id": df.iloc[i][id_col] if id_col in df.columns else i,
                    "true": int(true_labels[i]),
                    "pred": int(pred_labels[i]),
                }
                if text_col in df.columns:
                    text = str(df.iloc[i][text_col])
                    example["text"] = text[:200] + \
                        "..." if len(text) > 200 else text

                errors["examples"][pair].append(example)

    return errors


def print_error_analysis(errors: Dict, label_names: List[str] = None):
    """Pretty print error analysis"""

    if label_names is None:
        label_names = ["NO", "INTRINSIC", "EXTRINSIC"]

    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)

    print(f"\nTotal errors: {errors['total_errors']:,}")
    print(f"Error rate: {errors['error_rate']*100:.2f}%")

    if errors['confusion_pairs']:
        print("\nMost Common Confusion Pairs:")
        sorted_pairs = sorted(
            errors['confusion_pairs'].items(),
            key=lambda x: -x[1]
        )

        for pair, count in sorted_pairs[:10]:
            print(f"  {pair}: {count}")

        top_pair = sorted_pairs[0][0]
        if top_pair in errors['examples']:
            print(f"\nExample errors for '{top_pair}':")
            for i, ex in enumerate(errors['examples'][top_pair][:3], 1):
                print(f"\n  {i}. ID: {ex['id']}")
                print(
                    f"     True: {label_names[ex['true']]}, Pred: {label_names[ex['pred']]}")
                if 'text' in ex:
                    print(f"     Text: {ex['text']}")

    print("="*70 + "\n")


from scipy.special import softmax

def analyze_prediction_confidence(logits: np.ndarray, threshold: float = 0.8) -> Dict:
    """
    Analyze model confidence in predictions

    Args:
        logits: Raw logits from model (N x num_classes)
        threshold: Confidence threshold for "high confidence"

    Returns statistics about prediction confidence
    """

    probs = softmax(logits, axis=1)
    max_probs = probs.max(axis=1)

    stats = {
        "mean_confidence": float(max_probs.mean()),
        "median_confidence": float(np.median(max_probs)),
        "min_confidence": float(max_probs.min()),
        "max_confidence": float(max_probs.max()),
        "std_confidence": float(max_probs.std()),
        "high_confidence_ratio": float((max_probs >= threshold).mean()),
        "low_confidence_count": int((max_probs < 0.5).sum()),
    }

    bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(max_probs, bins=bins)
    stats["confidence_distribution"] = {
        f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(count)
        for i, count in enumerate(hist)
    }

    return stats


def save_evaluation_results(
    output_dir: str,
    metrics: Dict,
    predictions: np.ndarray,
    true_labels: np.ndarray = None,
    ids: np.ndarray = None,
):
    """
    Save comprehensive evaluation results

    Saves:
    - metrics.json: All computed metrics
    - predictions.csv: Predictions with IDs
    - confusion_matrix.txt: Detailed confusion matrix
    """

    import os
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    if ids is not None:
        pred_df = pd.DataFrame({
            'id': ids,
            'prediction': predictions,
        })
        if true_labels is not None:
            pred_df['true_label'] = true_labels
            pred_df['correct'] = predictions == true_labels

        pred_df.to_csv(output_path / "predictions.csv", index=False)

    if true_labels is not None:
        cm = confusion_matrix(true_labels, predictions)

        with open(output_path / "confusion_matrix.txt", 'w') as f:
            f.write("Confusion Matrix\n")
            f.write("="*50 + "\n\n")
            f.write(str(cm) + "\n\n")
            f.write("\nClassification Report\n")
            f.write("="*50 + "\n\n")
            f.write(classification_report(true_labels, predictions))

    print(f"\nâœ… Evaluation results saved to: {output_path}")


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
