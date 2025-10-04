"""
Ensemble predictions from multiple model checkpoints
Supports: majority voting, weighted voting, confidence-based selection
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from collections import Counter


LABEL2ID = {"no": 0, "intrinsic": 1, "extrinsic": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_predictions(file_paths: List[str]) -> List[pd.DataFrame]:
    """Load prediction CSVs"""
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        print(f"Loaded: {path} ({len(df)} samples)")
        dfs.append(df)
    return dfs


def majority_voting(predictions: List[pd.DataFrame]) -> pd.DataFrame:
    """Simple majority voting ensemble"""

    # Assume all have same IDs in same order
    ids = predictions[0]['id'].values

    # Collect all predictions
    all_preds = []
    for df in predictions:
        pred_ids = df['predict_label'].map(LABEL2ID).values
        all_preds.append(pred_ids)

    all_preds = np.array(all_preds)  # Shape: (num_models, num_samples)

    # Majority vote
    from scipy.stats import mode
    ensemble_ids, counts = mode(all_preds, axis=0)
    ensemble_ids = ensemble_ids.flatten()

    # Convert back to labels
    ensemble_labels = [ID2LABEL[int(i)] for i in ensemble_ids]

    # Calculate voting confidence
    vote_confidence = counts.flatten() / len(predictions)

    result = pd.DataFrame({
        'id': ids,
        'predict_label': ensemble_labels,
        'vote_confidence': vote_confidence,
    })

    return result


def confidence_based_voting(predictions: List[pd.DataFrame]) -> pd.DataFrame:
    """Weighted voting based on prediction confidence"""

    # Check if all predictions have confidence scores
    has_confidence = all('confidence' in df.columns for df in predictions)

    if not has_confidence:
        print(
            "⚠️  Not all predictions have confidence scores. Using majority voting instead.")
        return majority_voting(predictions)

    ids = predictions[0]['id'].values
    num_samples = len(ids)

    # Aggregate confidence-weighted votes
    vote_matrix = np.zeros((num_samples, 3))  # (samples, 3 classes)

    for df in predictions:
        pred_ids = df['predict_label'].map(LABEL2ID).values
        confidences = df['confidence'].values

        for i, (pred_id, conf) in enumerate(zip(pred_ids, confidences)):
            vote_matrix[i, pred_id] += conf

    # Select class with highest weighted vote
    ensemble_ids = vote_matrix.argmax(axis=1)
    ensemble_labels = [ID2LABEL[int(i)] for i in ensemble_ids]

    # Average confidence
    max_votes = vote_matrix.max(axis=1)
    total_votes = vote_matrix.sum(axis=1)
    avg_confidence = max_votes / total_votes

    result = pd.DataFrame({
        'id': ids,
        'predict_label': ensemble_labels,
        'confidence': avg_confidence,
    })

    return result


def weighted_voting(
    predictions: List[pd.DataFrame],
    weights: List[float]
) -> pd.DataFrame:
    """Weighted voting with custom weights for each model"""

    if len(weights) != len(predictions):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of predictions ({len(predictions)})")

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    ids = predictions[0]['id'].values
    num_samples = len(ids)

    # Aggregate weighted votes
    vote_matrix = np.zeros((num_samples, 3))

    for df, weight in zip(predictions, weights):
        pred_ids = df['predict_label'].map(LABEL2ID).values

        for i, pred_id in enumerate(pred_ids):
            vote_matrix[i, pred_id] += weight

    # Select class with highest vote
    ensemble_ids = vote_matrix.argmax(axis=1)
    ensemble_labels = [ID2LABEL[int(i)] for i in ensemble_ids]

    result = pd.DataFrame({
        'id': ids,
        'predict_label': ensemble_labels,
    })

    return result


def analyze_agreement(predictions: List[pd.DataFrame]) -> dict:
    """Analyze agreement between models"""

    all_preds = []
    for df in predictions:
        pred_ids = df['predict_label'].map(LABEL2ID).values
        all_preds.append(pred_ids)

    all_preds = np.array(all_preds)

    # Full agreement
    full_agreement = (all_preds == all_preds[0]).all(axis=0)
    full_agreement_ratio = full_agreement.mean()

    # Majority agreement (at least 50%)
    majority_threshold = len(predictions) / 2
    agreement_counts = []

    for i in range(all_preds.shape[1]):
        counts = Counter(all_preds[:, i])
        max_count = max(counts.values())
        agreement_counts.append(max_count)

    majority_agreement = np.array(agreement_counts) > majority_threshold
    majority_agreement_ratio = majority_agreement.mean()

    # Per-class agreement
    class_agreements = {}
    for class_id, class_name in ID2LABEL.items():
        mask = all_preds[0] == class_id
        if mask.sum() > 0:
            class_full_agreement = full_agreement[mask].mean()
            class_agreements[class_name] = class_full_agreement

    return {
        'full_agreement_ratio': full_agreement_ratio,
        'majority_agreement_ratio': majority_agreement_ratio,
        'class_agreements': class_agreements,
        'avg_agreement': np.mean(agreement_counts) / len(predictions),
    }


def print_ensemble_stats(result: pd.DataFrame, predictions: List[pd.DataFrame]):
    """Print ensemble statistics"""

    print("\n" + "="*70)
    print("ENSEMBLE STATISTICS")
    print("="*70)

    print(f"\nNumber of models: {len(predictions)}")
    print(f"Total samples: {len(result)}")

    # Prediction distribution
    print("\nEnsemble prediction distribution:")
    dist = result['predict_label'].value_counts()
    for label, count in dist.items():
        pct = 100 * count / len(result)
        print(f"  {label:12s}: {count:6,} ({pct:5.2f}%)")

    # Confidence statistics (if available)
    if 'confidence' in result.columns:
        print(f"\nConfidence statistics:")
        print(f"  Mean:   {result['confidence'].mean():.4f}")
        print(f"  Median: {result['confidence'].median():.4f}")
        print(f"  Min:    {result['confidence'].min():.4f}")
        print(f"  Max:    {result['confidence'].max():.4f}")

    if 'vote_confidence' in result.columns:
        print(f"\nVoting confidence:")
        print(
            f"  Full agreement: {(result['vote_confidence'] == 1.0).sum()} samples")
        print(
            f"  Majority vote:  {(result['vote_confidence'] > 0.5).sum()} samples")

    # Agreement analysis
    agreement = analyze_agreement(predictions)
    print(f"\nModel agreement:")
    print(
        f"  Full agreement:     {agreement['full_agreement_ratio']*100:.2f}%")
    print(
        f"  Majority agreement: {agreement['majority_agreement_ratio']*100:.2f}%")
    print(f"  Average agreement:  {agreement['avg_agreement']*100:.2f}%")

    if agreement['class_agreements']:
        print(f"\nPer-class agreement:")
        for class_name, agree in agreement['class_agreements'].items():
            print(f"  {class_name:12s}: {agree*100:.2f}%")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble predictions from multiple models")
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='Input prediction CSV files'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output ensemble CSV file'
    )
    parser.add_argument(
        '--method',
        choices=['voting', 'confidence', 'weighted'],
        default='voting',
        help='Ensemble method (default: voting)'
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        help='Weights for each model (only for weighted method)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ENSEMBLE PREDICTIONS")
    print("="*70)
    print(f"\nMethod: {args.method}")
    print(f"Input files: {len(args.inputs)}")
    print(f"Output: {args.output}\n")

    # Load predictions
    predictions = load_predictions(args.inputs)

    # Validate all have same IDs
    ids_list = [set(df['id'].values) for df in predictions]
    if not all(ids == ids_list[0] for ids in ids_list):
        raise ValueError("⚠️  All prediction files must have the same IDs!")

    # Ensemble
    if args.method == 'voting':
        result = majority_voting(predictions)
    elif args.method == 'confidence':
        result = confidence_based_voting(predictions)
    elif args.method == 'weighted':
        if not args.weights:
            raise ValueError("--weights required for weighted method")
        result = weighted_voting(predictions, args.weights)

    # Print statistics
    print_ensemble_stats(result, predictions)

    # Save
    result.to_csv(args.output, index=False)
    print(f"✅ Ensemble predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
