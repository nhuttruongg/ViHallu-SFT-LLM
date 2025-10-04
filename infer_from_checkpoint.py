"""
Inference from saved checkpoint (Modal GPU deployment)
Optimized for batch inference with proper error handling
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import PeftModel
from tqdm import tqdm

from src.config import Config
from src.preprocessing import normalize_light_vi, build_text

# Label mappings
LABEL2ID = {"no": 0, "intrinsic": 1, "extrinsic": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ====== Model Loading ======
def load_peft_model_for_inference(base_model_id: str, peft_checkpoint_dir: str):
    """Load base model + LoRA adapter for inference"""

    print(f"Loading base model: {base_model_id}")
    print(f"Loading LoRA adapter: {peft_checkpoint_dir}")

    # Quantization config (use fp16 for broader GPU compatibility)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Model config
    cfg = AutoConfig.from_pretrained(
        base_model_id,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )

    # Load base model with quantization
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, peft_checkpoint_dir)
    model.eval()

    # Load tokenizer from checkpoint (has same config as training)
    tokenizer = AutoTokenizer.from_pretrained(
        peft_checkpoint_dir,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Model loaded successfully")
    print(f"   Device: {model.device}")
    print(f"   Dtype: {model.dtype}")

    return model, tokenizer


# ====== Batch Inference ======
@torch.inference_mode()
def predict_batch(model, tokenizer, texts: list, max_length: int, batch_size: int = 32):
    """
    Run inference in batches with progress bar
    Returns logits as numpy array
    """
    all_logits = []

    print(
        f"\nRunning inference on {len(texts)} samples (batch_size={batch_size})...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(model.device)

        # Forward pass
        outputs = model(**encoded)
        logits = outputs.logits

        # Convert to CPU float32 for stability
        logits_cpu = logits.to(dtype=torch.float32, device="cpu").numpy()
        all_logits.append(logits_cpu)

    # Concatenate all batches
    if all_logits:
        return np.concatenate(all_logits, axis=0)
    else:
        return np.zeros((0, 3), dtype=np.float32)


# ====== Save Predictions ======
def save_predictions_to_csv(
    output_path: str,
    ids: np.ndarray,
    pred_ids: np.ndarray,
    id2label: dict,
    logits: np.ndarray = None
):
    """Save predictions with optional confidence scores"""

    # Create submission dataframe
    submission = pd.DataFrame({
        "id": ids,
        "predict_label": [id2label[int(p)] for p in pred_ids]
    })

    # Add confidence scores if logits provided
    if logits is not None and logits.ndim == 2:
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        max_probs = probs.max(axis=1)
        submission["confidence"] = max_probs

    # Save to CSV
    submission.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"   Total samples: {len(submission)}")

    # Print distribution
    print("\nPrediction distribution:")
    dist = submission["predict_label"].value_counts()
    print(dist)

    print("\nPercentages:")
    pct = (dist / len(submission) * 100).round(2)
    print(pct)

    # Confidence statistics if available
    if "confidence" in submission.columns:
        print(f"\nConfidence statistics:")
        print(f"   Mean: {submission['confidence'].mean():.4f}")
        print(f"   Median: {submission['confidence'].median():.4f}")
        print(f"   Min: {submission['confidence'].min():.4f}")
        print(f"   Max: {submission['confidence'].max():.4f}")

    return submission


# ====== Main Inference ======
def main():
    print("\n" + "="*70)
    print("INFERENCE FROM CHECKPOINT")
    print("="*70 + "\n")

    # Load config
    C = Config()

    # Get environment variables
    CKPT_DIR = os.getenv("CKPT_DIR", "")
    BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", C.model_id)
    TEST_CSV = os.getenv("TEST_CSV", C.test_csv)
    OUT_NAME = os.getenv("OUT_NAME", "preds_test.csv")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

    # Validate checkpoint directory
    if not CKPT_DIR:
        raise ValueError("❌ CKPT_DIR environment variable not set!")

    if not os.path.exists(CKPT_DIR):
        raise ValueError(f"❌ Checkpoint directory not found: {CKPT_DIR}")

    print(f"Checkpoint: {CKPT_DIR}")
    print(f"Base model: {BASE_MODEL_ID}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Output: {OUT_NAME}")
    print(f"Batch size: {BATCH_SIZE}")

    # Load model and tokenizer
    model, tokenizer = load_peft_model_for_inference(BASE_MODEL_ID, CKPT_DIR)

    # Load test data
    print(f"\nLoading test data from: {TEST_CSV}")
    if not os.path.exists(TEST_CSV):
        raise ValueError(f"❌ Test CSV not found: {TEST_CSV}")

    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df)} test samples")

    # Normalize text columns
    print("\nNormalizing text columns...")
    for col in [C.context_column, C.prompt_column, C.response_column]:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(
                str).fillna("").apply(normalize_light_vi)
        else:
            test_df[col] = ""
            print(f"⚠️  Column '{col}' not found, using empty string")

    # Build text representations
    if C.text_column not in test_df.columns:
        print(f"\nBuilding text column (k_sent={C.k_sentences})...")

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

    # Run inference
    texts = test_df[C.text_column].tolist()
    logits = predict_batch(model, tokenizer, texts, C.max_length, BATCH_SIZE)

    # Get predictions
    if logits.ndim == 2 and logits.shape[1] == 3:
        pred_ids = logits.argmax(axis=1)
    else:
        # Fallback for unexpected output shape
        print(f"⚠️  Unexpected logits shape: {logits.shape}")
        pred_ids = (1 / (1 + np.exp(-logits.reshape(-1))) >= 0.5).astype(int)

    # Save predictions
    save_predictions_to_csv(
        OUT_NAME,
        test_df[C.id_column].values,
        pred_ids,
        ID2LABEL,
        logits=logits
    )

    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
