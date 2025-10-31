"""
Modal Inference App for Vietnamese Hallucination Detection
Serves the fine-tuned model via HTTP endpoints
"""
from __future__ import annotations
import os
import json
import time
from typing import Dict, Optional
from dataclasses import dataclass
import re
import math
import unicodedata
from collections import Counter

import numpy as np
import torch
from modal import App, Image, Volume, Secret, asgi_app, method, web_endpoint, enter

# ==============================================================================
# PREPROCESSING UTILITIES (copied from src/preprocessing.py)
# ==============================================================================
_ZW = re.compile(r'[\u200B-\u200D\uFEFF]')
_MULTI_PUNC = re.compile(r'([!?.,;:])\1{2,}')
_MULTI_WS = re.compile(r'\s+')
_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SENT_SPLIT = re.compile(r'(?<=[\.!\?…])\s+|\n+')

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

def _sent_split(text: str):
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

def extract_keywords(text: str, top_k: int = 5):
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

def build_text(context: str, prompt: str, response: str, prompt_type: Optional[str] = None, 
               k_sent: int = 10, use_prompt_type_tag: bool = True, use_keywords: bool = True) -> str:
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

# ==============================================================================
# MODAL SETUP
# ==============================================================================
APP_NAME = "vihallu-inference"
app = App(APP_NAME)

try:
    HF_SECRET = Secret.from_name("huggingface-token")
except Exception:
    HF_SECRET = None

ARTIFACT_VOLUME = Volume.from_name("vihallu-artifacts", create_if_missing=False)
HF_CACHE = Volume.from_name("hf-cache", create_if_missing=True)

image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "torchaudio", 
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.43.3", "accelerate==0.30.1", "peft==0.11.1", 
        "bitsandbytes==0.43.1", "tokenizers==0.19.1", "numpy==1.25.2",
        "sentencepiece==0.1.99", "protobuf==3.20.3", "safetensors",
        "fastapi==0.109.0", "pydantic==2.5.3"
    )
)

LABEL2ID = {"no": 0, "intrinsic": 1, "extrinsic": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ==============================================================================
# MODEL CLASS
# ==============================================================================
@app.cls(
    image=image,
    gpu="A10G",  # Use A10G for inference (cheaper than A100)
    timeout=3600,
    volumes={"/outputs": ARTIFACT_VOLUME, "/root/.cache/huggingface": HF_CACHE},
    secrets=[HF_SECRET] if HF_SECRET else [],
    scaledown_window=300,  # Keep warm for 5 minutes
)
class HallucinationDetector:
    """Modal class for hallucination detection inference"""
    
    @enter()
    def _setup(self):
        """Load model from checkpoint"""
        from transformers import (
            AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
            BitsAndBytesConfig
        )
        from peft import PeftModel, PeftConfig
        
        print("🔄 Loading model from checkpoint...")
        start_time = time.time()
        
        model_path = "/outputs/finetuned-model"
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # Load base model
        cfg = AutoConfig.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True,
        )
        
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            config=cfg,
            quantization_config=bnb_cfg,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        
        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"   Device: {next(self.model.parameters()).device}")
    
    @method()
    def predict(
        self,
        context: str,
        prompt: str,
        response: str,
        prompt_type: Optional[str] = None,
        return_probabilities: bool = True,
    ) -> Dict:
        """
        Predict hallucination label for given input
        
        Args:
            context: Background context
            prompt: Question/prompt
            response: Model response to evaluate
            prompt_type: Type of prompt (factual/noisy/adversarial)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        # Build input text
        text = build_text(
            context=context,
            prompt=prompt,
            response=response,
            prompt_type=prompt_type,
            k_sent=10,
            use_prompt_type_tag=True,
            use_keywords=True
        )
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(self.model.device)
        
        # Inference
        with torch.inference_mode():
            outputs = self.model(**encoded)
            logits = outputs.logits.cpu().numpy()[0]
        
        # Get prediction
        pred_id = int(logits.argmax())
        pred_label = ID2LABEL[pred_id]
        
        result = {
            "label": pred_label,
            "label_id": pred_id,
            "processing_time": time.time() - start_time,
        }
        
        if return_probabilities:
            # Convert logits to probabilities
            probs = np.exp(logits) / np.exp(logits).sum()
            result["probabilities"] = {
                "no": float(probs[0]),
                "intrinsic": float(probs[1]),
                "extrinsic": float(probs[2])
            }
            result["confidence"] = float(probs[pred_id])
        
        return result

# ==============================================================================
# WEB ENDPOINTS
# ==============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

web_app = FastAPI(
    title="Vietnamese Hallucination Detection API",
    description="Detect hallucinations in Vietnamese LLM responses",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    context: str = Field(..., description="Background context", min_length=1)
    prompt: str = Field(..., description="Question/prompt", min_length=1)
    response: str = Field(..., description="Model response to evaluate", min_length=1)
    prompt_type: Optional[str] = Field(None, description="Prompt type: factual, noisy, adversarial")
    return_probabilities: bool = Field(True, description="Return probability scores")
    
    class Config:
        json_schema_extra = {
            "example": {
                "context": "Hà Nội là thủ đô của Việt Nam từ năm 1010. Thành phố có diện tích 3.344 km².",
                "prompt": "Thủ đô của Việt Nam là gì?",
                "response": "Thủ đô của Việt Nam là Hà Nội.",
                "prompt_type": "factual",
                "return_probabilities": True
            }
        }

class PredictionResponse(BaseModel):
    label: str
    label_id: int
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    processing_time: float
    explanation: str

@web_app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Vietnamese Hallucination Detection",
        "version": "1.0.0",
        "model": "VinAllama-7B-LoRA",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@web_app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "hallucination-detector",
        "model_loaded": True
    }

@web_app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """
    Predict hallucination label
    
    Returns:
    - no: No hallucination (response is consistent with context)
    - intrinsic: Intrinsic hallucination (response contradicts context)
    - extrinsic: Extrinsic hallucination (response adds info not in context)
    """
    try:
        # Call Modal function
        detector = HallucinationDetector()
        result = detector.predict.remote(
            context=request.context,
            prompt=request.prompt,
            response=request.response,
            prompt_type=request.prompt_type,
            return_probabilities=request.return_probabilities
        )
        
        # Add explanation
        explanations = {
            "no": "✅ Không phát hiện ảo giác (hallucination). Câu trả lời nhất quán với ngữ cảnh.",
            "intrinsic": "⚠️ Phát hiện ảo giác nội tại (intrinsic). Câu trả lời mâu thuẫn với ngữ cảnh.",
            "extrinsic": "⚠️ Phát hiện ảo giác ngoại tại (extrinsic). Câu trả lời chứa thông tin không có trong ngữ cảnh."
        }
        
        result["explanation"] = explanations.get(result["label"], "")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.function(image=image)
@asgi_app()
def fastapi_app():
    """Serve FastAPI app via Modal"""
    return web_app

# ==============================================================================
# CLI INTERFACE
# ==============================================================================
@app.local_entrypoint()
def main(
    context: str,
    prompt: str,
    response: str,
    prompt_type: str = None,
):
    """
    Command-line interface for prediction
    
    Usage:
        modal run modal_inference_app.py --context "..." --prompt "..." --response "..."
    """
    detector = HallucinationDetector()
    
    print("\n" + "="*70)
    print("VIETNAMESE HALLUCINATION DETECTION")
    print("="*70 + "\n")
    
    print("📝 Input:")
    print(f"  Context: {context[:100]}...")
    print(f"  Prompt: {prompt}")
    print(f"  Response: {response}")
    if prompt_type:
        print(f"  Type: {prompt_type}")
    print()
    
    result = detector.predict.remote(
        context=context,
        prompt=prompt,
        response=response,
        prompt_type=prompt_type,
        return_probabilities=True
    )
    
    print("🎯 Prediction:")
    print(f"  Label: {result['label'].upper()}")
    print(f"  Confidence: {result.get('confidence', 0)*100:.2f}%")
    print(f"  Processing time: {result['processing_time']:.3f}s")
    
    if result.get('probabilities'):
        print("\n📊 Probabilities:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {label:12} {prob*100:6.2f}% {bar}")
    
    print("\n" + "="*70 + "\n")

# ==============================================================================
# TEST FUNCTION
# ==============================================================================
@app.function(image=image)
def test_inference():
    """Test the inference pipeline"""
    detector = HallucinationDetector()
    
    test_cases = [
        {
            "name": "No Hallucination",
            "context": "Hà Nội là thủ đô của Việt Nam từ năm 1010.",
            "prompt": "Thủ đô của Việt Nam là gì?",
            "response": "Thủ đô của Việt Nam là Hà Nội.",
            "expected": "no"
        },
        {
            "name": "Intrinsic Hallucination",
            "context": "Hà Nội là thủ đô của Việt Nam.",
            "prompt": "Thủ đô của Việt Nam là gì?",
            "response": "Thủ đô của Việt Nam là Sài Gòn.",
            "expected": "intrinsic"
        },
        {
            "name": "Extrinsic Hallucination",
            "context": "Hà Nội là một thành phố ở Việt Nam.",
            "prompt": "Hà Nội có dân số bao nhiêu?",
            "response": "Hà Nội có dân số khoảng 8 triệu người.",
            "expected": "extrinsic"
        },
    ]
    
    print("\n" + "="*70)
    print("TESTING INFERENCE")
    print("="*70 + "\n")
    
    correct = 0
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        
        result = detector.predict.remote(
            context=test['context'],
            prompt=test['prompt'],
            response=test['response'],
            return_probabilities=True
        )
        
        is_correct = result['label'] == test['expected']
        correct += is_correct
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"  Expected: {test['expected']}")
        print(f"  Got: {result['label']} (confidence: {result.get('confidence', 0)*100:.1f}%)")
        print(f"  {status}\n")
    
    print(f"Results: {correct}/{len(test_cases)} passed ({correct/len(test_cases)*100:.1f}%)")
    print("="*70 + "\n")

# Ensure the app is properly linked and runs
if __name__ == "__main__":
    app.run()