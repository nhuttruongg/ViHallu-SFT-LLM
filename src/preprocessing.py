"""
Enhanced preprocessing for hallucination detection
- Better sentence selection with TF-IDF + semantic overlap
- Keyword extraction for highlighting important terms
- Prompt type tagging for model awareness
"""
import re
import math
import unicodedata
from collections import Counter
from typing import List, Tuple, Optional


_ZW = re.compile(r'[\u200B-\u200D\uFEFF]')
_MULTI_PUNC = re.compile(r'([!?.,;:])\1{2,}')
_MULTI_WS = re.compile(r'\s+')
_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SENT_SPLIT = re.compile(r'(?<=[\.!\?â€¦])\s+|\n+')


def normalize_light_vi(s: str) -> str:
    """Normalize Vietnamese text while preserving case and diacritics"""
    s = unicodedata.normalize("NFKC", str(s))
    s = _ZW.sub("", s)
    s = _MULTI_PUNC.sub(r"\1\1", s)
    s = _MULTI_WS.sub(" ", s).strip()
    return s


def _tf(text: str) -> Counter:
    """Term frequency counter"""
    tokens = _WORD.findall(text.lower())
    return Counter(tokens)


def _cosine_sim(tf1: Counter, tf2: Counter) -> float:
    """Cosine similarity between two TF vectors"""
    norm1 = math.sqrt(sum(v*v for v in tf1.values())) or 1e-9
    norm2 = math.sqrt(sum(v*v for v in tf2.values())) or 1e-9
    dot = sum(tf1[t] * tf2.get(t, 0) for t in tf1)
    return dot / (norm1 * norm2)


def _sent_split(text: str) -> List[str]:
    """Split text into sentences"""
    if not text:
        return []
    sents = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sents if s and len(s.strip()) > 10]


def select_sentences_mmr(
    context: str,
    query: str,
    k: int = 10,
    lambda_param: float = 0.7
) -> str:
    """
    Select top-k sentences using Maximal Marginal Relevance (MMR)
    - Balances relevance to query with diversity
    - lambda_param: tradeoff between relevance (1.0) and diversity (0.0)
    """
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
            max_sim = max(
                _cosine_sim(sent_tfs[i], sent_tfs[j])
                for j in selected_idx
            ) if selected_idx else 0
            mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim
            mmr_scores.append((mmr, i))

        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    selected_idx.sort()
    return " ".join(sents[i] for i in selected_idx)


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """Extract top-k keywords from text using simple TF scoring"""
    tf = _tf(text)

    stopwords = {"của", "và", "là", "có", "được", "trong", "cho", "từ",
                 "với", "này", "đó", "các", "những", "để", "một", "không"}
    filtered = {w: c for w, c in tf.items() if len(w) >
                2 and w not in stopwords}
    top = sorted(filtered.items(), key=lambda x: -x[1])[:top_k]
    return [w for w, _ in top]


PROMPT_TYPE_TAGS = {
    "factual": "<FACTUAL>",
    "noisy": "<NOISY>",
    "adversarial": "<ADVERSARIAL>"
}


def add_prompt_type_tag(prompt: str, prompt_type: Optional[str]) -> str:
    """Add special token to indicate prompt type"""
    if not prompt_type:
        return prompt

    pt_lower = str(prompt_type).strip().lower()
    tag = PROMPT_TYPE_TAGS.get(pt_lower, "")
    if tag:
        return f"{tag}\n{prompt}"
    return prompt


def build_text(
    context: str,
    prompt: str,
    response: str,
    prompt_type: Optional[str] = None,
    k_sent: int = 10,
    use_prompt_type_tag: bool = True,
    use_keywords: bool = True
) -> str:
    """
    Build structured input text for hallucination detection

    Format:
    [CONTEXT] (selected sentences)
    [KEYWORDS] (optional)
    [QUESTION] (with optional type tag)
    [RESPONSE]
    """
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


def compute_overlap_ratio(context: str, response: str) -> float:
    """Compute word overlap ratio between context and response"""
    ctx_words = set(_WORD.findall(context.lower()))
    resp_words = set(_WORD.findall(response.lower()))
    if not resp_words:
        return 0.0
    overlap = len(ctx_words & resp_words)
    return overlap / len(resp_words)


def detect_contradictions(context: str, response: str) -> List[str]:
    """
    Simple heuristic to detect potential contradictions
    Returns list of contradiction indicators found
    """
    indicators = []

    neg_patterns = [
        (r'\bkhông\s+\w+', r'\b(?!không)\w+'),  # "không X" vs "X"
        (r'\bkhông\s+phải', r'\blà\b'),
        (r'\bsai\b', r'\bđúng\b'),
    ]

    context_lower = context.lower()
    response_lower = response.lower()

    for neg, pos in neg_patterns:
        if re.search(neg, context_lower) and re.search(pos, response_lower):
            indicators.append(f"negation_mismatch:{neg}")
        elif re.search(pos, context_lower) and re.search(neg, response_lower):
            indicators.append(f"negation_mismatch:{pos}")

    return indicators
