"""
Inference. Scores PARAGRAPH-sized chunks (what the model was trained on), not isolated sentences.
Each sentence in the output inherits its chunk's probability, so reporting.py keeps working.

Blending strategy
-----------------
The trained XGBoost+TF-IDF ensemble is very accurate on the PAN20 academic writing
corpus but can under-score out-of-distribution ChatGPT prose (the lexical TF-IDF model
only recognises n-grams it saw during training).

We therefore blend three signals:
  1. p_model  – the trained bundle (style + lexical n-gram)
  2. p_rule   – a zero-shot stylometric rule scorer (see stylometric_scorer.py)
  3. p_vocab  – raw AI-vocabulary hit rate from p_rule internals

Blending rule (adaptive):
  final = max(p_model, p_rule)   when rule is very confident (p_rule > 0.80)
  final = p_model                when model is very confident and rule is low
  final = 0.4*p_model + 0.6*p_rule  otherwise (uncertain zone)

This guarantees that clear ChatGPT prose is detected even when the lexical model
hasn't seen its exact vocabulary, while preserving the model's precision on PAN20-style
academic AI text.
"""
import json
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from ai_detect.dataset_loader import split_into_sentences
from ai_detect.features import extract_feature_matrix
from ai_detect.stylometric_scorer import rule_score_batch

_DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models", "ai_detection"))
_CACHE: Dict[str, Any] = {}

MIN_CHUNK_WORDS = 40      # merge short paragraphs up to this size
MAX_CHUNK_WORDS = 220     # split very long paragraphs
RELIABLE_WORDS = 25       # below this, never label as AI


def get_detector(model_dir: Optional[str] = None):
    model_dir = os.path.abspath(model_dir or _DEFAULT_MODEL_DIR)
    if model_dir not in _CACHE:
        bpath = os.path.join(model_dir, "detector_bundle.joblib")
        if not os.path.isfile(bpath):
            raise RuntimeError(f"No trained model at {model_dir}. Run train_ai_detector() first "
                               "(old *.joblib files from earlier versions are incompatible).")
        meta = {}
        mpath = os.path.join(model_dir, "model_metadata.json")
        if os.path.isfile(mpath):
            with open(mpath, encoding="utf-8") as f:
                meta = json.load(f)
        _CACHE[model_dir] = (joblib.load(bpath), meta)
    return _CACHE[model_dir]


def _model_scores(texts: List[str], model_dir: Optional[str] = None) -> np.ndarray:
    """P(AI) from the trained bundle only (no rule blending)."""
    b, _ = get_detector(model_dir)
    X = extract_feature_matrix(texts)[:, b["model_cols"]]
    if b.get("use_perplexity"):
        from ai_detect.perplexity import ppl_features
        X = np.hstack([X, ppl_features(texts)])
    p_style = b["style"].predict_proba(b["scaler"].transform(X))[:, 1]
    p_lex = b["lexical"].predict_proba(texts)[:, 1]
    return b["w_style"] * p_style + (1 - b["w_style"]) * p_lex


def score_texts(texts: List[str], model_dir: Optional[str] = None) -> np.ndarray:
    """
    P(AI) for each text, using adaptive blend of trained model + rule scorer.

    Blending logic (per sample):
      - Rule confident AI   (p_rule ≥ 0.60): final = max(p_model, p_rule)
      - Model very confident AI  (p_model ≥ 0.75): final = p_model
      - Both uncertain:          final = 0.30*p_model + 0.70*p_rule
      - Rule confident Human (p_rule ≤ 0.20 and p_model ≤ 0.35): human
    """
    p_model = _model_scores(texts, model_dir)
    p_rule  = rule_score_batch(texts)

    final = np.empty(len(texts), dtype=np.float64)
    for i, (pm, pr) in enumerate(zip(p_model, p_rule)):
        if pr >= 0.60:
            # Rule scorer confident AI → take max (catches OOD ChatGPT text)
            final[i] = max(pm, pr)
        elif pr <= 0.20 and pm <= 0.35:
            # Both agree it's human → take min (conservative)
            final[i] = min(pm, pr)
        elif pm >= 0.75:
            # Trained model is very confident → trust it
            final[i] = pm
        else:
            # Uncertain zone: weight rule scorer higher (generalises better OOD)
            final[i] = 0.30 * pm + 0.70 * pr

    return np.clip(final, 0.0, 1.0)


def _thresholds(meta):
    # We use fixed thresholds calibrated for the hybrid (rule+model) scorer.
    # The metadata threshold (0.6476) was calibrated on PAN20 academic writing
    # alone and is too conservative for out-of-distribution ChatGPT text.
    # Fixed thresholds: human ≤ 0.35, AI ≥ 0.52
    return 0.35, 0.52


def _categorize(prob, words, hu, ai):
    warning = ""
    if prob >= ai:
        cat, conf = "Potentially AI-generated", 0.5 + 0.5 * (prob - ai) / max(1e-6, 1 - ai)
    elif prob <= hu:
        cat, conf = "Likely Human", 0.5 + 0.5 * (hu - prob) / max(1e-6, hu)
    else:
        cat, conf = "Needs Review", 1.0 - abs(prob - (hu + ai) / 2) / max(1e-6, (ai - hu) / 2) * 0.5
    if words < RELIABLE_WORDS:
        warning = f"Only {words} words of text: too short for a reliable verdict."
        if cat == "Potentially AI-generated":
            cat = "Needs Review"
    return cat, round(float(min(max(conf, 0.0), 1.0)), 2), warning


def _chunk_document(text: str) -> List[List[str]]:
    """Return chunks, each a list of sentences, sized for the paragraph-trained model."""
    chunks: List[List[str]] = []
    buf: List[str] = []
    buf_w = 0
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        sents = [s for s in split_into_sentences(para) if len(s.split()) >= 1]
        pw = sum(len(s.split()) for s in sents)
        if pw > MAX_CHUNK_WORDS:                       # split long paragraph
            if buf:
                chunks.append(buf); buf, buf_w = [], 0
            cur, cw = [], 0
            for s in sents:
                cur.append(s); cw += len(s.split())
                if cw >= MAX_CHUNK_WORDS // 2:
                    chunks.append(cur); cur, cw = [], 0
            if cur:
                if cw < MIN_CHUNK_WORDS and chunks:
                    chunks[-1].extend(cur)
                else:
                    chunks.append(cur)
            continue
        if pw >= MIN_CHUNK_WORDS:
            if buf:
                chunks.append(buf); buf, buf_w = [], 0
            chunks.append(sents)
        else:                                          # short paragraph: merge with neighbours
            buf.extend(sents); buf_w += pw
            if buf_w >= MIN_CHUNK_WORDS:
                chunks.append(buf); buf, buf_w = [], 0
    if buf:
        if chunks and buf_w < MIN_CHUNK_WORDS // 2:
            chunks[-1].extend(buf)
        else:
            chunks.append(buf)
    return chunks


def predict_text(text: str, model_dir: Optional[str] = None) -> Dict[str, Any]:
    """Single probability for a whole text (word-weighted over chunks)."""
    return predict_document(text, model_dir)["summary"]


def predict_sentence(sentence: str, context_text: Optional[str] = None,
                     model_dir: Optional[str] = None, **_) -> Dict[str, Any]:
    """Kept for API compatibility. Scores the surrounding context when given: single
    sentences are too short for reliable stylometry."""
    _, meta = get_detector(model_dir)
    hu, ai = _thresholds(meta)
    basis = context_text if context_text and len(context_text.split()) > len(sentence.split()) else sentence
    wc = len(basis.split())
    prob = float(score_texts([basis], model_dir)[0]) if wc >= 2 else 0.0
    cat, conf, warn = _categorize(prob, wc, hu, ai)
    return {"sentence": sentence, "word_count": len(sentence.split()), "prob_ai": round(prob, 3),
            "category": cat, "confidence": conf, "review_needed": cat == "Needs Review", "warning": warn}


def predict_document(text: str, model_dir: Optional[str] = None) -> Dict[str, Any]:
    _, meta = get_detector(model_dir)
    hu, ai = _thresholds(meta)
    chunks = _chunk_document(text)
    if not chunks:
        return {"total_sentences": 0, "sentences": [],
                "summary": {"ai_count": 0, "human_count": 0, "review_count": 0, "ai_percentage": 0.0,
                            "human_percentage": 0.0, "review_percentage": 0.0,
                            "overall_classification": "Empty document"}}

    chunk_texts = [" ".join(c) for c in chunks]
    chunk_words = np.array([len(t.split()) for t in chunk_texts], dtype=float)
    chunk_probs = score_texts(chunk_texts, model_dir)

    results, idx = [], 0
    for ci, (sents, p) in enumerate(zip(chunks, chunk_probs)):
        cat, conf, warn = _categorize(float(p), int(chunk_words[ci]), hu, ai)
        for s in sents:
            results.append({"sentence": s, "word_count": len(s.split()), "prob_ai": round(float(p), 3),
                            "category": cat, "confidence": conf, "review_needed": cat == "Needs Review",
                            "warning": warn, "sentence_index": idx, "paragraph_index": ci})
            idx += 1

    total_w = max(1.0, sum(r["word_count"] for r in results))
    def pct(c): return round(100.0 * sum(r["word_count"] for r in results if r["category"] == c) / total_w, 1)
    ai_pct, hu_pct, rv_pct = pct("Potentially AI-generated"), pct("Likely Human"), pct("Needs Review")
    mean_prob = float(np.average(chunk_probs, weights=chunk_words))

    if total_w < RELIABLE_WORDS:
        overall = "Inconclusive (text too short for reliable detection)"
    elif mean_prob >= ai or ai_pct >= 60:
        overall = "Substantial AI-generated content detected"
    elif ai_pct >= 20 or mean_prob >= (hu + ai) / 2:
        overall = "Mixed document (contains human and potentially AI-generated sections)"
    elif mean_prob <= hu or hu_pct >= 60:
        overall = "Predominantly human-written text"
    else:
        overall = "Inconclusive (majority of content requires human review)"

    return {"total_sentences": len(results), "sentences": results,
            "summary": {"ai_count": sum(r["category"] == "Potentially AI-generated" for r in results),
                        "human_count": sum(r["category"] == "Likely Human" for r in results),
                        "review_count": sum(r["category"] == "Needs Review" for r in results),
                        "ai_percentage": ai_pct, "human_percentage": hu_pct, "review_percentage": rv_pct,
                        "mean_prob_ai": round(mean_prob, 3), "overall_classification": overall,
                        "disclaimer": ("AI detection is a probabilistic estimate based on writing style. "
                                       "It is NOT proof of authorship.")}}
