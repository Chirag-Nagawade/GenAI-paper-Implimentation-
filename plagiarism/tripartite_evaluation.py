"""
Tripartite Benchmark Evaluation for PlagiarismAI.

Decomposes the binary Webis-CPC-11 corpus into three grounded operational categories:
1. EXACT COPYING: Verbatim or near-verbatim copies (including rejected worker submissions).
2. GENUINE PARAPHRASING: Meaning-preserving rewrites with creative lexical/syntactic variation.
3. UNRELATED TEXT: Spam, empty strings, or text with no semantic relationship.
Ambiguous pairs (partial edits / moderate rewrites) are flagged for manual review.

Provides separate benchmark tasks:
- Task 1: Exact Copying Detection (Exact Copying vs. Unrelated)
- Task 2: Genuine Paraphrase Detection (Genuine Paraphrase vs. Unrelated)
- Task 3: Unrelated Text Rejection (Plagiarism [Exact + Paraphrase] vs. Unrelated)
- Task 4: Combined Multi-Category Distribution Analysis
"""
import difflib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from plagiarism.webis_loader import WebisRecord

# Category Constants
EXACT_COPYING = "exact_copying"
GENUINE_PARAPHRASING = "genuine_paraphrasing"
UNRELATED_TEXT = "unrelated_text"
AMBIGUOUS_REVIEW = "ambiguous_review"

SPAM_KEYWORDS = [
    "please help", "thank you", "thankyou", "sir / madam",
    "sorry", "asdf", "test submission", "n/a", "don't know"
]


def classify_webis_record(record: WebisRecord) -> Dict[str, Any]:
    """
    Transparent, rule-based classification of a Webis record into tripartite categories.
    Preserves original label strictly while storing categorization metadata.

    Returns:
        dict with:
            - category: one of EXACT_COPYING, GENUINE_PARAPHRASING, UNRELATED_TEXT, AMBIGUOUS_REVIEW
            - manual_review_required: bool
            - is_inferred: bool
            - rule_name: str
            - rule_description: str
            - sequence_ratio: float
            - normalized_ratio: float
            - original_label: int (1 or 0)
    """
    orig = record.original_text.strip()
    para = record.paraphrase_text.strip()
    orig_label = record.label  # 1 (Paraphrase: Yes) or 0 (Paraphrase: No)

    norm_orig = re.sub(r"\s+", " ", orig.lower())
    norm_para = re.sub(r"\s+", " ", para.lower())

    seq_ratio = round(float(difflib.SequenceMatcher(None, orig, para).ratio()), 4)
    norm_ratio = round(float(difflib.SequenceMatcher(None, norm_orig, norm_para).ratio()), 4)

    # Word-level overlap (Jaccard similarity on lower-case tokens)
    w_orig = set(re.findall(r"\b\w+\b", norm_orig))
    w_para = set(re.findall(r"\b\w+\b", norm_para))
    word_overlap = round(float(len(w_orig & w_para) / max(len(w_orig | w_para), 1)), 4)

    is_spam = any(k in norm_para for k in SPAM_KEYWORDS) or len(norm_para) < 12

    # Rule 1: Exact or Near-Verbatim Copying (Ratio >= 0.85 or identical normalized text)
    if norm_orig == norm_para or norm_ratio >= 0.85:
        return {
            "category": EXACT_COPYING,
            "manual_review_required": False,
            "is_inferred": True,
            "rule_name": "RULE_NEAR_VERBATIM",
            "rule_description": f"Near-verbatim or identical text match (norm_ratio={norm_ratio:.3f} >= 0.85).",
            "sequence_ratio": seq_ratio,
            "normalized_ratio": norm_ratio,
            "word_overlap": word_overlap,
            "original_label": orig_label,
        }

    # Rule 2: Unrelated Text (Original label 0 and (spam/empty, word_overlap < 0.05, or norm_ratio < 0.22))
    if orig_label == 0 and (is_spam or word_overlap < 0.05 or norm_ratio < 0.22):
        return {
            "category": UNRELATED_TEXT,
            "manual_review_required": False,
            "is_inferred": True,
            "rule_name": "RULE_UNRELATED_SPAM",
            "rule_description": f"Rejected submission with no substantive semantic overlap (word_overlap={word_overlap:.3f}, ratio={norm_ratio:.3f}, is_spam={is_spam}).",
            "sequence_ratio": seq_ratio,
            "normalized_ratio": norm_ratio,
            "word_overlap": word_overlap,
            "original_label": orig_label,
        }

    # Rule 3: Genuine Paraphrase (Original label 1 and substantive rewording: ratio < 0.85 and (word_overlap >= 0.05 or ratio >= 0.15))
    if orig_label == 1:
        if word_overlap >= 0.05 or norm_ratio >= 0.15:
            return {
                "category": GENUINE_PARAPHRASING,
                "manual_review_required": False,
                "is_inferred": False,  # Human MTurk annotator verified the paraphrase
                "rule_name": "RULE_VERIFIED_PARAPHRASE",
                "rule_description": f"Human-annotated valid paraphrase with creative rewording (word_overlap={word_overlap:.3f}, norm_ratio={norm_ratio:.3f}).",
                "sequence_ratio": seq_ratio,
                "normalized_ratio": norm_ratio,
                "word_overlap": word_overlap,
                "original_label": orig_label,
            }
        else:
            # Positive label with extreme low overlap indicates source alignment noise in Webis corpus
            return {
                "category": AMBIGUOUS_REVIEW,
                "manual_review_required": True,
                "is_inferred": True,
                "rule_name": "RULE_LOW_RATIO_POSITIVE",
                "rule_description": f"Annotated as Paraphrase:Yes but possesses extreme low lexical overlap (word_overlap={word_overlap:.3f}, ratio={norm_ratio:.3f}); suspected corpus alignment noise.",
                "sequence_ratio": seq_ratio,
                "normalized_ratio": norm_ratio,
                "word_overlap": word_overlap,
                "original_label": orig_label,
            }

    # Rule 4: Ambiguous Negative (Original label 0 with moderate overlap: word_overlap >= 0.05 and norm_ratio < 0.85)
    # The annotator marked 'Paraphrase: No', but there is partial text overlap.
    # Requires human inspection to separate partial copies from poor paraphrases.
    return {
        "category": AMBIGUOUS_REVIEW,
        "manual_review_required": True,
        "is_inferred": True,
        "rule_name": "RULE_PARTIAL_OVERLAP_NEGATIVE",
        "rule_description": f"Negative submission with partial lexical overlap (word_overlap={word_overlap:.3f}, ratio={norm_ratio:.3f}); requires manual verification.",
        "sequence_ratio": seq_ratio,
        "normalized_ratio": norm_ratio,
        "word_overlap": word_overlap,
        "original_label": orig_label,
    }


def categorize_dataset(records: List[WebisRecord]) -> List[Dict[str, Any]]:
    """Classify an entire dataset of Webis records and return per-record metadata."""
    return [classify_webis_record(r) for r in records]


def compute_binary_task_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Calculate binary evaluation metrics for a specific benchmark task."""
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / max(fp + tn, 1)
    fdr = fp / max(fp + tp, 1)

    try:
        raw_roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        raw_roc_auc = 0.5

    try:
        p_vals, r_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(r_vals, p_vals)
    except Exception:
        pr_auc = 0.5

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "raw_roc_auc": round(float(raw_roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "false_discovery_rate": round(float(fdr), 4),
        "decision_threshold": round(float(threshold), 3),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "sample_size": len(y_true),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
    }


def evaluate_tripartite_benchmarks(
    model_name: str,
    test_scores: np.ndarray,
    test_records: List[WebisRecord],
    classifications: List[Dict[str, Any]],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Run the four distinct benchmark tasks for a given similarity model.

    Tasks:
    1. Task 1: Exact Copying Detection (Exact Copying vs. Unrelated)
    2. Task 2: Genuine Paraphrase Detection (Genuine Paraphrase vs. Unrelated)
    3. Task 3: Unrelated Text Rejection (Plagiarism [Exact + Paraphrase] vs. Unrelated)
    4. Task 4: Combined Multi-Category Analysis
    """
    assert len(test_scores) == len(test_records) == len(classifications)

    cats = [c["category"] for c in classifications]

    # Task 1: Exact Copying vs. Unrelated Text
    idx_t1 = [i for i, c in enumerate(cats) if c in (EXACT_COPYING, UNRELATED_TEXT)]
    if idx_t1 and sum(1 for i in idx_t1 if cats[i] == EXACT_COPYING) > 0 and sum(1 for i in idx_t1 if cats[i] == UNRELATED_TEXT) > 0:
        y_true_t1 = np.array([1 if cats[i] == EXACT_COPYING else 0 for i in idx_t1])
        scores_t1 = test_scores[idx_t1]
        task1_metrics = compute_binary_task_metrics(y_true_t1, scores_t1, threshold=threshold)
    else:
        task1_metrics = {"error": "Insufficient class representation in Task 1"}

    # Task 2: Genuine Paraphrase vs. Unrelated Text
    idx_t2 = [i for i, c in enumerate(cats) if c in (GENUINE_PARAPHRASING, UNRELATED_TEXT)]
    if idx_t2 and sum(1 for i in idx_t2 if cats[i] == GENUINE_PARAPHRASING) > 0 and sum(1 for i in idx_t2 if cats[i] == UNRELATED_TEXT) > 0:
        y_true_t2 = np.array([1 if cats[i] == GENUINE_PARAPHRASING else 0 for i in idx_t2])
        scores_t2 = test_scores[idx_t2]
        task2_metrics = compute_binary_task_metrics(y_true_t2, scores_t2, threshold=threshold)
    else:
        task2_metrics = {"error": "Insufficient class representation in Task 2"}

    # Task 3: Plagiarism (Exact + Paraphrase) vs. Unrelated Text
    idx_t3 = [i for i, c in enumerate(cats) if c in (EXACT_COPYING, GENUINE_PARAPHRASING, UNRELATED_TEXT)]
    if idx_t3 and sum(1 for i in idx_t3 if cats[i] in (EXACT_COPYING, GENUINE_PARAPHRASING)) > 0 and sum(1 for i in idx_t3 if cats[i] == UNRELATED_TEXT) > 0:
        y_true_t3 = np.array([1 if cats[i] in (EXACT_COPYING, GENUINE_PARAPHRASING) else 0 for i in idx_t3])
        scores_t3 = test_scores[idx_t3]
        task3_metrics = compute_binary_task_metrics(y_true_t3, scores_t3, threshold=threshold)
    else:
        task3_metrics = {"error": "Insufficient class representation in Task 3"}

    # Task 4: Combined Multi-Category Score Distributions
    category_score_stats = {}
    for cat_name in [EXACT_COPYING, GENUINE_PARAPHRASING, UNRELATED_TEXT, AMBIGUOUS_REVIEW]:
        sub_scores = [test_scores[i] for i, c in enumerate(cats) if c == cat_name]
        if sub_scores:
            arr = np.array(sub_scores)
            category_score_stats[cat_name] = {
                "count": len(sub_scores),
                "mean_score": round(float(np.mean(arr)), 4),
                "median_score": round(float(np.median(arr)), 4),
                "std_score": round(float(np.std(arr)), 4),
                "min_score": round(float(np.min(arr)), 4),
                "max_score": round(float(np.max(arr)), 4),
            }
        else:
            category_score_stats[cat_name] = {"count": 0}

    return {
        "model_name": model_name,
        "task1_exact_copying": task1_metrics,
        "task2_genuine_paraphrase": task2_metrics,
        "task3_unrelated_rejection": task3_metrics,
        "category_score_stats": category_score_stats,
    }


def save_tripartite_outputs(
    results: Dict[str, Any],
    output_dir: str = "outputs/evaluation/plagiarism"
) -> Tuple[str, str]:
    """
    Save tripartite evaluation results to separate JSON and CSV files.
    Preserves original benchmark files untouched.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "tripartite_evaluation_summary.json")
    csv_path = os.path.join(output_dir, "tripartite_experiment_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Compile comparative CSV across models and tasks
    rows = []
    headers = [
        "model", "task", "sample_size", "accuracy", "precision", "recall",
        "f1_score", "raw_roc_auc", "pr_auc", "fpr", "threshold"
    ]
    for m_key, m_data in results.get("models", {}).items():
        m_name = m_data.get("model_name", m_key)
        for task_key, task_label in [
            ("task1_exact_copying", "Exact Copying Detection"),
            ("task2_genuine_paraphrase", "Genuine Paraphrase Detection"),
            ("task3_unrelated_rejection", "Unrelated Text Rejection")
        ]:
            t = m_data.get(task_key, {})
            if "error" not in t and "accuracy" in t:
                rows.append([
                    m_name, task_label, t["sample_size"], t["accuracy"],
                    t["precision"], t["recall"], t["f1_score"],
                    t["raw_roc_auc"], t["pr_auc"], t["false_positive_rate"],
                    t["decision_threshold"]
                ])

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    return json_path, csv_path
