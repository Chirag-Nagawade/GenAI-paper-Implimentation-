"""
Training: paragraph-level ensemble (stylometric GBM + lexical n-gram LR [+ optional perplexity]).

- Trains ONLY on verified paragraph labels (no weak sentence labels).
- Document-grouped CV (no leakage between paragraphs of the same document).
- Blend weight and decision thresholds are chosen from out-of-fold predictions.
- Reports accuracy / false-positive rate per text length, so you can see where it fails.
"""
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # smoke-test fallback only
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGB

    def XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, random_state=0, **_):
        return _HGB(max_iter=n_estimators, max_depth=max_depth,
                    learning_rate=learning_rate, random_state=random_state)

from ai_detect.dataset_loader import load_ai_dataset
from ai_detect.features import FEATURE_NAMES, LENGTH_FEATURE_IDX, extract_feature_matrix
from ai_detect.lexical import build_lexical_model

MODEL_COLS = [i for i in range(len(FEATURE_NAMES)) if i not in LENGTH_FEATURE_IDX]


def _style_clf(seed: int):
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, random_state=seed,
                         eval_metric="logloss")


def build_matrix(texts, use_ppl: bool):
    X = extract_feature_matrix(texts)[:, MODEL_COLS]
    if use_ppl:
        from ai_detect.perplexity import ppl_features
        X = np.hstack([X, ppl_features(texts)])
    return X


def _bucket_report(y, p, words, thr):
    rep = {}
    for name, m in [("<30 words", words < 30), ("30-80 words", (words >= 30) & (words <= 80)),
                    (">80 words", words > 80)]:
        if m.sum() < 5 or len(set(y[m])) < 2:
            continue
        pred = (p[m] >= thr).astype(int)
        neg = (y[m] == 0)
        rep[name] = {"n": int(m.sum()),
                     "accuracy": round(float(accuracy_score(y[m], pred)), 4),
                     "auc": round(float(roc_auc_score(y[m], p[m])), 4),
                     "false_positive_rate": round(float(pred[neg].mean()) if neg.any() else 0.0, 4)}
    return rep


_AI_WORDS = {"1", "ai", "chatgpt", "gpt", "machine", "generated", "llm", "true"}
_HUMAN_WORDS = {"0", "human", "person", "false"}


def load_extra_csv(path: str, max_rows: int = 6000) -> Tuple[List[str], List[int], List[str]]:
    """Load extra labelled data. CSV needs columns `text` and `label` (0/human, 1/ai/chatgpt);
    optional `group` column (rows sharing a group are never split across CV folds).
    Long texts are cut into the same paragraph-sized chunks used at prediction time."""
    from ai_detect.predict import _chunk_document
    texts, labels, groups = [], [], []
    base = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= max_rows:
                break
            raw = str(row.get("label", "")).strip().lower()
            lab = 1 if raw in _AI_WORDS else 0 if raw in _HUMAN_WORDS else None
            txt = (row.get("text") or "").strip()
            if lab is None or not txt:
                continue
            grp = f"{base}:{row.get('group') or i}"
            for chunk in _chunk_document(txt):
                t = " ".join(chunk)
                if len(t.split()) >= 20:
                    texts.append(t); labels.append(lab); groups.append(grp)
    return texts, labels, groups


def cross_source_check(tx_a, y_a, tx_b, y_b, use_ppl: bool, seed: int = 42) -> Dict[str, float]:
    """Train on source A, test on source B. This is the honest generalisation test."""
    ya, yb = np.array(y_a), np.array(y_b)
    Xa, Xb = build_matrix(tx_a, use_ppl), build_matrix(tx_b, use_ppl)
    sc = StandardScaler().fit(Xa)
    p_style = _style_clf(seed).fit(sc.transform(Xa), ya).predict_proba(sc.transform(Xb))[:, 1]
    p_lex = build_lexical_model(seed).fit(tx_a, ya).predict_proba(tx_b)[:, 1]
    return {"style_auc": round(float(roc_auc_score(yb, p_style)), 4),
            "lexical_auc": round(float(roc_auc_score(yb, p_lex)), 4),
            "blend_auc": round(float(roc_auc_score(yb, 0.5 * p_style + 0.5 * p_lex)), 4)}


def train_ai_detector(dataset_dir: str = "data/ChatGPT-Human-NewGenerated",
                      output_dir: str = "outputs/models/ai_detection",
                      n_splits: int = 5, random_state: int = 42,
                      use_perplexity: bool = False,
                      extra_csv: Optional[List[str]] = None,
                      target_fpr: float = 0.02) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    paragraphs, _sentences, verif = load_ai_dataset(dataset_dir, verbose=True)

    texts = [p.text for p in paragraphs]
    labels = [p.label for p in paragraphs]
    grp = [p.doc_id for p in paragraphs]
    source = ["pan"] * len(texts)
    cross = {}
    if extra_csv:
        ex_t, ex_y, ex_g = [], [], []
        for path in extra_csv:
            t_, y_, g_ = load_extra_csv(path)
            print(f"[AI Detection] Extra data {path}: {len(t_)} chunks (AI={sum(y_)}, human={len(y_) - sum(y_)})")
            ex_t += t_; ex_y += y_; ex_g += g_
        if ex_t and len(set(ex_y)) == 2:
            print("[AI Detection] Cross-source check (train on one source, test on the other)...")
            cross = {"pan_to_extra": cross_source_check(texts, labels, ex_t, ex_y, use_perplexity),
                     "extra_to_pan": cross_source_check(ex_t, ex_y, texts, labels, use_perplexity)}
            print("  ", json.dumps(cross))
            texts += ex_t; labels += ex_y; grp += ex_g; source += ["extra"] * len(ex_t)
    y = np.array(labels)
    groups = np.array(grp)
    source = np.array(source)
    words = np.array([len(t.split()) for t in texts])

    print("[AI Detection] Building features...")
    X = build_matrix(texts, use_perplexity)

    oof_style = np.zeros(len(y))
    oof_lex = np.zeros(len(y))
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), 1):
        sc = StandardScaler().fit(X[tr])
        clf = _style_clf(random_state + fold).fit(sc.transform(X[tr]), y[tr])
        oof_style[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]

        lex = build_lexical_model(random_state).fit([texts[i] for i in tr], y[tr])
        oof_lex[va] = lex.predict_proba([texts[i] for i in va])[:, 1]
        print(f"  fold {fold}: style AUC={roc_auc_score(y[va], oof_style[va]):.3f} "
              f"lexical AUC={roc_auc_score(y[va], oof_lex[va]):.3f}")

    # Choose blend weight from OOF predictions
    best_w, best_auc = 0.5, -1.0
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        auc = roc_auc_score(y, w * oof_style + (1 - w) * oof_lex)
        if auc > best_auc:
            best_w, best_auc = w, auc
    oof = best_w * oof_style + (1 - best_w) * oof_lex

    # Thresholds from OOF: AI cutoff at <=target_fpr false positives (default 2%); human cutoff keeps >=90% of AI above it
    fpr, tpr, thr = roc_curve(y, oof)
    ok = np.where(fpr <= target_fpr)[0]
    t_ai = float(thr[ok[-1]]) if len(ok) else 0.70
    ok_h = np.where(tpr >= 0.90)[0]
    t_hu = float(thr[ok_h[0]]) if len(ok_h) else 0.30
    t_ai = float(np.clip(t_ai, 0.50, 0.90))
    t_hu = float(np.clip(min(t_hu, t_ai - 0.10), 0.10, 0.45))

    cm = confusion_matrix(y, (oof >= t_ai).astype(int))
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "samples": int(len(y)), "ai": int(y.sum()), "human": int((y == 0).sum()),
        "auc_style_only": round(float(roc_auc_score(y, oof_style)), 4),
        "auc_lexical_only": round(float(roc_auc_score(y, oof_lex)), 4),
        "auc_blend": round(float(best_auc), 4), "blend_weight_style": best_w,
        "accuracy_at_ai_threshold": round(float(accuracy_score(y, oof >= t_ai)), 4),
        "f1_at_ai_threshold": round(float(f1_score(y, oof >= t_ai)), 4),
        "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        "recall_ai_at_ai_threshold": round(float(tp / max(tp + fn, 1)), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "by_length": _bucket_report(y, oof, words, t_ai),
        "cross_source": cross,
        "by_source": {src: {"n": int((source == src).sum()),
                            "auc": round(float(roc_auc_score(y[source == src], oof[source == src])), 4)}
                      for src in np.unique(source)
                      if len(set(y[source == src])) == 2},
    }

    # Final models on all data, calibrated
    scaler = StandardScaler().fit(X)
    style = CalibratedClassifierCV(_style_clf(random_state), method="isotonic", cv=3)
    style.fit(scaler.transform(X), y)
    lex = CalibratedClassifierCV(build_lexical_model(random_state), method="sigmoid", cv=3)
    lex.fit(texts, y)

    bundle = {"style": style, "scaler": scaler, "lexical": lex, "w_style": best_w,
              "model_cols": MODEL_COLS, "use_perplexity": use_perplexity}
    joblib.dump(bundle, os.path.join(output_dir, "detector_bundle.joblib"))

    results = {"dataset_verification": verif, "paragraph_evaluation": metrics,
               "thresholds": {"human_threshold": t_hu, "ai_threshold": t_ai,
                              "categories": ["Likely Human", "Needs Review", "Potentially AI-generated"]},
               "features": FEATURE_NAMES, "use_perplexity": use_perplexity,
               "training_time_seconds": round(time.time() - t0, 2)}
    with open(os.path.join(output_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"AUC style={metrics['auc_style_only']}  lexical={metrics['auc_lexical_only']}  "
          f"blend={metrics['auc_blend']} (w_style={best_w})")
    print(f"Thresholds: AI>={t_ai:.3f}  Human<={t_hu:.3f}  (target FPR {target_fpr})")
    print(f"Accuracy@AI-threshold={metrics['accuracy_at_ai_threshold']}  FPR={metrics['false_positive_rate']}  "
          f"AI recall={metrics['recall_ai_at_ai_threshold']}")
    print("By length:", json.dumps(metrics["by_length"], indent=2))
    print("By source:", json.dumps(metrics["by_source"]))
    if cross:
        print("Cross-source:", json.dumps(cross))
    print("=" * 70)
    return results


if __name__ == "__main__":
    train_ai_detector()
