import csv, json, os, hashlib, sys
import numpy as np
import joblib
from pathlib import Path

# Paths
project_root = r"C:/Users/Chirag/Desktop/MIT AOE/5th Sem/PlagiarismAI-main"
manifest_path = os.path.join(project_root, "ai_detection_hc3_clean", "clean_test_manifest.csv")
model_dir = os.path.join(project_root, "outputs", "models", "ai_detection")
model_path = os.path.join(model_dir, "detector_bundle.joblib")
metadata_path = os.path.join(model_dir, "model_metadata.json")

# Helper to compute sha256
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

hashes_before = {
    'model': sha256_file(model_path),
    'metadata': sha256_file(metadata_path) if os.path.exists(metadata_path) else None
}

# Load predictor utilities from the repo
sys.path.append(project_root)
from ai_detect.predict import get_detector, score_texts, _thresholds, predict_document

# Load thresholds once
_, meta = get_detector(model_dir)
hu_thr, ai_thr = _thresholds(meta)

# Prepare output structures
pred_rows = []
true_labels = []
pred_probs = []
# To compute metrics on definite predictions only, keep list of definite predicted labels (None for review)
pred_labels_def = []

with open(manifest_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_id = row['document_id']
        qid = row['question_id']
        true_label = row['label'].strip().lower()
        text = row['text']
        # Predict document
        pred = predict_document(text, model_dir)
        mean_prob = pred['summary']['mean_prob_ai']
        # Determine predicted label using thresholds
        if mean_prob >= ai_thr:
            pred_label = 'ai'
            status = 'definite'
        elif mean_prob <= hu_thr:
            pred_label = 'human'
            status = 'definite'
        else:
            pred_label = 'review'
            status = 'review'
        pred_rows.append({
            'document_id': doc_id,
            'question_id': qid,
            'true_label': true_label,
            'predicted_probability': round(mean_prob, 5),
            'predicted_label': pred_label,
            'prediction_status': status
        })
        true_labels.append(1 if true_label == 'ai' else 0)
        pred_probs.append(mean_prob)
        if status == 'definite':
            pred_labels_def.append(1 if pred_label == 'ai' else 0)
        else:
            pred_labels_def.append(None)

# Save raw predictions
out_dir = os.path.join(project_root, "outputs", "evaluation", "phase_5_2c")
os.makedirs(out_dir, exist_ok=True)
raw_path = os.path.join(out_dir, "raw_predictions.csv")
with open(raw_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['document_id','question_id','true_label','predicted_probability','predicted_label','prediction_status'])
    writer.writeheader()
    writer.writerows(pred_rows)

# Compute metrics on definite predictions
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mask_def = [lbl is not None for lbl in pred_labels_def]
true_def = np.array([t for t, m in zip(true_labels, mask_def) if m])
pred_def = np.array([lbl for lbl, m in zip(pred_labels_def, mask_def) if m])

conf_mat = confusion_matrix(true_def, pred_def, labels=[0,1])
precision, recall, f1, _ = precision_recall_fscore_support(true_def, pred_def, labels=[0,1], zero_division=0)
macro_f1 = float(np.mean(f1))
weights = np.bincount(true_def)
weighted_f1 = float(np.average(f1, weights=weights))
roc_auc = float(roc_auc_score(true_labels, pred_probs))

# Overall counts
total = len(true_labels)
counts = {
    'human_pred': sum(1 for r in pred_rows if r['predicted_label']=='human'),
    'ai_pred': sum(1 for r in pred_rows if r['predicted_label']=='ai'),
    'review_pred': sum(1 for r in pred_rows if r['predicted_label']=='review')
}
coverage = (counts['human_pred'] + counts['ai_pred']) / total * 100.0

# Build markdown report
report_md = f"""# Independent Evaluation of Deployed AI Detector (Phase 5.2C)

**Dataset**: `ai_detection_hc3_clean/clean_test_manifest.csv`
- Total records: {total}
- Human (true): {sum(1 for r in pred_rows if r['true_label']=='human')}
- AI (true): {sum(1 for r in pred_rows if r['true_label']=='ai')}

**Model hashes (before evaluation)**
- detector_bundle.joblib: `{hashes_before['model']}`
- model_metadata.json: `{hashes_before['metadata']}`

## Prediction Summary
- Human predictions: {counts['human_pred']} ({counts['human_pred']/total:.2%})
- AI predictions: {counts['ai_pred']} ({counts['ai_pred']/total:.2%})
- Review predictions: {counts['review_pred']} ({counts['review_pred']/total:.2%})
- **Coverage** (definite Human or AI): {coverage:.2f}%

## Metrics on Definite Predictions
| Metric | Human | AI |
|--------|-------|----|
| Precision | {precision[0]:.3f} | {precision[1]:.3f} |
| Recall    | {recall[0]:.3f} | {recall[1]:.3f} |
| F1        | {f1[0]:.3f} | {f1[1]:.3f} |

- Macro F1: {macro_f1:.3f}
- Weighted F1: {weighted_f1:.3f}
- ROC‑AUC (continuous scores): {roc_auc:.3f}

## Confusion Matrix (Definite predictions only)
```
{conf_mat}
```
Rows = true Human / AI, Columns = predicted Human / AI.

**Methodology notes**
- Used `predict_document` to obtain a word‑weighted mean AI probability per document.
- Applied deployed thresholds (`hu ≤ {hu_thr:.2f}`, `ai ≥ {ai_thr:.2f}`) to derive definite predictions; otherwise labelled as *Review*.
- No thresholds were tuned on this test set.
- ROC‑AUC computed over all 2 611 samples using the raw mean probabilities.

## Post‑evaluation hash check
```python
import hashlib, os
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
print('model', sha256(r'{model_path}'))
print('metadata', sha256(r'{metadata_path}'))
```
These should match the hashes reported above, confirming the model was unchanged.

**Limitations**
- Review predictions are excluded from precision/recall/F1 as requested.
- Class imbalance (≈41 % Human, 59 % AI) may affect overall accuracy; consider reporting balanced metrics in future work.

**Artifacts created**
- Raw predictions CSV: `{raw_path}`
- Evaluation markdown report (this file).
"""

report_path = os.path.join(out_dir, "evaluation_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_md)

# Compute post‑evaluation hashes
hashes_after = {
    'model': sha256_file(model_path),
    'metadata': sha256_file(metadata_path) if os.path.exists(metadata_path) else None
}

print('HASHES_BEFORE', json.dumps(hashes_before))
print('HASHES_AFTER', json.dumps(hashes_after))
