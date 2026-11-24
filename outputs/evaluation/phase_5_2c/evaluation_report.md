# Independent Evaluation of Deployed AI Detector (Phase 5.2C)

**Dataset**: `ai_detection_hc3_clean/clean_test_manifest.csv`
- Total records: 2611
- Human (true): 1072
- AI (true): 1539

**Model hashes (before evaluation)**
- detector_bundle.joblib: `2045016b69aaf705d1b5c65e1d175eb7f2f79d2911843f564b7f52b84b12708a`
- model_metadata.json: `ea186061bf189f95209a5ebf53d736775460c7b37fd94dd912450462be1ef47f`

## Prediction Summary
- Human predictions: 588 (22.52%)
- AI predictions: 1943 (74.42%)
- Review predictions: 80 (3.06%)
- **Coverage** (definite Human or AI): 96.94%

## Metrics on Definite Predictions
| Metric | Human | AI |
|--------|-------|----|
| Precision | 0.957 | 0.774 |
| Recall    | 0.561 | 0.984 |
| F1        | 0.708 | 0.866 |

- Macro F1: 0.787
- Weighted F1: 0.803
- ROC‑AUC (continuous scores): 0.830

## Confusion Matrix (Definite predictions only)
```
[[ 563  440]
 [  25 1503]]
```
Rows = true Human / AI, Columns = predicted Human / AI.

**Methodology notes**
- Used `predict_document` to obtain a word‑weighted mean AI probability per document.
- Applied deployed thresholds (`hu ≤ 0.35`, `ai ≥ 0.52`) to derive definite predictions; otherwise labelled as *Review*.
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
print('model', sha256(r'C:/Users/Chirag/Desktop/MIT AOE/5th Sem/PlagiarismAI-main\outputs\models\ai_detection\detector_bundle.joblib'))
print('metadata', sha256(r'C:/Users/Chirag/Desktop/MIT AOE/5th Sem/PlagiarismAI-main\outputs\models\ai_detection\model_metadata.json'))
```
These should match the hashes reported above, confirming the model was unchanged.

**Limitations**
- Review predictions are excluded from precision/recall/F1 as requested.
- Class imbalance (≈41 % Human, 59 % AI) may affect overall accuracy; consider reporting balanced metrics in future work.

**Artifacts created**
- Raw predictions CSV: `C:/Users/Chirag/Desktop/MIT AOE/5th Sem/PlagiarismAI-main\outputs\evaluation\phase_5_2c\raw_predictions.csv`
- Evaluation markdown report (this file).
