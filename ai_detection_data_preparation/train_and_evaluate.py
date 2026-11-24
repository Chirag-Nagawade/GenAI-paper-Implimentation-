import os, sys, json, pathlib, traceback
import numpy as np
import joblib
from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')
DATA_DIR = PROJECT_ROOT / 'ai_detection_data_preparation'
TRAIN_PATH = DATA_DIR / 'train_features.npz'
VAL_PATH = DATA_DIR / 'validation_features.npz'

# ---- Load data ----
if not TRAIN_PATH.is_file() or not VAL_PATH.is_file():
    print(f"Missing feature files: {TRAIN_PATH} or {VAL_PATH}")
    sys.exit(1)

train = np.load(TRAIN_PATH)
val = np.load(VAL_PATH)
X_train_full = train['X']  # (N, 392)
y_train = train['y']
X_val_full = val['X']
y_val = val['y']

# ---- Feature slices ----
# MiniLM embeddings are first 384 columns, stylometric are last 8
X_train_emb = X_train_full[:, :384]
X_train_sty = X_train_full[:, 384:]
X_val_emb = X_val_full[:, :384]
X_val_sty = X_val_full[:, 384:]

# ---- Define experiments ----
experiments = {
    'embeddings': (X_train_emb, X_val_emb),
    'stylometric': (X_train_sty, X_val_sty),
    'combined': (X_train_full, X_val_full),
}

# ---- Helper functions ----
def evaluate(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['human', 'ai'], zero_division=0)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }

# ---- Train and evaluate classifiers ----
results = {}
best_f1 = -1
best_key = None
best_model = None
best_pipeline = None

# Check if XGBoost is available
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    print('XGBoost not available:', e)
    HAS_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for clf_name, clf_cls in [('LogisticRegression', LogisticRegression), ('XGBoost', None)]:
    for feat_name, (X_tr, X_va) in experiments.items():
        key = f"{clf_name}_{feat_name}"
        try:
            # Build pipeline: scaling -> classifier
            scaler = StandardScaler()
            if clf_name == 'LogisticRegression':
                clf = clf_cls(max_iter=1000, class_weight='balanced', n_jobs=-1)
            else:  # XGBoost
                if not HAS_XGB:
                    raise RuntimeError('XGBoost not installed')
                clf = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=42,
                )
            pipeline = Pipeline([('scaler', scaler), ('clf', clf)])
            pipeline.fit(X_tr, y_train)
            y_pred = pipeline.predict(X_va)
            metrics = evaluate(y_val, y_pred)
            results[key] = metrics
            # Track best by F1
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_key = key
                best_model = clf
                best_pipeline = pipeline
        except Exception as exc:
            err_msg = f"Error in {key}: {exc}\n{traceback.format_exc()}"
            results[key] = {'error': err_msg}
            print(err_msg)

# ---- Save best model and pipeline ----
if best_pipeline is not None:
    model_dir = DATA_DIR / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = model_dir / 'best_pipeline.joblib'
    joblib.dump(best_pipeline, pipeline_path)
    # Save metadata (feature config and label mapping)
    meta = {
        'selected_experiment': best_key,
        'feature_set': best_key.split('_')[1],
        'classifier': best_key.split('_')[0],
        'label_mapping': {0: 'human', 1: 'ai'},
    }
    meta_path = model_dir / 'model_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Best model saved to {pipeline_path}\nMetadata saved to {meta_path}")
else:
    print('No valid model was trained.')

# ---- Print summary table ----
print('\n=== Summary of Experiments ===')
header = ['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1']
print('\t'.join(header))
for key, metrics in results.items():
    if 'error' in metrics:
        print(f"{key}\tERROR")
    else:
        print(f"{key}\t{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1']:.4f}")

print('\nBest experiment:', best_key, 'with F1 =', best_f1)

sys.exit(0)
