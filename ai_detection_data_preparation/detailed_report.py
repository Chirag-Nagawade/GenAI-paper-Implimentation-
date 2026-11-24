import os, sys, json, traceback
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Paths
PROJECT_ROOT = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')
DATA_DIR = PROJECT_ROOT / 'ai_detection_data_preparation'
TRAIN_PATH = DATA_DIR / 'train_features.npz'
VAL_PATH = DATA_DIR / 'validation_features.npz'

# Load data
train = np.load(TRAIN_PATH)
val = np.load(VAL_PATH)
X_train_full = train['X']
y_train = train['y']
X_val_full = val['X']
y_val = val['y']

# Feature splits
X_train_emb = X_train_full[:, :384]
X_train_sty = X_train_full[:, 384:]
X_val_emb = X_val_full[:, :384]
X_val_sty = X_val_full[:, 384:]

experiments = {
    'embeddings': (X_train_emb, X_val_emb),
    'stylometric': (X_train_sty, X_val_sty),
    'combined': (X_train_full, X_val_full),
}

# Check XGBoost availability
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    print('XGBoost not available:', e)
    HAS_XGB = False

results = {}

for clf_name, clf_cls in [('LogisticRegression', LogisticRegression), ('XGBoost', None)]:
    for feat_name, (X_tr, X_va) in experiments.items():
        key = f"{clf_name}_{feat_name}"
        try:
            scaler = StandardScaler()
            if clf_name == 'LogisticRegression':
                clf = clf_cls(max_iter=1000, class_weight='balanced', n_jobs=-1)
            else:
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
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            cm = confusion_matrix(y_val, y_pred)
            report = classification_report(y_val, y_pred, target_names=['human','ai'], zero_division=0)
            results[key] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
            }
        except Exception as exc:
            results[key] = {'error': str(exc)}
            print(f'Error in {key}:', exc)

# Print detailed report
print('\n=== Detailed Experiment Results ===')
for key, res in results.items():
    print(f"\n--- {key} ---")
    if 'error' in res:
        print('Error:', res['error'])
        continue
    print(f"Accuracy : {res['accuracy']:.4f}")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall   : {res['recall']:.4f}")
    print(f"F1-score : {res['f1']:.4f}")
    print('Confusion Matrix:')
    for row in res['confusion_matrix']:
        print(' ', row)
    print('Classification Report:')
    print(res['classification_report'])

# Identify best model by F1
best_key = max(results, key=lambda k: results[k].get('f1', -1) if 'error' not in results[k] else -1)
print('\nBest experiment:', best_key, 'with F1 =', results[best_key]['f1'])

# Save best model again for reference
best_pipeline_path = DATA_DIR / 'models' / 'best_pipeline.joblib'
# (model already saved previously, just confirm existence)
print('Best pipeline path (should already exist):', best_pipeline_path)

sys.exit(0)
