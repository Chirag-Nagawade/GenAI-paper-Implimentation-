import csv, json, sys, traceback
import numpy as np
from pathlib import Path
from collections import Counter
import re

# ---------- Paths ----------
PROJECT_ROOT = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')
DATA_DIR = PROJECT_ROOT / 'ai_detection_data_preparation'
TRAIN_FEATURES = DATA_DIR / 'train_features.npz'
VAL_FEATURES = DATA_DIR / 'validation_features.npz'
TRAIN_MANIFEST = DATA_DIR / 'clean_train_manifest.csv'
VAL_MANIFEST = DATA_DIR / 'clean_validation_manifest.csv'
MODEL_DIR = DATA_DIR / 'models'
BEST_PIPELINE = MODEL_DIR / 'best_pipeline.joblib'
METADATA_JSON = MODEL_DIR / 'model_metadata.json'

# ---------- Helper functions ----------
def load_manifest(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def label_counts(labels):
    uniq, counts = np.unique(labels, return_counts=True)
    return dict(zip(uniq, counts))

# ---------- 1. Verify data ----------
print('\n' + '='*20 + ' 1. Data verification ' + '='*20 + '\n')
if not TRAIN_FEATURES.is_file() or not VAL_FEATURES.is_file():
    print('Feature files missing')
    sys.exit(1)
train = np.load(TRAIN_FEATURES)
val = np.load(VAL_FEATURES)
X_train = train['X']
y_train = train['y']
ids_train = train['ids']
X_val = val['X']
y_val = val['y']
ids_val = val['ids']

print('Training matrix shape:', X_train.shape)
print('Validation matrix shape:', X_val.shape)
print('Training label distribution:', label_counts(y_train))
print('Validation label distribution:', label_counts(y_val))
print('Unique training IDs:', len(set(ids_train)))
print('Unique validation IDs:', len(set(ids_val)))

train_manifest = load_manifest(TRAIN_MANIFEST)
val_manifest = load_manifest(VAL_MANIFEST)
print('Training manifest rows:', len(train_manifest))
print('Validation manifest rows:', len(val_manifest))

ids_from_manifest_train = {r['filepath'] for r in train_manifest}
ids_from_manifest_val = {r['filepath'] for r in val_manifest}
print('IDs match train manifest vs npz:', ids_from_manifest_train == set(ids_train))
print('IDs match validation manifest vs npz:', ids_from_manifest_val == set(ids_val))

sha_train = {r['sha256'] for r in train_manifest}
sha_val = {r['sha256'] for r in val_manifest}
overlap_sha = sha_train.intersection(sha_val)
print('Overlapping SHA-256 hashes between splits:', len(overlap_sha))

overlap_ids = ids_from_manifest_train.intersection(ids_from_manifest_val)
print('Overlapping document IDs between splits:', len(overlap_ids))

# ---------- 2. Reproduce experiments ----------
print('\n' + '='*20 + ' 2. Reproducing experiments ' + '='*20 + '\n')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# feature slices
X_train_emb = X_train[:, :384]
X_train_sty = X_train[:, 384:]
X_val_emb = X_val[:, :384]
X_val_sty = X_val[:, 384:]

experiments = {
    'embeddings': (X_train_emb, X_val_emb),
    'stylometric': (X_train_sty, X_val_sty),
    'combined': (X_train, X_val),
}

# check XGBoost availability
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
            pipe = Pipeline([('scaler', scaler), ('clf', clf)])
            pipe.fit(X_tr, y_train)
            y_pred = pipe.predict(X_va)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            cm = confusion_matrix(y_val, y_pred)
            results[key] = {
                'train_samples': X_tr.shape[0],
                'val_samples': X_va.shape[0],
                'predictions': int(len(y_pred)),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'confusion_matrix': cm.tolist(),
            }
        except Exception as exc:
            results[key] = {'error': str(exc)}
            print(f'Error in {key}:', exc)

for k, r in results.items():
    print(f"\n--- {k} ---")
    if 'error' in r:
        print('Error:', r['error'])
        continue
    print(f"Train samples: {r['train_samples']}")
    print(f"Validation samples: {r['val_samples']}")
    print(f"Predictions made: {r['predictions']}")
    print(f"Accuracy : {r['accuracy']:.4f}")
    print(f"Precision: {r['precision']:.4f}")
    print(f"Recall   : {r['recall']:.4f}")
    print(f"F1-score : {r['f1']:.4f}")
    print('Confusion matrix:')
    for row in r['confusion_matrix']:
        print(' ', row)

# ---------- 3. Quick dataset bias inspection ----------
print('\n' + '='*20 + ' 3. Dataset bias inspection (quick stats) ' + '='*20 + '\n')
# load raw texts for a sample of documents from training set
texts = []
labels = []
for r in train_manifest:
    filepath = PROJECT_ROOT / r['filepath']
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            txt = f.read()
    except Exception:
        txt = ''
    texts.append(txt)
    labels.append(0 if r['label'] == 'human' else 1)
texts = np.array(texts)
labels = np.array(labels)

for cls, name in [(0, 'human'), (1, 'ai')]:
    idx = np.where(labels == cls)[0]
    if len(idx) == 0:
        continue
    chars = np.array([len(texts[i]) for i in idx])
    words = np.array([len(re.findall(r"\\b\\w+\\b", texts[i])) for i in idx])
    print(f"Class {name}: avg chars={chars.mean():.1f}, std={chars.std():.1f}, avg words={words.mean():.1f}, std={words.std():.1f}")
    # top tokens
    token_counter = Counter()
    for i in idx:
        token_counter.update(re.findall(r"\\b\\w+\\b", texts[i].lower()))
    most_common = token_counter.most_common(10)
    print(f"Top tokens for {name} (token,count): {most_common}")

# ---------- 4. Verify saved model ----------
print('\n' + '='*20 + ' 4. Saved model verification ' + '='*20 + '\n')
if not BEST_PIPELINE.is_file() or not METADATA_JSON.is_file():
    print('Saved model files missing')
    sys.exit(1)
import joblib
pipeline = joblib.load(BEST_PIPELINE)
with open(METADATA_JSON, 'r', encoding='utf-8') as f:
    meta = json.load(f)
print('Model metadata:', json.dumps(meta, indent=2))
feature_set = meta.get('feature_set')
print('Feature set recorded in metadata:', feature_set)
# Predict on validation data using the appropriate feature slice
if feature_set == 'embeddings':
    X_val_used = X_val_emb
elif feature_set == 'stylometric':
    X_val_used = X_val_sty
else:
    X_val_used = X_val
preds = pipeline.predict(X_val_used)
print('Loaded pipeline predictions count:', len(preds))
# Compute metrics for the loaded pipeline
acc = accuracy_score(y_val, preds)
prec = precision_score(y_val, preds, pos_label=1, zero_division=0)
rec = recall_score(y_val, preds, pos_label=1, zero_division=0)
f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
print(f"Metrics from loaded pipeline -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# ---------- Write verification report ----------
report_path = PROJECT_ROOT / 'phase_3_2_verification_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('Phase 3.2 Verification Report\n')
    f.write('Generated on: ' + __import__('datetime').datetime.now().isoformat() + '\n\n')
    f.write('--- Summary of experiment metrics ---\n')
    for k, r in results.items():
        if 'error' in r:
            f.write(f"{k}: ERROR - {r['error']}\n")
        else:
            f.write(f"{k}: Acc={r['accuracy']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}, F1={r['f1']:.4f}, Conf={r['confusion_matrix']}\n")
    f.write('\n--- Manifest vs NPZ ID checks ---\n')
    f.write(f"Train IDs match manifest: {ids_from_manifest_train == set(ids_train)}\n")
    f.write(f"Validation IDs match manifest: {ids_from_manifest_val == set(ids_val)}\n")
    f.write(f"SHA-256 overlap count: {len(overlap_sha)}\n")
    f.write(f"Document ID overlap count: {len(overlap_ids)}\n")
    f.write('\n--- Model loading verification ---\n')
    f.write('Metadata:\n' + json.dumps(meta, indent=2) + '\n')
    f.write(f"Loaded pipeline metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n")

print('Verification report written to', report_path)

sys.exit(0)
