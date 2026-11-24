import os
import sys
import csv
import hashlib
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

# Dependencies
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print('scikit-learn not available', file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print('sentence_transformers not available', file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------
# Paths (project‑root relative)
PROJECT_ROOT = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')
DATA_ROOT = PROJECT_ROOT / 'data' / 'ChatGPT-Human-NewGenerated'
MODEL_PATH = PROJECT_ROOT / 'models' / 'all-MiniLM-L6-v2'
OUTPUT_DIR = PROJECT_ROOT / 'ai_detection_data_preparation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original manifests produced by the previous step
TRAIN_MANIFEST = OUTPUT_DIR / 'train_manifest.csv'
VAL_MANIFEST   = OUTPUT_DIR / 'validation_manifest.csv'
TEST_MANIFEST  = OUTPUT_DIR / 'test_manifest.csv'

# ---------------------------------------------------------------------
# Helper: read a manifest CSV into a list of dicts
def read_manifest(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

train_rows = read_manifest(TRAIN_MANIFEST)
val_rows   = read_manifest(VAL_MANIFEST)
test_rows  = read_manifest(TEST_MANIFEST)

# ---------------------------------------------------------------------
# Part A – Duplicate‑aware cleaning (only on labelled data)
labelled_rows = train_rows + val_rows  # both contain human/ai labels
hash_to_rows = defaultdict(list)
for r in labelled_rows:
    hash_to_rows[r['sha256']].append(r)

# Keep one representative per hash (first occurrence)
clean_labelled = []
for h, rows in hash_to_rows.items():
    # rows are in the order they appeared in the combined list (train then val)
    clean_labelled.append(rows[0])

# Verify no duplicate hash remains
assert all(len(v) == 1 for v in defaultdict(list, [(r['sha256'], r) for r in clean_labelled]).values())

# Re‑split into train/validation (stratified, reproducible)
labels = [r['label'] for r in clean_labelled]
filepaths = [r['filepath'] for r in clean_labelled]
train_fp, val_fp, train_lbl, val_lbl = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels)

clean_train = []
clean_val   = []
for r in clean_labelled:
    if r['filepath'] in train_fp:
        r['split'] = 'train'
        clean_train.append(r)
    else:
        r['split'] = 'validation'
        clean_val.append(r)

# ---------------------------------------------------------------------
# Write cleaned manifests
def write_manifest(rows, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'label', 'split', 'sha256'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

CLEAN_TRAIN_MANIFEST = OUTPUT_DIR / 'clean_train_manifest.csv'
CLEAN_VAL_MANIFEST   = OUTPUT_DIR / 'clean_validation_manifest.csv'
CLEAN_TEST_MANIFEST  = OUTPUT_DIR / 'clean_test_manifest.csv'  # copy of original test

write_manifest(clean_train, CLEAN_TRAIN_MANIFEST)
write_manifest(clean_val,   CLEAN_VAL_MANIFEST)
# copy test unchanged
import shutil
shutil.copy(TEST_MANIFEST, CLEAN_TEST_MANIFEST)

# ---------------------------------------------------------------------
# Part B – Feature extraction
# Load MiniLM (CPU)
model = SentenceTransformer(str(MODEL_PATH), device='cpu')

# Simple stylometric extractor (sentence‑wise, no external libs)
punctuation_chars = set('.;,!?-')

def stylometric_features(text):
    # Paragraph count (split on blank lines)
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    para_cnt = len(paragraphs)

    # Basic sentence split (very naive)
    sentences = re.split(r'[.!?]+\s*', text)
    sentences = [s for s in sentences if s.strip()]
    sent_cnt = len(sentences)
    if sent_cnt == 0:
        avg_sent_len = 0.0
        std_sent_len = 0.0
    else:
        sent_lens = [len(s.split()) for s in sentences]
        avg_sent_len = float(np.mean(sent_lens))
        std_sent_len = float(np.std(sent_lens, ddof=0))

    # Token level statistics
    words = re.findall(r"\b\w+\b", text.lower())
    word_cnt = len(words)
    if word_cnt == 0:
        avg_word_len = 0.0
        vocab_richness = 0.0
        ttr = 0.0
    else:
        word_lens = [len(w) for w in words]
        avg_word_len = float(np.mean(word_lens))
        vocab = set(words)
        vocab_richness = len(vocab) / word_cnt
        ttr = vocab_richness  # same as vocab richness here

    # Punctuation frequency (per character)
    total_chars = len(text)
    punct_cnt = sum(1 for ch in text if ch in punctuation_chars)
    punct_freq = punct_cnt / total_chars if total_chars else 0.0

    # Assemble feature vector
    return np.array([
        para_cnt,
        sent_cnt,
        avg_sent_len,
        std_sent_len,
        avg_word_len,
        vocab_richness,
        ttr,
        punct_freq,
    ], dtype=np.float32)

# Function to process a list of rows and return feature matrix and labels
def process_rows(rows):
    texts = []
    ids = []
    labels = []
    for r in rows:
        full_path = PROJECT_ROOT / r['filepath']
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                txt = f.read()
        except Exception as e:
            print(f"Failed to read {full_path}: {e}", file=sys.stderr)
            txt = ''
        texts.append(txt)
        ids.append(r['filepath'])
        # Convert label to binary (human=0, ai=1) for downstream use
        if r['label'] == 'human':
            labels.append(0)
        elif r['label'] == 'ai':
            labels.append(1)
        else:
            labels.append(-1)  # unknown – should not happen in train/val
    # MiniLM embeddings (batch to avoid OOM)
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
    # L2‑normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    # Stylometric features per document
    styl_features = np.vstack([stylometric_features(t) for t in texts])
    # Concatenate
    X = np.hstack([embeddings, styl_features])
    y = np.array(labels, dtype=np.int32)
    return X, y, ids

# Process clean train and validation sets
X_train, y_train, ids_train = process_rows(clean_train)
X_val,   y_val,   ids_val   = process_rows(clean_val)

# Basic sanity checks – no NaNs
assert not np.isnan(X_train).any()
assert not np.isnan(X_val).any()

# Save feature matrices (using np.savez for compactness)
np.savez(OUTPUT_DIR / 'train_features.npz', X=X_train, y=y_train, ids=ids_train)
np.savez(OUTPUT_DIR / 'validation_features.npz', X=X_val, y=y_val, ids=ids_val)

# ---------------------------------------------------------------------
# Summary output
print('=== Duplicate‑aware cleaning & feature extraction summary ===')
print(f'Cleaned labelled documents : {len(clean_labelled)}')
print(f'  Train : {len(clean_train)} (human {sum(1 for r in clean_train if r["label"]=="human")}, ai {sum(1 for r in clean_train if r["label"]=="ai")})')
print(f'  Validation : {len(clean_val)} (human {sum(1 for r in clean_val if r["label"]=="human")}, ai {sum(1 for r in clean_val if r["label"]=="ai")})')
print(f'Unlabelled test documents : {len(test_rows)}')
print(f'Embedding dimension : {model.get_sentence_embedding_dimension()}')
print(f'Stylistic feature count : {X_train.shape[1] - model.get_sentence_embedding_dimension()}')
print(f'Train feature matrix shape : {X_train.shape}')
print(f'Validation feature matrix shape : {X_val.shape}')
print('Generated files:')
print(' -', CLEAN_TRAIN_MANIFEST)
print(' -', CLEAN_VAL_MANIFEST)
print(' -', CLEAN_TEST_MANIFEST)
print(' -', OUTPUT_DIR / 'train_features.npz')
print(' -', OUTPUT_DIR / 'validation_features.npz')

sys.exit(0)