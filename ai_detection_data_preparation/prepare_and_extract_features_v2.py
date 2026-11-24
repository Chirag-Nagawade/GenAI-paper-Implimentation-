import os, sys, csv, hashlib, traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import re

# Ensure torch and sentence_transformers are importable (CPU only)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print('Failed to import SentenceTransformer:', e)
    sys.exit(1)

# ---------------------------------------------------------------------
# Paths
PROJECT_ROOT = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')
DATA_ROOT = PROJECT_ROOT / 'data' / 'ChatGPT-Human-NewGenerated'
MODEL_PATH = PROJECT_ROOT / 'models' / 'all-MiniLM-L6-v2'
OUTPUT_DIR = PROJECT_ROOT / 'ai_detection_data_preparation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original manifests (generated earlier)
TRAIN_MANIFEST = OUTPUT_DIR / 'train_manifest.csv'
VAL_MANIFEST   = OUTPUT_DIR / 'validation_manifest.csv'
TEST_MANIFEST  = OUTPUT_DIR / 'test_manifest.csv'

# ---------------------------------------------------------------------
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
# Recalculate SHA-256 hashes from the actual files (ensures integrity)

def sha256_of_file(filepath: Path):
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f'Error reading {filepath}: {e}', file=sys.stderr)
        return None

# Update hash values for all rows (train + val)
all_labelled = []
for r in train_rows + val_rows:
    full_path = PROJECT_ROOT / r['filepath']
    new_hash = sha256_of_file(full_path)
    if new_hash is None:
        sys.exit(1)
    r['sha256'] = new_hash
    all_labelled.append(r)

# ---------------------------------------------------------------------
# Duplicate detection (exact content) across labelled data
hash_to_rows = defaultdict(list)
for r in all_labelled:
    hash_to_rows[r['sha256']].append(r)

# Keep one representative per hash (first occurrence)
unique_labelled = [group[0] for group in hash_to_rows.values()]

# Verify no duplicate hashes remain
assert len(set(r['sha256'] for r in unique_labelled)) == len(unique_labelled)

# ---------------------------------------------------------------------
# Stratified train/validation split (80/20) on unique labelled set
labels = [r['label'] for r in unique_labelled]
filepaths = [r['filepath'] for r in unique_labelled]

from sklearn.model_selection import train_test_split
train_fp, val_fp, train_lbl, val_lbl = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
)

clean_train = []
clean_val   = []
for r in unique_labelled:
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
        writer = csv.DictWriter(f, fieldnames=['filepath','label','split','sha256'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

CLEAN_TRAIN_MANIFEST = OUTPUT_DIR / 'clean_train_manifest.csv'
CLEAN_VAL_MANIFEST   = OUTPUT_DIR / 'clean_validation_manifest.csv'
CLEAN_TEST_MANIFEST  = OUTPUT_DIR / 'clean_test_manifest.csv'

write_manifest(clean_train, CLEAN_TRAIN_MANIFEST)
write_manifest(clean_val,   CLEAN_VAL_MANIFEST)
# copy test manifest unchanged
import shutil
shutil.copy(TEST_MANIFEST, CLEAN_TEST_MANIFEST)

# ---------------------------------------------------------------------
# Load MiniLM model (CPU, frozen)
model = SentenceTransformer(str(MODEL_PATH), device='cpu')
embed_dim = model.get_sentence_embedding_dimension()

# ---------------------------------------------------------------------
# Stylometric feature extractor (same as before)
punctuation_chars = set('.;,!?-')

def stylometric_features(text: str):
    # Paragraph count (split on blank lines)
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    para_cnt = len(paragraphs)
    # Very simple sentence split
    sentences = re.split(r'[.!?]+\s*', text)
    sentences = [s for s in sentences if s.strip()]
    sent_cnt = len(sentences)
    if sent_cnt == 0:
        avg_sent_len = std_sent_len = 0.0
    else:
        sent_lens = [len(s.split()) for s in sentences]
        avg_sent_len = float(np.mean(sent_lens))
        std_sent_len = float(np.std(sent_lens))
    # Word‑level stats
    words = re.findall(r"\b\w+\b", text.lower())
    word_cnt = len(words)
    if word_cnt == 0:
        avg_word_len = vocab_rich = ttr = 0.0
    else:
        word_lens = [len(w) for w in words]
        avg_word_len = float(np.mean(word_lens))
        vocab = set(words)
        vocab_rich = len(vocab) / word_cnt
        ttr = vocab_rich
    # Punctuation frequency per character
    total_chars = len(text)
    punct_cnt = sum(1 for ch in text if ch in punctuation_chars)
    punct_freq = punct_cnt / total_chars if total_chars else 0.0
    return np.array([
        para_cnt,
        sent_cnt,
        avg_sent_len,
        std_sent_len,
        avg_word_len,
        vocab_rich,
        ttr,
        punct_freq,
    ], dtype=np.float32)

# ---------------------------------------------------------------------
# Helper to read a file's text safely
def read_text(rel_path):
    full_path = PROJECT_ROOT / rel_path
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f'Failed to read {rel_path}: {e}', file=sys.stderr)
        return ''

# ---------------------------------------------------------------------
# Process a list of rows to produce embeddings and stylometric features
def process_rows(rows, batch_size=4):
    texts = []
    ids = []
    labels = []
    for r in rows:
        txt = read_text(r['filepath'])
        texts.append(txt)
        ids.append(r['filepath'])
        labels.append(0 if r['label']=='human' else 1)
    # MiniLM embeddings in batches
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
        # L2‑normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms==0, 1, norms)
        all_emb.append(emb)
    embeddings = np.vstack(all_emb)  # (N, 384)
    # Stylometric features
    styl_feats = np.vstack([stylometric_features(t) for t in texts])  # (N, 8)
    # Concatenate
    X = np.hstack([embeddings, styl_feats])
    y = np.array(labels, dtype=np.int32)
    return X, y, ids

# ---------------------------------------------------------------------
X_train, y_train, ids_train = process_rows(clean_train, batch_size=4)
X_val,   y_val,   ids_val   = process_rows(clean_val,   batch_size=4)

# ---------------------------------------------------------------------
# Sanity checks
assert not np.isnan(X_train).any(), 'NaNs in training features'
assert not np.isnan(X_val).any(),   'NaNs in validation features'
assert not np.isinf(X_train).any(), 'Infs in training features'
assert not np.isinf(X_val).any(),   'Infs in validation features'

# ---------------------------------------------------------------------
# Save results
np.savez(OUTPUT_DIR / 'train_features.npz', X=X_train, y=y_train, ids=ids_train)
np.savez(OUTPUT_DIR / 'validation_features.npz', X=X_val,   y=y_val,   ids=ids_val)

# ---------------------------------------------------------------------
# Summary report
print('=== Duplicate‑aware cleaning & feature extraction summary ===')
print(f'Unique labelled documents after deduplication: {len(clean_train)+len(clean_val)}')
print(f'  Train: {len(clean_train)} (human {sum(1 for r in clean_train if r["label"]=="human")}, ai {sum(1 for r in clean_train if r["label"]=="ai")})')
print(f'  Validation: {len(clean_val)} (human {sum(1 for r in clean_val if r["label"]=="human")}, ai {sum(1 for r in clean_val if r["label"]=="ai")})')
print(f'Unlabelled test documents (unchanged): {len(test_rows)}')
print(f'Embedding dimension: {embed_dim}')
print(f'Stylistic feature count: {X_train.shape[1]-embed_dim}')
print(f'Train feature matrix shape: {X_train.shape}')
print(f'Validation feature matrix shape: {X_val.shape}')
print('Generated files:')
print(' -', CLEAN_TRAIN_MANIFEST)
print(' -', CLEAN_VAL_MANIFEST)
print(' -', CLEAN_TEST_MANIFEST)
print(' -', OUTPUT_DIR / 'train_features.npz')
print(' -', OUTPUT_DIR / 'validation_features.npz')

sys.exit(0)
