import os
import hashlib
import csv
import sys
from pathlib import Path
from collections import defaultdict

# sklearn is likely installed in the environment; if not, we can implement manual stratified split.
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print('sklearn not available; exiting', file=sys.stderr)
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(r'C:\Users\Chirag\Desktop\MIT AOE\5th Sem\PlagiarismAI-main')
DATA_ROOT = PROJECT_ROOT / 'data' / 'ChatGPT-Human-NewGenerated'
OUTPUT_DIR = PROJECT_ROOT / 'ai_detection_data_preparation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper to compute SHA-256
def sha256_of_file(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return None

records = []
issues = []
# 1. Gather training human files
human_dir = DATA_ROOT / 'Training-Truth'
ai_dir = DATA_ROOT / 'Training-Problems'
testing_dir = DATA_ROOT / 'Testing'

for label_name, dir_path, label in [('human', human_dir, 'human'), ('ai', ai_dir, 'ai')]:
    if not dir_path.is_dir():
        issues.append(f"Missing expected directory: {dir_path}")
        continue
    for file_path in sorted(dir_path.glob('*.txt')):
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        hash_val = sha256_of_file(file_path)
        if hash_val is None:
            issues.append(f"Unreadable file: {rel_path}")
            continue
        size = file_path.stat().st_size
        if size == 0:
            issues.append(f"Empty file: {rel_path}")
        records.append({
            'filepath': rel_path,
            'label': label,
            'split': 'train_val',  # placeholder, will be split later
            'sha256': hash_val
        })

# 2. Gather testing files (unknown label)
if testing_dir.is_dir():
    for file_path in sorted(testing_dir.glob('*.txt')):
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        hash_val = sha256_of_file(file_path)
        if hash_val is None:
            issues.append(f"Unreadable file: {rel_path}")
            continue
        size = file_path.stat().st_size
        if size == 0:
            issues.append(f"Empty file: {rel_path}")
        records.append({
            'filepath': rel_path,
            'label': 'unknown',
            'split': 'test',
            'sha256': hash_val
        })
else:
    issues.append(f"Missing Testing directory: {testing_dir}")

# 3. Duplicate detection across all records
hash_to_paths = defaultdict(list)
for rec in records:
    hash_to_paths[rec['sha256']].append(rec['filepath'])
duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

# 4. Stratified train/validation split (only on labelled records)
labelled = [r for r in records if r['label'] in ('human', 'ai')]
labels = [r['label'] for r in labelled]
filepaths = [r['filepath'] for r in labelled]
# Use sklearn split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
)
# Assign splits back to records
for rec in records:
    if rec['split'] == 'test':
        continue
    if rec['filepath'] in train_paths:
        rec['split'] = 'train'
    elif rec['filepath'] in val_paths:
        rec['split'] = 'validation'
    else:
        # Should not happen
        issues.append(f"File not assigned to split: {rec['filepath']}")

# 5. Write CSVs per split
def write_manifest(split_name, rows):
    out_path = OUTPUT_DIR / f"{split_name}_manifest.csv"
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filepath', 'label', 'split', 'sha256'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_path

train_rows = [r for r in records if r['split'] == 'train']
val_rows = [r for r in records if r['split'] == 'validation']
test_rows = [r for r in records if r['split'] == 'test']

train_csv = write_manifest('train', train_rows)
val_csv = write_manifest('validation', val_rows)
test_csv = write_manifest('test', test_rows)

# 6. Summary output
print('=== Dataset preparation summary ===')
print(f"Total labelled (human+ai) documents: {len(labelled)}")
print(f"  Human: {sum(1 for l in labels if l=='human')}, AI: {sum(1 for l in labels if l=='ai')}")
print(f"Train split: {len(train_rows)} (human {sum(1 for r in train_rows if r['label']=='human')}, ai {sum(1 for r in train_rows if r['label']=='ai')})")
print(f"Validation split: {len(val_rows)} (human {sum(1 for r in val_rows if r['label']=='human')}, ai {sum(1 for r in val_rows if r['label']=='ai')})")
print(f"Testing files (unknown label): {len(test_rows)}")
print(f"Duplicate content groups: {len(duplicates)}")
if duplicates:
    for h, paths in duplicates.items():
        print(f"  SHA256 {h}: {len(paths)} files -> {paths}")
if issues:
    print('Issues detected:')
    for i in issues:
        print(' -', i)
else:
    print('No missing/empty/unreadable file issues.')

print('CSV files generated:')
for p in [train_csv, val_csv, test_csv]:
    print(' -', p)

sys.exit(0)
