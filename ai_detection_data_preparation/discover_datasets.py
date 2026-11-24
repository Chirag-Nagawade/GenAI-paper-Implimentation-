import os, sys, json, textwrap
from pathlib import Path

PROJECT_ROOT = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')

# candidate extensions for textual documents
EXTS = {'.txt', '.md', '.csv', '.json', '.jsonl', '.tsv'}

candidates = []

# walk limited depth (up to 4 levels) to avoid huge output
for root, dirs, files in os.walk(PROJECT_ROOT):
    rel = Path(root).relative_to(PROJECT_ROOT)
    if len(rel.parts) > 4:
        # prune deeper branches
        continue
    txt_files = [f for f in files if Path(f).suffix.lower() in EXTS]
    if txt_files:
        candidates.append({
            'folder': str(rel),
            'file_count': len(txt_files),
            'sample_files': txt_files[:5]  # first few names
        })

# write a simple report to stdout (will be captured)
print('Found candidate text containers:')
for cand in candidates:
    print(f"Folder: {cand['folder'] or '.'}, files: {cand['file_count']}, examples: {cand['sample_files']}")

# also list any subfolders named like 'datasets' or containing the word 'data'
potential = []
for d in PROJECT_ROOT.rglob('*'):
    if d.is_dir() and ('data' in d.name.lower() or 'dataset' in d.name.lower()):
        potential.append(str(d.relative_to(PROJECT_ROOT)))
print('\nPotential dataset directories (by name):')
for p in sorted(set(potential)):
    print(p)

sys.exit(0)
