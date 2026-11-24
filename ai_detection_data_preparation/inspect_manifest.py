import csv, sys
from pathlib import Path
proj = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')
manifest = proj / 'ai_detection_data_preparation' / 'clean_validation_manifest.csv'
if not manifest.is_file():
    print('manifest not found')
    sys.exit(1)
rows = []
with open(manifest, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
print('Rows count:', len(rows))
# derive label from path
human = 0
ai = 0
for r in rows:
    if 'Training-Truth' in r['filepath']:
        human += 1
    else:
        ai += 1
print('Human:', human, 'AI:', ai)
# list first few ids
print('First 5 filepaths:', [r['filepath'] for r in rows[:5]])
sys.exit(0)
