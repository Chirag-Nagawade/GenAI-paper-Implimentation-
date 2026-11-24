import os
from sentence_transformers import SentenceTransformer
model_path = r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main\\models\\all-MiniLM-L6-v2'
print('Model directory exists:', os.path.isdir(model_path))
# Verify required files
required_files = [
    'config.json',
    'model.safetensors',
    'tokenizer.json',
    'vocab.txt',
    'sentence_bert_config.json',
    'config_sentence_transformers.json'
]
missing = [f for f in required_files if not os.path.isfile(os.path.join(model_path, f))]
if missing:
    print('Missing files:', missing)
else:
    print('All required files present')
try:
    model = SentenceTransformer(model_path, device='cpu')
    print('Model loaded successfully')
    emb = model.encode(['test sentence'])
    print('Embedding shape:', emb.shape)
except Exception as e:
    print('Error loading model:', e)
