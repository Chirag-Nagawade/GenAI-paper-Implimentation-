import os, sys, numpy as np
from sentence_transformers import SentenceTransformer
model_path = r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main\\models\\all-MiniLM-L6-v2'
print('Loading model from:', model_path)
model = SentenceTransformer(model_path, device='cpu')
print('Model loaded')
texts = [
    'Artificial intelligence is transforming technology.',
    'Students are learning machine learning.'
]
emb = model.encode(texts, convert_to_numpy=True)
print('Embeddings shape:', emb.shape)
print('All finite:', np.isfinite(emb).all())
