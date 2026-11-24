import os, sys, traceback
# Force sentence_transformers to avoid torch backend
os.environ['TRANSFORMERS_NO_TORCH'] = '1'
# Additionally, prevent loading torch in sentence_transformers backend
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
try:
    from sentence_transformers import SentenceTransformer
    print('SentenceTransformer imported successfully')
except Exception as e:
    print('Import error:', e)
    traceback.print_exc()
