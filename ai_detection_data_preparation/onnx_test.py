import os, sys, json, traceback
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
PROJECT_ROOT = Path(r'C:\\Users\\Chirag\\Desktop\\MIT AOE\\5th Sem\\PlagiarismAI-main')
MODEL_DIR = PROJECT_ROOT / 'models' / 'all-MiniLM-L6-v2'
ONNX_PATH = MODEL_DIR / 'model.onnx'

# ---------------------------------------------------------------------
# Helper to print and exit with code
def exit_report(message, code=0):
    print(message)
    sys.exit(code)

# ---------------------------------------------------------------------
# 1. Check for existing ONNX file
if ONNX_PATH.is_file():
    print(f'ONNX model already exists at {ONNX_PATH}')
else:
    print('ONNX model not found, will attempt export using torch')
    # Import torch and transformers (torch import should succeed as CPU build)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        exit_report(f'Failed to import torch/transformers for export: {e}', 1)

    # Load the huggingface model (torch backend)
    try:
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        exit_report(f'Error loading pretrained model for export: {e}', 1)

    model.eval()
    dummy_text = "test sentence for onnx export"
    inputs = tokenizer(dummy_text, return_tensors='pt')
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs['attention_mask']),
            str(ONNX_PATH),
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'seq'},
                'attention_mask': {0: 'batch', 1: 'seq'},
                'last_hidden_state': {0: 'batch', 1: 'seq'}
            },
            opset_version=14,
            do_constant_folding=True,
        )
        print(f'Exported ONNX model to {ONNX_PATH}')
    except Exception as e:
        exit_report(f'ONNX export failed: {e}', 1)

# ---------------------------------------------------------------------
# 2. Verify ONNX Runtime version and providers
try:
    import onnxruntime as ort
    print('onnxruntime version', ort.__version__)
    print('available providers', ort.get_available_providers())
except Exception as e:
    exit_report(f'Failed to import onnxruntime: {e}', 1)

# ---------------------------------------------------------------------
# 3. Load tokenizer (CPU) – reuse same tokenizer as before, re-import if needed
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    exit_report(f'Failed to load tokenizer: {e}', 1)

# ---------------------------------------------------------------------
# 4. Prepare two sample sentences
samples = [
    "Artificial intelligence is transforming technology.",
    "Students are learning machine learning."
]
enc = tokenizer(samples, padding=True, truncation=True, return_tensors='np')
input_ids = enc['input_ids'].astype(np.int64)
attention_mask = enc['attention_mask'].astype(np.int64)

# ---------------------------------------------------------------------
# 5. Run ONNX session
try:
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(str(ONNX_PATH), sess_options, providers=['CPUExecutionProvider'])
    outputs = sess.run(['last_hidden_state'], {'input_ids': input_ids, 'attention_mask': attention_mask})
    hidden = outputs[0]  # shape (batch, seq_len, hidden_dim)
    print('ONNX raw output shape', hidden.shape)
except Exception as e:
    exit_report(f'ONNX inference failed: {e}', 1)

# ---------------------------------------------------------------------
# 6. Mean pooling (as in sentence‑transformers) and L2‑normalize
try:
    # Convert attention mask to shape (batch, seq_len, 1)
    mask = np.expand_dims(attention_mask, axis=-1)
    # Zero‑out masked tokens
    masked_hidden = hidden * mask
    # Sum over token dimension
    sum_embeddings = masked_hidden.sum(axis=1)
    # Compute lengths (avoid division by zero)
    lengths = mask.sum(axis=1)
    # Avoid zero division
    lengths = np.where(lengths == 0, 1, lengths)
    mean_pooled = sum_embeddings / lengths
    # L2 normalize
    norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    normalized = mean_pooled / np.where(norms == 0, 1, norms)
    print('Final embedding shape', normalized.shape)
    print('All finite?', np.isfinite(normalized).all())
except Exception as e:
    exit_report(f'Post‑processing failed: {e}', 1)

# If we reach here, everything succeeded
exit_report('ONNX inference test completed successfully.', 0)
