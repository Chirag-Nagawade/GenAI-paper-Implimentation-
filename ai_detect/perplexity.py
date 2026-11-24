"""Optional perplexity features from a small causal LM (distilgpt2).
AI text tends to have LOWER and more UNIFORM token surprise than human text.
Requires: pip install torch transformers  (first use downloads the model)."""
from typing import List, Optional
import numpy as np

PPL_FEATURE_NAMES = ["ppl_mean_logprob", "ppl_std_logprob", "ppl_frac_top_tokens"]
_CACHE = {}


def is_available() -> bool:
    try:
        import torch  # noqa
        import transformers  # noqa
        return True
    except Exception:
        return False


def _load(name: str = "distilgpt2"):
    if name not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).eval()
        _CACHE[name] = (tok, model, torch)
    return _CACHE[name]


def ppl_features(texts: List[str], max_tokens: int = 256) -> np.ndarray:
    tok, model, torch = _load()
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"]
        if ids.shape[1] < 3:
            out.append([0.0, 0.0, 0.0]); continue
        with torch.no_grad():
            logits = model(ids).logits[0, :-1]
        logp = torch.log_softmax(logits, dim=-1)
        tgt = ids[0, 1:]
        tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
        top = (logits.argmax(-1) == tgt).float().mean().item()   # how often the LM's #1 guess is right
        out.append([tok_lp.mean().item(), tok_lp.std().item(), top])
    return np.array(out, dtype=np.float32)
