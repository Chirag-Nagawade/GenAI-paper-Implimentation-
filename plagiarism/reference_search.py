"""
Local Reference Database Indexing and Search Module.

Enables local reference document repositories:
- Upload and index PDF/DOCX/TXT documents.
- Segment into sliding-window chunks.
- Compute and persist all-MiniLM-L6-v2 embeddings per chunk.
- Batch vector cosine similarity using pure NumPy (FAISS-free).
- Pass candidate matches through the Evidence Gate.
- Manage stored reference documents (list, inspect, delete).
"""
import hashlib
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from plagiarism.chunking import create_sliding_windows
from storage.repository import get_storage


def compute_content_hash(text: str) -> str:
    """SHA-256 fingerprint of normalized document content for deduplication."""
    norm = " ".join(text.strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _default_extract_sentences(text: str) -> List[str]:
    app_mod = sys.modules.get("app")
    if app_mod and hasattr(app_mod, "extract_sentences"):
        return app_mod.extract_sentences(text)
    # Safe fallback splitting if app is not loaded
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [s for s in raw if len(s.split()) >= 4]


def _default_encode_sentences(model: Any, sentences: List[str]) -> np.ndarray:
    app_mod = sys.modules.get("app")
    if app_mod and hasattr(app_mod, "encode_sentences"):
        return app_mod.encode_sentences(model, sentences)
    if hasattr(model, "encode"):
        return model.encode(sentences, batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
    raise ValueError("Cannot encode sentences without a valid model or encode_sentences function.")


def index_reference_document(
    model: Any,
    document_name: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    storage_backend: Optional[Any] = None,
    window_size: int = 3,
    stride: int = 1,
    extract_sentences_fn: Optional[Callable[[str], List[str]]] = None,
    encode_sentences_fn: Optional[Callable[[Any, List[str]], Any]] = None
) -> Dict[str, Any]:
    """
    Index a document into the Local Reference Database.
    Extracts sentences, creates sliding windows, encodes them, and saves to storage.
    """
    storage = storage_backend or get_storage()
    extract_fn = extract_sentences_fn or _default_extract_sentences
    sentences = extract_fn(text)
    if not sentences:
        return {"ok": False, "reason": "No valid sentences extracted from document."}

    content_hash = compute_content_hash(text)
    # Check for duplicate document
    existing_docs = storage.get_reference_documents()
    for d in existing_docs:
        meta = d.get("metadata", {})
        if (meta and meta.get("content_hash") == content_hash) or d.get("document_name") == document_name:
            return {
                "ok": False,
                "reason": f"Duplicate document detected (identical content or name matching '{d.get('document_name')}').",
                "doc_id": d.get("id")
            }

    # Generate sliding window chunks
    chunks = create_sliding_windows(sentences, window_size=window_size, stride=stride)
    if not chunks:
        return {"ok": False, "reason": "Could not generate chunks from sentences."}

    chunk_texts = [c.text for c in chunks]
    # Encode with MiniLM
    encode_fn = encode_sentences_fn or _default_encode_sentences
    embeddings = encode_fn(model, chunk_texts)

    chunk_records = []
    for c, emb in zip(chunks, embeddings):
        chunk_records.append({
            "chunk_id": c.chunk_id,
            "sentence_indices": c.sentence_indices,
            "text": c.text,
            "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb
        })

    doc_meta = metadata or {}
    doc_meta["content_hash"] = content_hash
    doc_meta["total_sentences"] = len(sentences)
    doc_meta["total_chunks"] = len(chunks)

    doc_id = storage.save_reference_document(
        document_name=document_name,
        metadata=doc_meta,
        chunks=chunk_records
    )

    return {
        "ok": True,
        "doc_id": doc_id,
        "document_name": document_name,
        "total_sentences": len(sentences),
        "total_chunks": len(chunks)
    }


def search_reference_database(
    model: Any,
    sentences: List[str],
    user_embeddings: np.ndarray,
    storage_backend: Optional[Any] = None,
    threshold: float = 0.70,
    near: float = 0.90,
    use_hybrid: bool = False,
    hybrid_alpha: float = 0.75,
    encode_sentences_fn: Optional[Callable[[Any, List[str]], Any]] = None,
    match_against_source_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None
) -> List[Dict[str, Any]]:
    """
    Search user sentences against all chunks in the Local Reference Database.
    Reuses pre-computed vector embeddings, applies configurable hybrid scoring,
    and returns sentence-level matches with evidence fields for the Evidence Gate.
    
    Returns:
        List of match dicts matching app.py match format.
    """
    storage = storage_backend or get_storage()
    ref_chunks = storage.get_reference_chunks()
    if not ref_chunks:
        return []

    # Organize chunks by document
    docs_chunks: Dict[str, List[Dict[str, Any]]] = {}
    for ch in ref_chunks:
        doc_name = ch.get("document_name", "Local Reference")
        docs_chunks.setdefault(doc_name, []).append(ch)

    encode_fn = encode_sentences_fn or _default_encode_sentences

    all_matches = []

    for doc_name, chunks in docs_chunks.items():
        source_sentences = [c["text"] for c in chunks]
        source_url = f"local-ref://{doc_name}"
        source_title = f"[Local Ref] {doc_name}"

        # MiniLM embeddings: reuse stored embeddings directly without re-encoding
        emb_list = [c.get("embedding") for c in chunks]
        if any(e is None for e in emb_list):
            src_embeddings = encode_fn(model, source_sentences)
        else:
            src_embeddings = np.array(emb_list, dtype=np.float32)

        # Match against this reference document using standard pipeline cascade
        match_fn = match_against_source_fn
        if not match_fn:
            app_mod = sys.modules.get("app")
            if app_mod and hasattr(app_mod, "match_against_source"):
                match_fn = app_mod.match_against_source

        if match_fn:
            doc_matches = match_fn(
                model=model,
                sentences=sentences,
                user_embeddings=user_embeddings,
                source_sentences=source_sentences,
                title=source_title,
                url=source_url,
                semantic_threshold=threshold,
                near_threshold=near,
                source_embeddings=src_embeddings,
                use_hybrid=use_hybrid,
                hybrid_alpha=hybrid_alpha,
                use_chunking=False
            )
        else:
            doc_matches = []

        # Deduplicate overlapping chunk matches per sentence index for this source
        best_per_sentence: Dict[int, Dict[str, Any]] = {}
        for m in doc_matches:
            s_idx = m["index"]
            if s_idx not in best_per_sentence or m.get("score", 0.0) > best_per_sentence[s_idx].get("score", 0.0):
                best_per_sentence[s_idx] = m
        all_matches.extend(best_per_sentence.values())

    return all_matches


def list_reference_documents(storage_backend: Optional[Any] = None) -> List[Dict[str, Any]]:
    """List all stored reference documents."""
    storage = storage_backend or get_storage()
    return storage.get_reference_documents()


def delete_reference_document(doc_id: str, storage_backend: Optional[Any] = None) -> bool:
    """Delete a reference document from storage."""
    storage = storage_backend or get_storage()
    return storage.delete_reference_document(doc_id)
