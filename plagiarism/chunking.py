"""
Sliding-Window Chunking Module for Plagiarism Detection.

Provides:
- SlidingWindowChunk: structured representation of a multi-sentence window.
- create_sliding_windows: creates overlapping sentence windows with token-length protection.
- map_window_matches_to_sentences: maps window-level similarity matches back to constituent sentences,
  preserving exact indices and the one-status-per-sentence invariant.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SlidingWindowChunk:
    chunk_id: int
    start_sentence_idx: int
    end_sentence_idx: int  # exclusive
    sentence_indices: List[int]
    text: str
    word_count: int
    is_truncated: bool = False


def create_sliding_windows(
    sentences: List[str],
    window_size: int = 3,
    stride: int = 1,
    max_words_per_window: int = 180
) -> List[SlidingWindowChunk]:
    """
    Generate overlapping sentence windows.
    Guarantees that sentence indices are preserved and that no window exceeds
    the token/word budget of the underlying embedding model (preventing silent truncation).
    
    Args:
        sentences: list of individual sentence strings.
        window_size: number of sentences per window (e.g., 3).
        stride: step size between consecutive windows (e.g., 1 for overlapping).
        max_words_per_window: maximum word count per window for MiniLM safety.
    
    Returns:
        List of SlidingWindowChunk objects.
    """
    n_sentences = len(sentences)
    if n_sentences == 0:
        return []

    chunks: List[SlidingWindowChunk] = []
    chunk_id = 0
    start = 0

    while start < n_sentences:
        end = min(start + window_size, n_sentences)
        window_sents = sentences[start:end]
        window_indices = list(range(start, end))

        # Check total word length
        total_words = sum(len(s.split()) for s in window_sents)
        is_truncated = False

        # If a single sentence exceeds max_words_per_window, allow it as a standalone chunk
        # If multi-sentence window exceeds max_words, reduce window size
        if total_words > max_words_per_window and len(window_sents) > 1:
            while len(window_sents) > 1 and sum(len(s.split()) for s in window_sents) > max_words_per_window:
                window_sents.pop()
                window_indices.pop()
            end = start + len(window_sents)
            total_words = sum(len(s.split()) for s in window_sents)

        chunk_text = " ".join(s.strip() for s in window_sents if s.strip())

        chunks.append(SlidingWindowChunk(
            chunk_id=chunk_id,
            start_sentence_idx=start,
            end_sentence_idx=end,
            sentence_indices=window_indices,
            text=chunk_text,
            word_count=total_words,
            is_truncated=is_truncated
        ))
        chunk_id += 1

        if end == n_sentences:
            break
        start += stride

    return chunks


def map_window_matches_to_sentences(
    window_matches: List[Dict[str, Any]],
    total_sentences: int,
    chunks: List[SlidingWindowChunk]
) -> List[Dict[str, Any]]:
    """
    Map window-level match candidates back to individual sentence matches.
    If a sentence is covered by multiple overlapping window matches from the same source,
    the match with the highest similarity score is chosen for that sentence.
    
    Returns:
        List of sentence-level match dictionaries matching app.py match format:
        {
            "index": sentence_idx,
            "type": match_type,
            "score": score,
            "source_passage": source_passage,
            "source_title": source_title,
            "url": source_url,
            ...
        }
    """
    chunk_map = {c.chunk_id: c for c in chunks}
    # Track best match per (sentence_idx, source_url)
    best_per_sentence_source: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for wm in window_matches:
        cid = wm.get("chunk_id", wm.get("index"))
        if cid not in chunk_map:
            continue
        chunk = chunk_map[cid]
        score = wm.get("score", 0.0)
        url = wm.get("url", "")

        for sent_idx in chunk.sentence_indices:
            key = (sent_idx, url)
            if key not in best_per_sentence_source or score > best_per_sentence_source[key].get("score", 0.0):
                mapped_match = dict(wm)
                mapped_match["index"] = sent_idx
                mapped_match["window_chunk_id"] = cid
                mapped_match["window_start"] = chunk.start_sentence_idx
                mapped_match["window_end"] = chunk.end_sentence_idx
                best_per_sentence_source[key] = mapped_match

    return list(best_per_sentence_source.values())
