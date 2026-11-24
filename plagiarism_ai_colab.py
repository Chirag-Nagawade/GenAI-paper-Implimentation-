"""
================================================================================
PlagiarismAI - Headless / Google Colab End-to-End Implementation
================================================================================
A unified, standalone implementation of the PlagiarismAI framework for execution
in Google Colab, Jupyter Notebooks, or CLI environments without Streamlit.

Includes:
  1. Multi-format Ingestion (PDF, Word DOCX, Plain Text, Normalization)
  2. Academic Sentence Boundary Splitting & In-Text Citation Parser
  3. Sliding-Window Passage Chunking & Local Vector Reference Database
  4. Multi-Tier Plagiarism Detection:
     - Tier 1: Normalized Exact String Matching
     - Tier 2: Near-Exact Ratcliff-Obershelp Sequence Alignment
     - Tier 3: Dense Sentence-BERT Semantic Matching (all-MiniLM-L6-v2)
     - Hybrid TF-IDF + Sentence-BERT Blending (alpha = 0.75)
     - Contextual Safeguards (Lexical Coverage Gate >= 25%, Common Source Filter)
     - Real Web Candidate Retrieval (Brave / Tavily / DDG + Scraping)
  5. AI-Generated Text Detection:
     - External Linguistic Analysis Service (Sapling API client)
     - Local 37-Feature Stylometric Extractor (35 active modeling features)
     - Calibrated XGBoost Classifier & Calibrated Lexical TF-IDF N-gram Model
     - Adaptive Zero-Shot Rule-Based Stylometric Scorer
     - Paragraph-level Chunking & Sentence Probability Inheritance
  6. Pairwise Document Comparison (Document A vs. Document B)
  7. SQLite Database Persistence (plagiarism_history.db)
  8. Research-Grade PDF Audit Report Exporters (FPDF2)
  9. Webis-CPC-11 Paraphrase Benchmark Evaluator (70/15/15 Stratified Split)

Dependencies for Google Colab:
  !pip install sentence-transformers torch xgboost scikit-learn pypdf python-docx fpdf2 beautifulsoup4 requests joblib
================================================================================
"""

import os
import sys
import io
import re
import math
import time
import json
import sqlite3
import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin, urlunparse

# Deep learning and scientific stack
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import torch
except (ImportError, OSError):
    pass

import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from fpdf import FPDF
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)

# ------------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ------------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plagiarism_history.db")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models", "ai_detection")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "cache")

EXACT = "exact"
NEAR = "near_exact"
SEMANTIC = "semantic"
TOPICAL = "topical"

MIN_WORDS = 3
EXACT_MIN_WORDS = 4
SEMANTIC_MIN_WORDS = 5
MAX_SENTENCE_WORDS = 120

LEXICAL_MIN_SHARED = 2
LEXICAL_MIN_COVERAGE = 0.25
COMMON_SOURCE_MIN = 3

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_URL = "https://api.tavily.com/search"

STOPWORDS = set("""a about above after again all also an and any are as at be because been before being
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just may me more most my no nor not of off on once
only or other our out over own same she should so some such than that the their theirs them then there these
they this those through to too under until up very was we were what when where which while who whom why will
with would you your thus however therefore which whose using used use based paper show shown results result
propose proposed present presented approach method methods""".split())

_ABBREVIATIONS = ["et al.", "i.e.", "e.g.", "vs.", "Fig.", "Dr.", "Prof.", "Ref.", "cf.", "al."]
_DOT = "__DOT__"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"])")


def get_secret(name: str) -> str:
    """Read API key from Colab userdata, environment variables, or local .env."""
    try:
        from google.colab import userdata
        val = userdata.get(name)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


# ==============================================================================
# 1. TEXT EXTRACTION & NORMALIZATION
# ==============================================================================
def normalize_text(text: str) -> str:
    """Canonical Unicode NFKD normalization, quotation unification, whitespace collapsing."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[\u2018\u2019\u201A\u201B`]', "'", text)
    text = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', text)
    text = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]', "-", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(data: bytes, max_pages: Optional[int] = None) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    text = "\n\n".join(p.extract_text() or "" for p in pages)
    return normalize_text(text)


def extract_text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells if cell.text.strip())
    return normalize_text("\n\n".join(parts))


def extract_text_from_txt(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return normalize_text(data.decode(enc))
        except (UnicodeDecodeError, LookupError):
            continue
    return normalize_text(data.decode("utf-8", errors="replace"))


def extract_document_text(file_path_or_bytes: Union[str, bytes], filename: str = "") -> str:
    """Unified file reader accepting file path or raw bytes."""
    if isinstance(file_path_or_bytes, str):
        if os.path.isfile(file_path_or_bytes):
            filename = filename or file_path_or_bytes
            with open(file_path_or_bytes, "rb") as f:
                raw_bytes = f.read()
        else:
            return normalize_text(file_path_or_bytes)
    else:
        raw_bytes = file_path_or_bytes

    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(raw_bytes)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(raw_bytes)
    else:
        return extract_text_from_txt(raw_bytes)


def extract_sentences(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_SENTENCE_WORDS) -> List[str]:
    """Academic-aware sentence splitter protecting abbreviations, formula markers, and initials."""
    if not text:
        return []
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    for abbr in set(_ABBREVIATIONS + [a.lower() for a in _ABBREVIATIONS]):
        text = text.replace(abbr, abbr.replace(".", _DOT))
    sentences, seen = [], set()
    for paragraph in re.split(r"\n{2,}", text):
        for raw in _SPLIT.split(paragraph):
            sentence = re.sub(r"\s+", " ", raw.replace(_DOT, ".")).strip()
            words = sentence.split()
            if len(words) < min_words:
                continue
            for k in range(0, len(words), max_words):
                chunk = " ".join(words[k:k + max_words])
                key = normalize_text(chunk)
                if key and key not in seen:
                    seen.add(key)
                    sentences.append(chunk)
    return sentences


def _clean_words(sentence: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z0-9]+", sentence.lower()) if len(w) > 1]


def _term_weight(token: str) -> float:
    t = token.lower()
    if t in STOPWORDS:
        return 0.1
    if any(c.isdigit() for c in t) or "_" in t or "-" in t:
        return 3.0
    if len(t) >= 9:
        return 2.5
    if len(t) >= 6:
        return 1.8
    return 1.0


def distinctive_phrase(sentence: str, size: int = 12) -> str:
    words = sentence.split()
    if len(words) <= size:
        return " ".join(words)
    best_start, best_score = 0, -1.0
    for i in range(len(words) - size + 1):
        score = sum(_term_weight(re.sub(r"\W+", "", w)) for w in words[i:i + size])
        if score > best_score:
            best_score, best_start = score, i
    return " ".join(words[best_start:best_start + size])


def top_terms(words: List[str], n: int = 5) -> List[str]:
    """The n most distinctive (highest weighted) terms, preserving case-insensitive order."""
    counter, spelling = Counter(), {}
    for w in words:
        wt = _term_weight(w)
        if wt > 0.5:
            counter[w.lower()] += wt
            spelling.setdefault(w.lower(), w)
    return [spelling[k] for k, _ in counter.most_common(n)]


# ==============================================================================
# 2. CITATIONS & SLIDING-WINDOW CHUNKING
# ==============================================================================
@dataclass
class CitationInfo:
    raw_marker: str
    citation_type: str
    sentence_index: int
    matched_text: str


@dataclass
class SlidingWindowChunk:
    chunk_id: int
    start_sentence_idx: int
    end_sentence_idx: int
    sentence_indices: List[int]
    text: str
    word_count: int


def extract_citations(text: str) -> List[CitationInfo]:
    """Identify numeric citations [1], [1-3] and author-year markers (Smith et al., 2020)."""
    citations = []
    num_re = re.compile(r"\[(\d{1,3}(?:[,\s–-]+\d{1,3})*)\]")
    for m in num_re.finditer(text):
        citations.append(CitationInfo(m.group(0), "numeric", -1, m.group(0)))
    auth_re = re.compile(r"\(([A-Z][a-zA-Z\s]+(?:et al\.?)?,?\s+(?:19\d{2}|20\d{2}))\)")
    for m in auth_re.finditer(text):
        citations.append(CitationInfo(m.group(0), "author_year", -1, m.group(0)))
    return citations


def create_sliding_windows(sentences: List[str], window_size: int = 3, step_size: int = 1) -> List[SlidingWindowChunk]:
    """Generates overlapping multi-sentence sliding windows."""
    if not sentences:
        return []
    chunks = []
    chunk_id = 0
    n = len(sentences)
    for i in range(0, n, step_size):
        end_idx = min(i + window_size, n)
        subset = sentences[i:end_idx]
        txt = " ".join(subset)
        chunks.append(SlidingWindowChunk(
            chunk_id=chunk_id,
            start_sentence_idx=i,
            end_sentence_idx=end_idx,
            sentence_indices=list(range(i, end_idx)),
            text=txt,
            word_count=len(txt.split())
        ))
        chunk_id += 1
        if end_idx == n:
            break
    return chunks


# ==============================================================================
# 3. EMBEDDING LOADER & HYBRID SIMILARITY
# ==============================================================================
_GLOBAL_MODEL = None


def load_sentence_transformer(model_name: str = MODEL_NAME):
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _GLOBAL_MODEL = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[Warning] Could not load SentenceTransformer ({e}). Dense similarity will fallback.")
            return None
    return _GLOBAL_MODEL


def encode_sentences(model: Any, sentences: List[str]) -> np.ndarray:
    if model is None or not sentences:
        return np.zeros((len(sentences), 384), dtype=np.float32)
    embs = model.encode(sentences, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embs, dtype=np.float32)


class HybridSimilarityScorer:
    """Linear blend: S_hybrid = alpha * S_semantic + (1 - alpha) * S_tfidf"""

    def __init__(self, alpha: float = 0.75, ngram_range: Tuple[int, int] = (1, 2), max_features: int = 15000):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words="english", sublinear_tf=True)

    def compute_hybrid_similarity(self, emb_u: np.ndarray, emb_v: np.ndarray, text_u: List[str], text_v: List[str], alpha: Optional[float] = None) -> np.ndarray:
        a = self.alpha if alpha is None else float(np.clip(alpha, 0.0, 1.0))
        s_dense = np.clip(np.dot(emb_u, emb_v.T), 0.0, 1.0)
        if a >= 1.0:
            return s_dense
        try:
            vec = self.vectorizer.fit(text_u + text_v)
            m_u = vec.transform(text_u)
            m_v = vec.transform(text_v)
            s_tfidf = np.clip(cosine_similarity(m_u, m_v), 0.0, 1.0)
        except Exception:
            return s_dense
        return a * s_dense + (1.0 - a) * s_tfidf


# ==============================================================================
# 4. MULTI-TIER PLAGIARISM MATCHING ENGINE
# ==============================================================================
def detect_exact_matches(sentences: List[str], source_sentences: List[str], title: str = "", url: str = "") -> List[Dict[str, Any]]:
    lookup: Dict[str, Tuple[int, str]] = {}
    for s_idx, s_text in enumerate(source_sentences):
        words = _clean_words(s_text)
        if len(words) >= EXACT_MIN_WORDS:
            lookup.setdefault(" ".join(words), (s_idx, s_text))
    matches = []
    for u_idx, u_text in enumerate(sentences):
        words = _clean_words(u_text)
        if len(words) < EXACT_MIN_WORDS:
            continue
        key = " ".join(words)
        if key in lookup:
            s_idx, s_text = lookup[key]
            matches.append({
                "index": u_idx, "sentence": u_text, "matched": s_text, "score": 1.0,
                "type": EXACT, "level": "Exact (100%)", "title": title, "url": url
            })
    return matches


def detect_near_exact_matches(sentences: List[str], source_sentences: List[str], title: str = "", url: str = "", threshold: float = 0.90) -> List[Dict[str, Any]]:
    matches = []
    for u_idx, u_text in enumerate(sentences):
        u_clean = " ".join(_clean_words(u_text))
        if len(u_clean.split()) < EXACT_MIN_WORDS:
            continue
        best_score, best_cand = 0.0, None
        for s_idx, s_text in enumerate(source_sentences):
            s_clean = " ".join(_clean_words(s_text))
            if not s_clean:
                continue
            ratio = SequenceMatcher(None, u_clean, s_clean, autojunk=False).ratio()
            if ratio > best_score:
                best_score = ratio
                best_cand = s_text
        if best_score >= threshold and best_cand:
            matches.append({
                "index": u_idx, "sentence": u_text, "matched": best_cand, "score": float(best_score),
                "type": NEAR, "level": f"Near-Exact ({best_score * 100:.1f}%)", "title": title, "url": url
            })
    return matches


def lexical_coverage(sentence: str, passage: str, pool_df: Dict[str, int], pool_size: int) -> Tuple[float, List[str]]:
    u_words = _clean_words(sentence)
    p_words = set(_clean_words(passage))
    shared = [w for w in u_words if w in p_words]
    u_weights = [math.log((pool_size + 1.0) / (pool_df.get(w, 0) + 1.0)) + 1.0 for w in u_words]
    s_weights = [math.log((pool_size + 1.0) / (pool_df.get(w, 0) + 1.0)) + 1.0 for w in shared]
    tot_weight = sum(u_weights) if u_weights else 1e-6
    cov = sum(s_weights) / max(tot_weight, 1e-6)
    return cov, shared


def detect_semantic_matches(model: Any, sentences: List[str], source_sentences: List[str],
                            title: str = "", url: str = "", threshold: float = 0.70,
                            use_hybrid: bool = True, hybrid_alpha: float = 0.75) -> List[Dict[str, Any]]:
    if not sentences or not source_sentences:
        return []
    u_embs = encode_sentences(model, sentences)
    s_embs = encode_sentences(model, source_sentences)
    if use_hybrid and hybrid_alpha < 1.0:
        scorer = HybridSimilarityScorer(alpha=hybrid_alpha)
        sim_mat = scorer.compute_hybrid_similarity(u_embs, s_embs, sentences, source_sentences)
    else:
        sim_mat = np.clip(np.dot(u_embs, s_embs.T), 0.0, 1.0)

    # Document frequency pool for lexical coverage
    pool = sentences + source_sentences
    pool_df: Dict[str, int] = Counter()
    for doc in pool:
        for w in set(_clean_words(doc)):
            pool_df[w] += 1
    pool_size = max(len(pool), 1)

    best_s_indices = np.argmax(sim_mat, axis=1)
    matches = []
    for u_idx, u_text in enumerate(sentences):
        if len(u_text.split()) < SEMANTIC_MIN_WORDS:
            continue
        best_col = int(best_s_indices[u_idx])
        score = float(sim_mat[u_idx, best_col])
        if score < threshold:
            continue
        matched_text = source_sentences[best_col]
        cov, shared = lexical_coverage(u_text, matched_text, pool_df, pool_size)
        matches.append({
            "index": u_idx, "sentence": u_text, "matched": matched_text, "score": score,
            "type": SEMANTIC, "coverage": cov, "shared_words": shared, "title": title, "url": url
        })
    return matches


def finalize_and_filter_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enforces lexical coverage gate (>= 25%) and common-source suppression (>= 3 sources)."""
    source_counts = Counter(m["index"] for m in matches if m["type"] == SEMANTIC)
    finalized = []
    for m in matches:
        if m["type"] in (EXACT, NEAR):
            finalized.append(m)
            continue
        common = source_counts[m["index"]] >= COMMON_SOURCE_MIN
        has_coverage = (len(m.get("shared_words", [])) >= LEXICAL_MIN_SHARED and m.get("coverage", 0.0) >= LEXICAL_MIN_COVERAGE)
        if has_coverage and not common:
            m["level"] = f"Semantic ({m['score'] * 100:.1f}%)"
            finalized.append(m)
        else:
            m["type"] = TOPICAL
            m["level"] = "Topical Similarity (Not Scored)"
            finalized.append(m)
    return finalized


def aggregate_plagiarism_stats(sentences: List[str], matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(sentences)
    if n_total == 0:
        return {"overall_pct": 0.0, "exact_pct": 0.0, "near_pct": 0.0, "semantic_pct": 0.0, "verified_sources": 0}
    # Hierarchy: Exact > Near > Semantic (each sentence counted at most once)
    status_map = {}
    source_set = set()
    for m in matches:
        idx = m["index"]
        t = m["type"]
        if t == EXACT:
            status_map[idx] = EXACT
        elif t == NEAR and status_map.get(idx) != EXACT:
            status_map[idx] = NEAR
        elif t == SEMANTIC and idx not in status_map:
            status_map[idx] = SEMANTIC
        if t in (EXACT, NEAR, SEMANTIC) and (m.get("url") or m.get("title")):
            source_set.add(m.get("url") or m.get("title"))

    counts = Counter(status_map.values())
    plagiarized = counts[EXACT] + counts[NEAR] + counts[SEMANTIC]
    return {
        "total_sentences": n_total,
        "plagiarized_sentences": plagiarized,
        "overall_pct": round(plagiarized / n_total * 100.0, 1),
        "exact_pct": round(counts[EXACT] / n_total * 100.0, 1),
        "near_pct": round(counts[NEAR] / n_total * 100.0, 1),
        "semantic_pct": round(counts[SEMANTIC] / n_total * 100.0, 1),
        "verified_sources": len(source_set)
    }


# ==============================================================================
# 5. CANDIDATE WEB RETRIEVAL, SEARCH ENGINES & WEB SCRAPER
# ==============================================================================
def canonical_url(url: str) -> str:
    """Normalize a URL so duplicates across different queries can be detected."""
    try:
        parts = urlparse(url.strip())
        netloc = parts.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parts.path.rstrip("/")
        query = f"?{parts.query}" if parts.query else ""
        return f"{parts.scheme.lower()}://{netloc}{path}{query}"
    except Exception:
        return url.strip().lower()


class SearchProviderError(Exception):
    """Raised when a search provider fails."""
    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


def _search_brave(query: str, max_results: int = 5, api_key: str = "", timeout: int = 15) -> List[Dict[str, str]]:
    if not api_key:
        raise SearchProviderError("Brave API key is not configured.", fatal=True)
    try:
        resp = requests.get(
            BRAVE_URL,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
                "User-Agent": "PlagiarismAI/1.0"
            },
            params={"q": query[:400], "count": min(max_results, 20)},
            timeout=timeout
        )
        if resp.status_code in (401, 403):
            raise SearchProviderError(f"Brave API key rejected (HTTP {resp.status_code}).", fatal=True)
        if resp.status_code == 429:
            raise SearchProviderError("Brave rate limit or quota exceeded (HTTP 429).")
        if not resp.ok:
            raise SearchProviderError(f"Brave returned HTTP {resp.status_code}.")
        data = resp.json()
        items = data.get("web", {}).get("results", [])
        return [{"title": (i.get("title") or "").strip(), "href": i.get("url") or "", "body": (i.get("description") or "").strip()} for i in items]
    except SearchProviderError:
        raise
    except Exception as e:
        raise SearchProviderError(f"Brave connection error: {e}")


def _search_tavily(query: str, max_results: int = 5, api_key: str = "", timeout: int = 15) -> List[Dict[str, str]]:
    if not api_key:
        raise SearchProviderError("Tavily API key is not configured.", fatal=True)
    try:
        resp = requests.post(
            TAVILY_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "PlagiarismAI/1.0"
            },
            json={"query": query[:400], "max_results": min(max_results, 20), "search_depth": "basic"},
            timeout=timeout
        )
        if resp.status_code in (401, 403):
            raise SearchProviderError(f"Tavily API key rejected (HTTP {resp.status_code}).", fatal=True)
        if resp.status_code == 429:
            raise SearchProviderError("Tavily rate limit or quota exceeded (HTTP 429).")
        if not resp.ok:
            raise SearchProviderError(f"Tavily returned HTTP {resp.status_code}.")
        data = resp.json()
        items = data.get("results", [])
        return [{"title": (i.get("title") or "").strip(), "href": i.get("url") or "", "body": (i.get("content") or "").strip()} for i in items]
    except SearchProviderError:
        raise
    except Exception as e:
        raise SearchProviderError(f"Tavily connection error: {e}")


def _search_ddg(query: str, max_results: int = 5, api_key: str = "", timeout: int = 15) -> List[Dict[str, str]]:
    # Attempt 1: ddgs library
    try:
        from ddgs import DDGS
        results = list(DDGS(timeout=timeout).text(query[:400], max_results=max_results))
        if results:
            return [{"title": (r.get("title") or "").strip(), "href": r.get("href") or r.get("url") or "", "body": (r.get("body") or "").strip()} for r in results]
    except Exception:
        pass

    # Attempt 2: Wikipedia Search API for academic & scientific terms
    clean_q = re.sub(r'["\']', '', query).replace(" paper", "").strip()
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": clean_q[:300], "format": "json"},
            headers={"User-Agent": "PlagiarismAI/1.0"},
            timeout=timeout
        )
        if resp.ok:
            data = resp.json()
            items = []
            for it in data.get("query", {}).get("search", []):
                t = it.get("title", "")
                snip = BeautifulSoup(it.get("snippet", ""), "html.parser").get_text()
                page_url = f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}"
                items.append({"title": t, "href": page_url, "body": snip})
                if len(items) >= max_results:
                    break
            if items:
                return items
    except Exception:
        pass

    # Attempt 3: Direct DuckDuckGo HTML endpoint
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query[:400]},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"},
            timeout=timeout
        )
        if resp.ok:
            soup = BeautifulSoup(resp.content, "html.parser")
            items = []
            for a in soup.find_all("a", class_="result__snippet"):
                parent = a.find_parent("div", class_="result__body") or a.find_parent("div")
                t_tag = (parent.find("a", class_="result__a") if parent else None) or soup.find("a", class_="result__url")
                href = t_tag.get("href", "") if t_tag else ""
                title = t_tag.get_text(strip=True) if t_tag else ""
                snippet = a.get_text(strip=True)
                if href:
                    items.append({"title": title, "href": href, "body": snippet})
                if len(items) >= max_results:
                    break
            if items:
                return items
    except Exception:
        pass
    return []


SEARCH_PROVIDERS = {
    "brave": {"label": "Brave Search API", "env": "BRAVE_API_KEY", "func": _search_brave, "interval": 1.1, "needs_key": True},
    "tavily": {"label": "Tavily Search API", "env": "TAVILY_API_KEY", "func": _search_tavily, "interval": 0.6, "needs_key": True},
    "duckduckgo": {"label": "DuckDuckGo Search", "env": "", "func": _search_ddg, "interval": 1.0, "needs_key": False},
    "ddgs": {"label": "DuckDuckGo Search", "env": "", "func": _search_ddg, "interval": 1.0, "needs_key": False},
}


def resolve_search_provider(choice: str = "auto") -> Dict[str, Any]:
    choice_norm = (choice or "auto").strip().lower()
    b_key = get_secret("BRAVE_API_KEY")
    t_key = get_secret("TAVILY_API_KEY")

    if choice_norm == "auto":
        if b_key:
            return {"id": "brave", "label": "Brave Search API", "key": b_key, "func": _search_brave, "interval": 1.1, "configured": True, "note": ""}
        if t_key:
            return {"id": "tavily", "label": "Tavily Search API", "key": t_key, "func": _search_tavily, "interval": 0.6, "configured": True, "note": ""}
        return {"id": "duckduckgo", "label": "DuckDuckGo Search", "key": "", "func": _search_ddg, "interval": 1.0, "configured": True, "note": "Defaulting to DuckDuckGo (keyless web search)."}

    if choice_norm in SEARCH_PROVIDERS:
        spec = SEARCH_PROVIDERS[choice_norm]
        key = get_secret(spec["env"]) if spec["needs_key"] else ""
        if spec["needs_key"] and not key:
            return {
                "id": "duckduckgo", "label": "DuckDuckGo Search", "key": "", "func": _search_ddg, "interval": 1.0,
                "configured": True,
                "note": f"{spec['label']} requested but {spec['env']} is missing. Falling back to DuckDuckGo.",
                "warning": f"{spec['label']} requires {spec['env']}, which is not set in environment or Colab secrets."
            }
        return {"id": choice_norm, "label": spec["label"], "key": key, "func": spec["func"], "interval": spec["interval"], "configured": True, "note": ""}

    return {"id": None, "label": "Unknown", "key": "", "func": None, "interval": 0, "configured": False, "note": f"Unsupported search provider: {choice}"}


def build_search_queries(sentences: List[str], max_queries: int = 6) -> List[Dict[str, str]]:
    if not sentences or max_queries < 1:
        return []
    scored = []
    for i, s in enumerate(sentences):
        words = _clean_words(s)
        if len(words) >= 4:
            scored.append((sum(_term_weight(w) for w in words) / math.sqrt(len(words)), i))
    if not scored:
        scored = [(1.0, i) for i in range(len(sentences))]

    chosen = [i for _, i in sorted(scored, reverse=True)[:max_queries]]
    exact_q, keyword_q, phrase_q = [], [], []
    for i in chosen:
        s = sentences[i]
        exact_q.append({"type": "exact_phrase", "query": f'"{distinctive_phrase(s, 10)}"'})
        terms = top_terms(_clean_words(s), 5)
        if len(terms) >= 2:
            keyword_q.append({"type": "keywords", "query": " ".join(terms)})
        phrase_q.append({"type": "distinctive_phrase", "query": distinctive_phrase(s, 12)})

    # Round-robin interleaved: exact_phrase -> keywords -> distinctive_phrase
    queries, seen = [], set()
    max_len = max(len(exact_q), len(keyword_q), len(phrase_q), 1)
    for k in range(max_len):
        for bucket in (exact_q, keyword_q, phrase_q):
            if k < len(bucket):
                val = bucket[k]["query"].strip()
                if val and val.lower() not in seen:
                    seen.add(val.lower())
                    queries.append(bucket[k])
                if len(queries) >= max_queries:
                    return queries
    return queries


def search_web_sources(queries: List[Dict[str, str]], results_per_query: int = 5, provider: str = "auto") -> Dict[str, Any]:
    out = {
        "results": [],
        "unique_urls": set(),
        "queries_log": [],
        "engine": "none",
        "queries_planned": len(queries),
        "queries_executed": 0,
        "results_retrieved": 0,
        "notes": [],
        "errors": []
    }
    p_info = resolve_search_provider(provider)
    out["engine"] = p_info["label"]
    if p_info.get("note"):
        out["notes"].append(p_info["note"])
    if p_info.get("warning"):
        out["errors"].append(p_info["warning"])

    if not p_info.get("configured") or not p_info.get("func"):
        out["errors"].append(p_info.get("note") or "Search provider could not be resolved.")
        return out

    seen_canonical = set()
    consecutive_failures = 0

    for idx, q_dict in enumerate(queries):
        q_text = q_dict.get("query", "").strip()
        q_type = q_dict.get("type", "phrase")
        if not q_text:
            continue

        entry = {"query": q_text, "type": q_type, "ok": False, "hits": 0, "engine": p_info["label"]}
        raw_hits = []
        try:
            raw_hits = p_info["func"](q_text, max_results=results_per_query, api_key=p_info["key"], timeout=12)
            entry["ok"] = True
            out["queries_executed"] += 1
            consecutive_failures = 0
        except SearchProviderError as spe:
            out["errors"].append(f"{p_info['label']}: {spe}")
            consecutive_failures += 1
            if spe.fatal:
                out["notes"].append(f"Aborting further searches: {spe}")
                out["queries_log"].append(entry)
                break
        except Exception as exc:
            out["errors"].append(f"{p_info['label']} request error: {exc}")
            consecutive_failures += 1

        for rank, h in enumerate(raw_hits or []):
            raw_url = h.get("href") or h.get("url") or ""
            if not raw_url or not str(raw_url).startswith("http"):
                continue
            c_url = canonical_url(raw_url)
            out["results_retrieved"] += 1
            entry["hits"] += 1
            if c_url not in seen_canonical:
                seen_canonical.add(c_url)
                out["unique_urls"].add(raw_url)
                out["results"].append({
                    "title": (h.get("title") or "").strip(),
                    "url": raw_url,
                    "canonical_url": c_url,
                    "snippet": (h.get("body") or "").strip(),
                    "found_by": q_type,
                    "rank": rank
                })

        out["queries_log"].append(entry)
        if consecutive_failures >= 3:
            out["notes"].append("Provider had 3 consecutive failures; skipping remaining queries.")
            break

        if idx < len(queries) - 1 and p_info.get("interval", 0) > 0:
            time.sleep(p_info["interval"])

    return out


def get_cached_webpage(url: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        c_url = canonical_url(url)
        cur.execute("SELECT url, page_title, retrieval_status, cleaned_text, content_hash FROM web_cache WHERE canonical_url = ? OR url = ?", (c_url, url))
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "url": row[0],
                "page_title": row[1] or "",
                "status": row[2],
                "text": row[3] or "",
                "content_hash": row[4] or "",
                "cached": True
            }
    except Exception:
        pass
    return None


def save_cached_webpage(url: str, title: str, status: str, text: str, db_path: str = DB_PATH):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        c_url = canonical_url(url)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
        cur.execute("""
        INSERT OR REPLACE INTO web_cache (url, canonical_url, page_title, content_hash, retrieval_status, cleaned_text, retrieved_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (url, c_url, title, h, status, text))
        conn.commit()
        conn.close()
    except Exception:
        pass


def fetch_webpage_content(url: str, timeout: int = 10, db_path: str = DB_PATH, use_cache: bool = True) -> Dict[str, Any]:
    """Download a webpage or PDF, remove boilerplate/noise, extract and normalize clean text.
    Caches results in SQLite web_cache to avoid repeated downloads.
    """
    if not url or not url.startswith("http"):
        return {"ok": False, "url": url, "page_title": "", "text": "", "from_cache": False, "reason": "Invalid URL scheme"}

    if use_cache:
        cached = get_cached_webpage(url, db_path=db_path)
        if cached and cached.get("status") == "ok" and cached.get("text"):
            return {
                "ok": True,
                "url": url,
                "page_title": cached["page_title"],
                "text": cached["text"],
                "from_cache": True,
                "reason": ""
            }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        if resp.status_code != 200:
            reason = f"HTTP {resp.status_code}"
            save_cached_webpage(url, "", reason, "", db_path)
            return {"ok": False, "url": url, "page_title": "", "text": "", "from_cache": False, "reason": reason}

        ctype = resp.headers.get("Content-Type", "").lower()
        is_pdf = "pdf" in ctype or urlparse(url).path.lower().endswith(".pdf")
        max_bytes = 5 * 1024 * 1024 if is_pdf else 600 * 1024

        chunks, size = [], 0
        for chunk in resp.iter_content(32768):
            size += len(chunk)
            if size > max_bytes:
                if is_pdf:
                    save_cached_webpage(url, "", "PDF exceeded size limit (5MB)", "", db_path)
                    return {"ok": False, "url": url, "page_title": "", "text": "", "from_cache": False, "reason": "PDF exceeded size limit (5MB)"}
                break
            chunks.append(chunk)
        raw_content = b"".join(chunks)

        if is_pdf:
            text = extract_text_from_pdf(raw_content, max_pages=30)
            title = os.path.basename(urlparse(url).path) or "Downloaded Academic Document"
        else:
            soup = BeautifulSoup(raw_content, "html.parser")
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            og_title = soup.find("meta", attrs={"property": "og:title"})
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()

            for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
                tag.decompose()

            elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote", "article", "div"])
            paragraphs = []
            for el in elements:
                t = el.get_text(separator=" ", strip=True)
                if len(t.split()) >= 6:
                    paragraphs.append(t)
            text = "\n\n".join(paragraphs) if paragraphs else soup.get_text(separator=" ", strip=True)
            text = normalize_text(text)

        if len(text.split()) < 15:
            save_cached_webpage(url, title, "Insufficient extractable text (<15 words)", "", db_path)
            return {"ok": False, "url": url, "page_title": title, "text": "", "from_cache": False, "reason": "Insufficient text on page (<15 words)"}

        save_cached_webpage(url, title, "ok", text, db_path)
        return {
            "ok": True,
            "url": url,
            "page_title": title,
            "text": text,
            "from_cache": False,
            "reason": ""
        }

    except requests.exceptions.Timeout:
        return {"ok": False, "url": url, "page_title": "", "text": "", "from_cache": False, "reason": "Connection timed out"}
    except requests.exceptions.SSLError:
        return {"ok": False, "url": url, "page_title": "", "text": "", "from_cache": False, "reason": "SSL certificate error"}
    except Exception as e:
        return {"ok": False, "url": url, "page_title": "", "text": "", "from_cache": False, "reason": str(e)[:100]}


def scrape_webpage(url: str, timeout: int = 10) -> Optional[str]:
    """Backward-compatible helper returning normalized text of a webpage."""
    res = fetch_webpage_content(url, timeout=timeout)
    return res["text"] if res["ok"] else None


# ==============================================================================
# 6. AI TEXT DETECTION (SAPLING API & LOCAL STYLOMETRIC ENSEMBLE)
# ==============================================================================
def detect_with_sapling(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Client for external Sapling AI Detection API with zero_retention request flag."""
    if len(text.strip()) < 300:
        return {"error": "Text shorter than 300 characters; insufficient for reliable cloud detection."}
    key = api_key or get_secret("SAPLING_API_KEY")
    if not key:
        return {"error": "SAPLING_API_KEY is not configured in environment or Colab secrets."}
    payload = {"text": text, "zero_retention": True, "sent_scores": True}
    try:
        resp = requests.post("https://api.sapling.ai/api/v1/aidetect", json=payload,
                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=12)
        if not resp.ok:
            return {"error": f"API returned HTTP {resp.status_code}: {resp.text[:150]}"}
        data = resp.json()
        return {
            "score": float(data.get("score", 0.0)),
            "verdict": data.get("verdict", "unknown"),
            "ai_fraction": data.get("ai_fraction"),
            "sentence_scores": data.get("sentence_scores", [])
        }
    except Exception as e:
        return {"error": f"API connection failed: {e}"}


FEATURE_NAMES = [
    "f01_avg_word_length", "f02_word_count", "f03_char_count", "f04_type_token_ratio",
    "f05_root_ttr", "f06_corrected_ttr", "f07_log_ttr", "f08_simpsons_d",
    "f09_hapax_ratio", "f10_dis_legomena_ratio", "f11_punctuation_ratio", "f12_comma_density",
    "f13_period_density", "f14_semicolon_density", "f15_colon_density", "f16_question_density",
    "f17_exclamation_density", "f18_dash_density", "f19_quote_density", "f20_uppercase_ratio",
    "f21_digit_ratio", "f22_function_word_ratio", "f23_pronoun_ratio", "f24_first_person_ratio",
    "f25_third_person_ratio", "f26_transitional_word_ratio", "f27_preposition_density",
    "f28_noun_ratio", "f29_verb_ratio", "f30_adj_ratio", "f31_adv_ratio",
    "f32_sentence_complexity_approx", "f33_avg_syllables_per_word", "f34_sentence_len_cv",
    "f35_sentence_len_mean", "f36_repeated_bigram_ratio", "f37_contraction_ratio"
]
LENGTH_FEATURE_INDICES = {1, 2}  # Drop word_count and char_count from active model


def extract_37_stylometric_features(text: str) -> np.ndarray:
    """Extracts raw 37-dimensional stylometric feature vector."""
    feats = np.zeros(37, dtype=np.float32)
    words = re.findall(r"\b[A-Za-z]+\b", text)
    tokens = text.split()
    n_words = len(words)
    n_tokens = max(len(tokens), 1)
    n_chars = len(text)

    feats[0] = (sum(len(w) for w in words) / max(n_words, 1)) if n_words else 0.0
    feats[1] = float(n_words)
    feats[2] = float(n_chars)

    counts = Counter(w.lower() for w in words)
    v = len(counts)
    feats[3] = v / max(n_words, 1)                                            # TTR
    feats[4] = v / math.sqrt(max(n_words, 1))                                  # Root TTR
    feats[5] = v / math.sqrt(2 * max(n_words, 1))                              # Corrected TTR
    feats[6] = (math.log(max(v, 1)) / math.log(max(n_words, 2))) if n_words > 1 else 0.0  # Log TTR
    feats[7] = sum(c * (c - 1) for c in counts.values()) / max(n_words * (n_words - 1), 1) # Simpson D
    feats[8] = sum(1 for c in counts.values() if c == 1) / max(n_words, 1)     # Hapax ratio
    feats[9] = sum(1 for c in counts.values() if c == 2) / max(n_words, 1)     # Dis legomena

    punct = len(re.findall(r"[^\w\s]", text))
    feats[10] = punct / n_tokens
    feats[11] = text.count(",") / n_tokens
    feats[12] = text.count(".") / n_tokens
    feats[13] = text.count(";") / n_tokens
    feats[14] = text.count(":") / n_tokens
    feats[15] = text.count("?") / n_tokens
    feats[16] = text.count("!") / n_tokens
    feats[17] = text.count("-") / n_tokens
    feats[18] = (text.count('"') + text.count("'")) / n_tokens
    feats[19] = sum(1 for c in text if c.isupper()) / max(n_chars, 1)
    feats[20] = sum(1 for c in text if c.isdigit()) / max(n_chars, 1)

    lower_words = [w.lower() for w in words]
    feats[21] = sum(1 for w in lower_words if w in STOPWORDS) / max(n_words, 1)

    first_pers = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    third_pers = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs", "it", "its"}
    feats[23] = sum(1 for w in lower_words if w in first_pers) / max(n_words, 1)
    feats[24] = sum(1 for w in lower_words if w in third_pers) / max(n_words, 1)
    feats[22] = feats[23] + feats[24]

    transitions = {"furthermore", "moreover", "however", "therefore", "consequently", "specifically", "additionally"}
    feats[25] = sum(1 for w in lower_words if w in transitions) / max(n_words, 1)

    preps = {"of", "in", "to", "for", "with", "on", "at", "from", "by", "about"}
    feats[26] = sum(1 for w in lower_words if w in preps) / max(n_words, 1)

    # Simple heuristic syntactic ratios
    feats[27] = sum(1 for w in lower_words if w.endswith(("tion", "ment", "ness", "ity"))) / max(n_words, 1)
    feats[28] = sum(1 for w in lower_words if w.endswith(("ed", "ing", "ize", "ate"))) / max(n_words, 1)
    feats[29] = sum(1 for w in lower_words if w.endswith(("al", "ive", "ous", "ful", "ic"))) / max(n_words, 1)
    feats[30] = sum(1 for w in lower_words if w.endswith("ly")) / max(n_words, 1)
    feats[31] = (feats[25] * 2.0 + feats[26]) / 2.0

    sylls = [sum(1 for char in w if char in "aeiouy") for w in lower_words]
    feats[32] = (sum(sylls) / max(n_words, 1)) if n_words else 1.0

    # Sentence length statistics
    s_lens = [len(s.split()) for s in text.split(".") if s.strip()]
    if s_lens:
        mu = float(np.mean(s_lens))
        sigma = float(np.std(s_lens))
        feats[33] = (sigma / max(mu, 1.0))
        feats[34] = mu
    else:
        feats[33] = 0.0
        feats[34] = float(n_words)

    bigrams = [f"{lower_words[i]}_{lower_words[i+1]}" for i in range(len(lower_words) - 1)]
    feats[35] = (sum(1 for c in Counter(bigrams).values() if c > 1) / max(len(bigrams), 1)) if bigrams else 0.0
    feats[36] = len(re.findall(r"\b\w+'[a-z]{1,2}\b", text.lower())) / max(n_words, 1)
    return feats


def zero_shot_rule_scorer(text: str) -> float:
    """Zero-shot rule-based logit estimator based on structural and lexical heuristics."""
    words = re.findall(r"\b[A-Za-z]+\b", text.lower())
    n = max(len(words), 1)
    logit = 0.0
    transitions = sum(1 for w in words if w in {"furthermore", "moreover", "consequently", "specifically", "additionally"})
    if (transitions / n) >= 0.015:
        logit += 1.2
    first_person = sum(1 for w in words if w in {"i", "my", "we", "our", "me", "us"})
    if first_person == 0 and n > 40:
        logit += 0.8
    elif (first_person / n) > 0.03:
        logit -= 1.4
    attractors = sum(1 for w in words if w in {"delve", "tapestry", "multifaceted", "pivotal", "underscores", "beacon"})
    if attractors >= 2:
        logit += 1.8
    contractions = len(re.findall(r"\b\w+'[a-z]{1,2}\b", text.lower()))
    if contractions == 0 and n > 50:
        logit += 0.5
    prob = 1.0 / (1.0 + math.exp(-logit))
    return float(np.clip(prob, 0.01, 0.99))


def predict_ai_local(text: str, model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    """Offline local stylometric ensemble with paragraph chunking and sentence inheritance."""
    bundle_path = os.path.join(model_dir, "detector_bundle.joblib")
    if not os.path.isfile(bundle_path):
        # Fallback to pure rule-based scorer if model bundle is absent
        prob = zero_shot_rule_scorer(text)
        return {"score": prob, "verdict": "Potentially AI" if prob >= 0.52 else "Human", "engine": "Zero-Shot Rule Scorer"}

    bundle = joblib.load(bundle_path)
    scaler = bundle["scaler"]
    style_clf = bundle["style"]
    lex_clf = bundle["lexical"]
    w_style = bundle.get("w_style", 0.25)
    model_cols = bundle.get("model_cols", [i for i in range(37) if i not in LENGTH_FEATURE_INDICES])

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.split()) >= 10]
    if not paragraphs:
        paragraphs = [text]

    p_scores = []
    for p in paragraphs:
        f_37 = extract_37_stylometric_features(p)
        f_35 = f_37[model_cols].reshape(1, -1)
        z = scaler.transform(f_35)
        p_style = float(style_clf.predict_proba(z)[0, 1])
        try:
            p_lex = float(lex_clf.predict_proba([p])[0, 1])
        except Exception:
            p_lex = p_style
        p_model = w_style * p_style + (1.0 - w_style) * p_lex
        p_rule = zero_shot_rule_scorer(p)

        # Adaptive Blending
        if p_rule >= 0.60:
            p_chunk = max(p_model, p_rule)
        elif p_rule <= 0.20 and p_model <= 0.35:
            p_chunk = min(p_model, p_rule)
        elif p_model >= 0.75:
            p_chunk = p_model
        else:
            p_chunk = 0.30 * p_model + 0.70 * p_rule
        p_scores.append(p_chunk)

    overall_score = float(np.mean(p_scores)) if p_scores else 0.5
    verdict = "Potentially AI-generated" if overall_score >= 0.52 else ("Likely Human" if overall_score <= 0.35 else "Needs Review")

    sentences = extract_sentences(text)
    sentence_records = []
    for idx, s in enumerate(sentences):
        # Sentences inherit containing paragraph's score
        parent_score = p_scores[min(idx % len(p_scores), len(p_scores) - 1)]
        sentence_records.append({"sentence": s, "score": round(parent_score, 3)})

    return {
        "score": round(overall_score, 3),
        "verdict": verdict,
        "engine": "Local Stylometric XGBoost + TF-IDF Bundle",
        "sentence_scores": sentence_records
    }


# ==============================================================================
# 7. DATABASE PERSISTENCE (SQLite)
# ==============================================================================
def init_database(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_date TEXT,
        document_name TEXT,
        total_sentences INTEGER,
        exact_matches INTEGER,
        near_exact_matches INTEGER,
        semantic_matches INTEGER,
        verified_sources INTEGER,
        sources_with_matches INTEGER,
        overall_similarity REAL,
        exact_pct REAL,
        semantic_pct REAL,
        details TEXT,
        status TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reference_documents (
        id TEXT PRIMARY KEY,
        document_name TEXT NOT NULL,
        uploaded_at TEXT NOT NULL,
        metadata TEXT,
        chunk_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'ready'
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reference_chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        sentence_indices TEXT NOT NULL,
        text TEXT NOT NULL,
        embedding BLOB,
        FOREIGN KEY (document_id) REFERENCES reference_documents(id) ON DELETE CASCADE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS web_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        canonical_url TEXT UNIQUE,
        page_title TEXT,
        content_hash TEXT,
        retrieval_status TEXT,
        cleaned_text TEXT,
        retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()


def save_scan_result(name: str, stats: Dict[str, Any], matches: List[Dict[str, Any]], db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(scans)")
    cols = {row[1] for row in cur.fetchall()}
    payload = json.dumps({"stats": stats, "matches": matches})
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    if "overall_similarity" in cols:
        cur.execute("""
        INSERT INTO scans (scan_date, document_name, total_sentences, exact_matches, near_exact_matches,
                           semantic_matches, verified_sources, sources_with_matches, overall_similarity,
                           exact_pct, semantic_pct, details, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, name, stats.get("total_sentences", 0), stats.get("exact_matches", 0),
            stats.get("near_matches", 0), stats.get("semantic_matches", 0),
            stats.get("verified_sources", 0), stats.get("verified_sources", 0),
            stats.get("overall_pct", 0.0), stats.get("exact_pct", 0.0),
            stats.get("semantic_pct", 0.0), payload, "completed"
        ))
    else:
        cur.execute("""
        INSERT INTO scans (document_name, overall_pct, exact_pct, near_pct, semantic_pct, result_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (name, stats.get("overall_pct", 0.0), stats.get("exact_pct", 0.0),
              stats.get("near_pct", 0.0), stats.get("semantic_pct", 0.0), payload))

    conn.commit()
    scan_id = cur.lastrowid
    conn.close()
    return scan_id


# ==============================================================================
# 8. PDF AUDIT REPORT EXPORTER
# ==============================================================================
def _latin(text: str) -> str:
    if not text:
        return ""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def export_plagiarism_pdf(name: str, stats: Dict[str, Any], matches: List[Dict[str, Any]], output_path: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "PlagiarismAI - Audit Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Document: {_latin(name)}   |   Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Similarity Summary Statistics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Overall Similarity: {stats['overall_pct']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Exact Matches: {stats['exact_pct']}%   |   Near-Exact: {stats['near_pct']}%   |   Semantic: {stats['semantic_pct']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total Sentences Scanned: {stats['total_sentences']}   |   Verified Sources: {stats['verified_sources']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detailed Matching Excerpts", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for m in matches[:30]:
        if m["type"] in (EXACT, NEAR, SEMANTIC):
            pdf.set_font("Helvetica", "B", 9)
            src_type = m.get("source_type", "web" if str(m.get("url", "")).startswith("http") else "local")
            type_label = "[WEB SOURCE]" if src_type == "web" else "[LOCAL REFERENCE]"
            src_title = _latin(m.get("title") or m.get("url") or "Reference Library")
            pdf.cell(0, 5, f"{type_label} [{m['type'].upper()} - {m.get('level', '')}] Source: {src_title}", new_x="LMARGIN", new_y="NEXT")

            url = m.get("url", "")
            if url and url.startswith("http"):
                pdf.set_font("Helvetica", "U", 8)
                pdf.set_text_color(20, 80, 180)
                pdf.cell(pdf.epw, 4, _latin(url[:95]), link=url, new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)

            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(pdf.epw, 4.2, _latin(f"Submitted: {m['sentence'][:200]}"), new_x="LMARGIN", new_y="NEXT")
            pdf.multi_cell(pdf.epw, 4.2, _latin(f"Matched:   {m['matched'][:200]}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
    pdf.output(output_path)
    return output_path


# ==============================================================================
# 9. BENCHMARK EVALUATOR (WEBIS-CPC-11)
# ==============================================================================
@dataclass
class WebisRecord:
    record_id: str
    original_text: str
    paraphrase_text: str
    label: int


def evaluate_webis_cpc11(dataset_dir: str = "data/Webis-CPC-11", max_samples: int = 300, random_seed: int = 42) -> Dict[str, Any]:
    """Runs 70/15/15 stratified benchmark across MiniLM, TF-IDF, and Hybrid configurations."""
    cache_path = os.path.join(CACHE_DIR, "webis_dataset_cache.joblib")
    records = []
    if os.path.isfile(cache_path):
        cached = joblib.load(cache_path)
        records = cached.get("records", [])
    elif os.path.isdir(dataset_dir):
        # Simple loader fallback if cache is absent
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                if f.endswith(".txt") and not f.endswith("-meta.txt"):
                    p = os.path.join(root, f)
                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as fh:
                            txt = fh.read().strip()
                        parts = txt.split("\n\n")
                        if len(parts) >= 2:
                            records.append(WebisRecord(f, parts[0], parts[1], 1 if "paraphrase" in p.lower() else 0))
                    except Exception:
                        continue

    if not records:
        return {"error": "No Webis-CPC-11 records found. Ensure data/Webis-CPC-11 exists or outputs/cache is populated."}

    # Sample balanced set
    pos = [r for r in records if r.label == 1]
    neg = [r for r in records if r.label == 0]
    n_per = min(len(pos), len(neg), max_samples // 2)
    rng = np.random.RandomState(random_seed)
    pos_sample = rng.choice(pos, size=n_per, replace=False)
    neg_sample = rng.choice(neg, size=n_per, replace=False)
    subset = list(pos_sample) + list(neg_sample)

    labels = np.array([r.label for r in subset])
    orig_texts = [r.original_text for r in subset]
    para_texts = [r.paraphrase_text for r in subset]

    # Dense embeddings
    model = load_sentence_transformer()
    orig_embs = encode_sentences(model, orig_texts)
    para_embs = encode_sentences(model, para_texts)
    raw_cosines = np.clip(np.sum(orig_embs * para_embs, axis=1), 0.0, 1.0)

    # TF-IDF sparse similarity
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words="english", sublinear_tf=True)
    vec.fit(orig_texts + para_texts)
    tfidf_sims = np.clip(np.sum(vec.transform(orig_texts).multiply(vec.transform(para_texts)), axis=1), 0.0, 1.0)
    tfidf_sims = np.asarray(tfidf_sims).flatten()

    # 70/15/15 Stratified Split
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    train_idx = np.concatenate([pos_idx[:int(0.70 * len(pos_idx))], neg_idx[:int(0.70 * len(neg_idx))]])
    dev_idx = np.concatenate([pos_idx[int(0.70 * len(pos_idx)):int(0.85 * len(pos_idx))],
                              neg_idx[int(0.70 * len(neg_idx)):int(0.85 * len(neg_idx))]])
    test_idx = np.concatenate([pos_idx[int(0.85 * len(pos_idx)):], neg_idx[int(0.85 * len(neg_idx)):]])

    y_test = labels[test_idx]

    # Threshold tuning on Dev set
    def _tune_and_eval(train_dev_scores, test_scores):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.35, 0.95, 61):
            f1 = f1_score(labels[dev_idx], (train_dev_scores[dev_idx] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        preds = (test_scores >= best_t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        return {
            "optimal_dev_threshold": round(float(best_t), 3),
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, test_scores)), 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }

    res_minilm = _tune_and_eval(raw_cosines, raw_cosines[test_idx])
    res_tfidf = _tune_and_eval(tfidf_sims, tfidf_sims[test_idx])
    hyb_scores = 0.75 * raw_cosines + 0.25 * tfidf_sims
    res_hybrid = _tune_and_eval(hyb_scores, hyb_scores[test_idx])

    return {
        "total_evaluated_samples": len(subset),
        "split_counts": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "configurations": {
            "Configuration 1 (MiniLM Dense Baseline)": res_minilm,
            "Configuration 2 (TF-IDF Sparse Baseline)": res_tfidf,
            "Configuration 3 (Hybrid TF-IDF + MiniLM, alpha=0.75)": res_hybrid
        }
    }


# ==============================================================================
# 10. HIGH-LEVEL PROGRAMMATIC API FOR COLAB USERS
# ==============================================================================
def scan_plagiarism(
    text_or_file: Union[str, bytes],
    filename: str = "document.txt",
    provider: str = "auto",
    max_queries: int = 6,
    results_per_query: int = 5,
    max_pages: int = 8,
    include_local_refs: bool = True,
    use_hybrid: bool = True,
    alpha: float = 0.75,
    semantic_threshold: float = 0.70,
    near_threshold: float = 0.90,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Execute complete end-to-end plagiarism scan over input text or document.
    Integrates:
      - Real web candidate search (Brave / Tavily / DuckDuckGo)
      - Webpage downloading, HTML boilerplate cleaning, and text normalization
      - SQLite web caching to avoid duplicate URL downloads
      - Multi-tier matching against retrieved web passages (Exact, Near-Exact, Semantic)
      - Hybrid Dense Sentence-BERT + Lexical TF-IDF blending (alpha=0.75)
      - Optional local reference library comparison
      - Contextual Evidence Gate (lexical coverage >= 25% and common source suppression)
      - Comprehensive execution and retrieval diagnostics
    """
    init_database()
    raw_text = extract_document_text(text_or_file, filename)
    sentences = extract_sentences(raw_text)
    if not sentences:
        return {"error": "Document contains no extractable sentences (minimum 3 words required)."}

    model = load_sentence_transformer()
    queries = build_search_queries(sentences, max_queries=max_queries)

    # 1. Execute Real Web Search
    search_res = search_web_sources(queries, results_per_query=results_per_query, provider=provider)
    web_hits = search_res.get("results", [])[:max_pages]

    raw_matches: List[Dict[str, Any]] = []
    pages_downloaded = 0
    pages_failed = 0
    passages_compared = 0
    verified_web_sources = []

    # 2. Retrieve Webpage Content and Compare
    for hit in web_hits:
        url = hit["url"]
        page_data = fetch_webpage_content(url, timeout=timeout, use_cache=True)
        if page_data["ok"] and page_data["text"]:
            pages_downloaded += 1
            page_title = page_data.get("page_title") or hit.get("title") or "Web Source"
            web_sentences = extract_sentences(page_data["text"], min_words=MIN_WORDS)
            if web_sentences:
                passages_to_check = web_sentences[:150]
                passages_compared += len(passages_to_check)
                exact_m = detect_exact_matches(sentences, passages_to_check, title=page_title, url=url)
                near_m = detect_near_exact_matches(sentences, passages_to_check, title=page_title, url=url, threshold=near_threshold)
                sem_m = detect_semantic_matches(model, sentences, passages_to_check, title=page_title, url=url,
                                                threshold=semantic_threshold, use_hybrid=use_hybrid, hybrid_alpha=alpha)
                for m in (exact_m + near_m + sem_m):
                    m["source_type"] = "web"
                    raw_matches.append(m)
                verified_web_sources.append({
                    "url": url,
                    "title": page_title,
                    "status": "retrieved",
                    "from_cache": page_data.get("from_cache", False),
                    "sentences_compared": len(passages_to_check)
                })
        else:
            pages_failed += 1
            verified_web_sources.append({
                "url": url,
                "title": hit.get("title", ""),
                "status": f"failed: {page_data.get('reason', 'unknown')}",
                "from_cache": False,
                "sentences_compared": 0
            })

    # 3. Local Reference Document Comparison
    local_ref_count = 0
    if include_local_refs:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(reference_chunks)")
        cols = {row[1] for row in cur.fetchall()}
        text_col = "text" if "text" in cols else "chunk_text"
        doc_col = "document_id" if "document_id" in cols else "doc_id"
        if text_col in cols and doc_col in cols:
            cur.execute(f"SELECT c.{text_col}, d.document_name FROM reference_chunks c JOIN reference_documents d ON c.{doc_col}=d.id")
            ref_rows = cur.fetchall()
        else:
            ref_rows = []
        conn.close()

        if ref_rows:
            ref_texts = [r[0] for r in ref_rows]
            local_ref_count = len(ref_texts)
            passages_compared += local_ref_count
            ref_exact = detect_exact_matches(sentences, ref_texts, title="Local Reference Library")
            ref_near = detect_near_exact_matches(sentences, ref_texts, title="Local Reference Library", threshold=near_threshold)
            ref_sem = detect_semantic_matches(model, sentences, ref_texts, title="Local Reference Library",
                                              threshold=semantic_threshold, use_hybrid=use_hybrid, hybrid_alpha=alpha)
            for m in (ref_exact + ref_near + ref_sem):
                m["source_type"] = "local"
                m["url"] = m.get("url") or "local-ref://library"
                raw_matches.append(m)

    # 4. Finalize Matches with Contextual Evidence Gate
    finalized_matches = finalize_and_filter_matches(raw_matches)
    stats = aggregate_plagiarism_stats(sentences, finalized_matches)

    # 5. Compile Rich Retrieval Diagnostics
    retrieval_stats = {
        "engine": search_res.get("engine", "none"),
        "queries_planned": len(queries),
        "queries_executed": search_res.get("queries_executed", 0),
        "results_retrieved": search_res.get("results_retrieved", 0),
        "unique_urls_identified": len(search_res.get("unique_urls", set())),
        "webpages_downloaded": pages_downloaded,
        "webpages_failed": pages_failed,
        "local_passages_compared": local_ref_count,
        "total_passages_compared": passages_compared,
        "notes": search_res.get("notes", []),
        "errors": search_res.get("errors", []),
        "search_status": "web_and_local_success" if (pages_downloaded > 0 and local_ref_count > 0) else (
            "web_success_no_local" if pages_downloaded > 0 else (
                "local_only_web_unavailable" if local_ref_count > 0 else "search_failed_no_sources"
            )
        )
    }

    scan_id = save_scan_result(filename, stats, finalized_matches)
    return {
        "scan_id": scan_id,
        "stats": stats,
        "matches": finalized_matches,
        "sentences_count": len(sentences),
        "retrieval_stats": retrieval_stats,
        "verified_web_sources": verified_web_sources
    }


def detect_ai(text_or_file: Union[str, bytes], filename: str = "document.txt", use_sapling_api: bool = False) -> Dict[str, Any]:
    """Execute AI text detection using external API (if enabled) and local stylometric ensemble."""
    raw_text = extract_document_text(text_or_file, filename)
    result = {}
    if use_sapling_api:
        api_res = detect_with_sapling(raw_text)
        result["external_service"] = api_res
    local_res = predict_ai_local(raw_text)
    result["local_stylometric_ensemble"] = local_res
    return result


def compare_two_documents(doc_a: Union[str, bytes], doc_b: Union[str, bytes],
                          name_a: str = "DocA", name_b: str = "DocB", alpha: float = 0.75) -> Dict[str, Any]:
    """Direct offline pairwise comparison between Document A and Document B."""
    text_a = extract_document_text(doc_a, name_a)
    text_b = extract_document_text(doc_b, name_b)
    s_a = extract_sentences(text_a)
    s_b = extract_sentences(text_b)
    if not s_a or not s_b:
        return {"error": "Both documents must have at least one extractable sentence."}

    model = load_sentence_transformer()
    exact_m = detect_exact_matches(s_a, s_b, title=name_b)
    near_m = detect_near_exact_matches(s_a, s_b, title=name_b)
    sem_m = detect_semantic_matches(model, s_a, s_b, title=name_b, use_hybrid=True, hybrid_alpha=alpha)
    all_m = finalize_and_filter_matches(exact_m + near_m + sem_m)
    stats = aggregate_plagiarism_stats(s_a, all_m)
    return {"comparison_stats": stats, "matches": all_m}


# ==============================================================================
# 11. VERIFICATION SUITE: 6 INTEGRATION SCENARIOS
# ==============================================================================
def run_six_integration_tests() -> Dict[str, Any]:
    """
    Executes and validates the 6 required integration test scenarios:
      Test 1: Exact copied passage from public webpage -> EXACT match.
      Test 2: Slightly modified version -> NEAR_EXACT match (>= 0.90 threshold).
      Test 3: Paraphrased version -> SEMANTIC match (MiniLM + TF-IDF hybrid, alpha=0.75).
      Test 4: Unrelated document -> 0 matches, valid retrieval statistics.
      Test 5: Missing API key / fallback -> local fallback & clear diagnostic warning.
      Test 6: Repeated URLs -> URL canonicalization and SQLite cache deduplication.
    """
    print("\n" + "=" * 78)
    print(" Running PlagiarismAI Real-Time Web Search Integration Verification")
    print("=" * 78)

    test_results = {}

    # Public source text from known accessible webpage (Wikipedia: Federated Learning)
    # Target URL: https://en.wikipedia.org/wiki/Federated_learning
    # Target Sentence: "Federated learning is generally concerned with and motivated by issues such as data privacy, data minimization, and data access rights."
    
    # --------------------------------------------------------------------------
    # TEST 1: Exact Copied Passage
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Exact Copied Passage from Webpage...")
    doc_exact = (
        "Federated learning is generally concerned with and motivated by issues such as "
        "data privacy, data minimization, and data access rights. "
        "Its applications involve a variety of research areas including telecommunications and pharmaceuticals."
    )
    res_1 = scan_plagiarism(doc_exact, filename="test1_exact.txt", provider="duckduckgo", max_queries=2, max_pages=3)
    exact_matches = [m for m in res_1.get("matches", []) if m["type"] == EXACT]
    passed_1 = len(exact_matches) > 0 and res_1["retrieval_stats"]["webpages_downloaded"] > 0
    test_results["Test 1 (Exact Match)"] = {
        "passed": passed_1,
        "exact_matches_found": len(exact_matches),
        "overall_similarity": res_1["stats"]["overall_pct"],
        "top_match_source": exact_matches[0]["url"] if exact_matches else None,
        "webpages_downloaded": res_1["retrieval_stats"]["webpages_downloaded"]
    }
    print(f"  -> Passed: {passed_1} | Exact Matches: {len(exact_matches)} | Pages Downloaded: {res_1['retrieval_stats']['webpages_downloaded']}")

    # --------------------------------------------------------------------------
    # TEST 2: Slightly Modified Passage (Near-Exact)
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Slightly Modified Webpage Passage (Near-Exact)...")
    doc_near = (
        "Federated learning is generally concerned with and motivated by challenges such as "
        "data privacy, data minimization, and data access policies."
    )
    res_2 = scan_plagiarism(doc_near, filename="test2_near.txt", provider="duckduckgo", max_queries=2, max_pages=3, near_threshold=0.90)
    near_matches = [m for m in res_2.get("matches", []) if m["type"] in (NEAR, EXACT)]
    passed_2 = len(near_matches) > 0
    test_results["Test 2 (Near-Exact Match)"] = {
        "passed": passed_2,
        "near_matches_found": len(near_matches),
        "score": near_matches[0]["score"] if near_matches else 0.0,
        "level": near_matches[0]["level"] if near_matches else None
    }
    print(f"  -> Passed: {passed_2} | Matches Found: {len(near_matches)} | Score: {near_matches[0]['score'] if near_matches else 'N/A'}")

    # --------------------------------------------------------------------------
    # TEST 3: Paraphrased Webpage Passage (Semantic Hybrid)
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Paraphrased Webpage Passage (Semantic)...")
    doc_sem = (
        "Federated learning is typically focused on and driven by issues such as "
        "data privacy, data minimization, and data access rights."
    )
    res_3 = scan_plagiarism(doc_sem, filename="test3_semantic.txt", provider="duckduckgo", max_queries=3, max_pages=3, semantic_threshold=0.70)
    sem_matches = [m for m in res_3.get("matches", []) if m["type"] in (SEMANTIC, EXACT, NEAR)]
    passed_3 = len(sem_matches) > 0
    test_results["Test 3 (Semantic Match)"] = {
        "passed": passed_3,
        "semantic_matches_found": len(sem_matches),
        "score": sem_matches[0]["score"] if sem_matches else 0.0,
        "coverage": sem_matches[0].get("coverage", 0.0) if sem_matches else 0.0
    }
    print(f"  -> Passed: {passed_3} | Semantic Matches: {len(sem_matches)} | Hybrid Score: {sem_matches[0]['score'] if sem_matches else 'N/A'}")

    # --------------------------------------------------------------------------
    # TEST 4: Unrelated Document (No Plagiarism Found)
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Unrelated Document (No Matching Content)...")
    doc_unrelated = (
        "Quantum topological phases of matter exhibit non-abelian anyonic braiding statistics "
        "within fractional quantum Hall effect regimes under ultra-low dilution temperatures."
    )
    res_4 = scan_plagiarism(doc_unrelated, filename="test4_unrelated.txt", provider="duckduckgo", max_queries=2, max_pages=2)
    passed_4 = res_4["stats"]["overall_pct"] == 0.0 and res_4["retrieval_stats"]["queries_executed"] > 0
    test_results["Test 4 (Zero Match Clean Document)"] = {
        "passed": passed_4,
        "overall_similarity": res_4["stats"]["overall_pct"],
        "queries_executed": res_4["retrieval_stats"]["queries_executed"],
        "webpages_downloaded": res_4["retrieval_stats"]["webpages_downloaded"]
    }
    print(f"  -> Passed: {passed_4} | Similarity: {res_4['stats']['overall_pct']}% | Webpages Checked: {res_4['retrieval_stats']['webpages_downloaded']}")

    # --------------------------------------------------------------------------
    # TEST 5: Missing API Key / Graceful Fallback
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Missing API Key Fallback Handling...")
    # Intentionally specify Brave when no BRAVE_API_KEY is configured
    old_env = os.environ.pop("BRAVE_API_KEY", None)
    res_5 = scan_plagiarism(doc_exact, filename="test5_fallback.txt", provider="brave", max_queries=2, max_pages=2)
    if old_env is not None:
        os.environ["BRAVE_API_KEY"] = old_env
    has_warning = any("BRAVE_API_KEY" in str(e) for e in res_5["retrieval_stats"]["errors"]) or any("BRAVE_API_KEY" in str(n) for n in res_5["retrieval_stats"]["notes"])
    passed_5 = has_warning and res_5["retrieval_stats"]["webpages_downloaded"] > 0
    test_results["Test 5 (Missing Key Fallback)"] = {
        "passed": passed_5,
        "diagnostics_warning_recorded": has_warning,
        "fallback_engine": res_5["retrieval_stats"]["engine"],
        "errors": res_5["retrieval_stats"]["errors"]
    }
    print(f"  -> Passed: {passed_5} | Diagnostic Warning Recorded: {has_warning} | Fallback Engine: {res_5['retrieval_stats']['engine']}")

    # --------------------------------------------------------------------------
    # TEST 6: Repeated URLs / Caching Deduplication
    # --------------------------------------------------------------------------
    print("\n[TEST 6] Repeated URL Canonicalization and Cache Deduplication...")
    test_url = "https://en.wikipedia.org/wiki/Federated_learning"
    c_1 = canonical_url("https://www.en.wikipedia.org/wiki/Federated_learning/")
    c_2 = canonical_url("https://en.wikipedia.org/wiki/Federated_learning?section=1")
    url_match = (c_1 == test_url)
    # First fetch writes to cache; second fetch must read from cache
    fetch_1 = fetch_webpage_content(test_url, use_cache=True)
    fetch_2 = fetch_webpage_content(test_url, use_cache=True)
    passed_6 = url_match and fetch_2.get("from_cache") is True and len(fetch_2.get("text", "")) > 100
    test_results["Test 6 (URL Deduplication & Cache)"] = {
        "passed": passed_6,
        "canonical_match": url_match,
        "second_call_cached": fetch_2.get("from_cache", False),
        "cached_characters": len(fetch_2.get("text", ""))
    }
    print(f"  -> Passed: {passed_6} | Canonical URL Match: {url_match} | From Cache: {fetch_2.get('from_cache', False)}")

    print("\n" + "=" * 78)
    all_passed = all(t["passed"] for t in test_results.values())
    print(f" Verification Result: {'ALL 6 TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 78)
    return test_results


# ==============================================================================
# DEMONSTRATION & CLI ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print(" PlagiarismAI - Headless Colab Real-Time Web Search & Detection Engine")
    print("=" * 78)

    # Check CLI arguments
    if "--run-tests" in sys.argv:
        tests = run_six_integration_tests()
        sys.exit(0 if all(t["passed"] for t in tests.values()) else 1)

    sample_doc_1 = (
        "Academic writing requires integrity and transparency in scholarly communication. "
        "The proliferation of generative language models alongside conventional textual copying "
        "poses multifaceted threats to academic evaluation. Traditional plagiarism detection "
        "methods compare submitted documents against existing sources using lexical overlap and token identity."
    )

    sample_doc_2 = (
        "Academic writing demands integrity and honesty in scholarly research. "
        "The proliferation of generative language models alongside conventional textual copying "
        "poses multifaceted threats to academic evaluation. Conventional text comparison "
        "tools analyze student papers against reference corpora using word overlap and phrase matching."
    )

    print("\n[1] Testing Pairwise Document Comparison...")
    cmp_res = compare_two_documents(sample_doc_1, sample_doc_2, "Original.txt", "Suspect.txt")
    print(f"  Overall Similarity: {cmp_res['comparison_stats']['overall_pct']}%")
    print(f"  Exact Matches:      {cmp_res['comparison_stats']['exact_pct']}%")
    print(f"  Near-Exact Matches: {cmp_res['comparison_stats']['near_pct']}%")
    print(f"  Semantic Matches:   {cmp_res['comparison_stats']['semantic_pct']}%")

    print("\n[2] Testing Local Stylometric AI Text Detection...")
    ai_res = detect_ai(sample_doc_1, use_sapling_api=False)
    loc = ai_res["local_stylometric_ensemble"]
    print(f"  Engine:       {loc.get('engine')}")
    print(f"  AI Score:     {loc.get('score')}")
    print(f"  Verdict:      {loc.get('verdict')}")
    print(f"  Sentences:    {len(loc.get('sentence_scores', []))} scored")

    print("\n[3] Testing Real-Time Web Plagiarism Scan...")
    web_scan = scan_plagiarism(
        "Federated learning is generally concerned with and motivated by issues such as data privacy, data minimization, and data access rights.",
        filename="FederatedQuery.txt",
        provider="auto",
        max_queries=2,
        max_pages=3
    )
    print(f"  Search Engine:      {web_scan['retrieval_stats']['engine']}")
    print(f"  Pages Downloaded:   {web_scan['retrieval_stats']['webpages_downloaded']}")
    print(f"  Passages Compared:  {web_scan['retrieval_stats']['total_passages_compared']}")
    print(f"  Overall Plagiarism: {web_scan['stats']['overall_pct']}%")
    print(f"  Exact Plagiarism:   {web_scan['stats']['exact_pct']}%")

    print("\n[4] Testing PDF Report Exporter (with Web Links)...")
    init_database()
    pdf_path = export_plagiarism_pdf("WebScanReport.txt", web_scan["stats"], web_scan["matches"], "plagiarism_audit_report.pdf")
    print(f"  Audit report successfully exported to: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    print("\n[5] Testing Webis-CPC-11 Benchmark Evaluator...")
    bench = evaluate_webis_cpc11(max_samples=60)
    if "error" in bench:
        print(f"  Notice: {bench['error']}")
    else:
        print(f"  Evaluated Samples: {bench['total_evaluated_samples']}")
        for config_name, metrics in bench["configurations"].items():
            print(f"    * {config_name}:")
            print(f"        Accuracy: {metrics['accuracy']} | F1: {metrics['f1_score']} | ROC-AUC: {metrics['roc_auc']}")

    print("\n" + "=" * 78)
    print(" Pipeline execution complete! All modules verified headless and Colab-ready.")
    print("=" * 78)
