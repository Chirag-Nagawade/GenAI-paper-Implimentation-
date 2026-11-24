"""
Hybrid TF-IDF + MiniLM Similarity Module.

Combines dense semantic representations (all-MiniLM-L6-v2) with sparse lexical
representations (TF-IDF with n-grams) during candidate generation, while keeping
the contextual Evidence Gate strictly separate as an independent verification stage.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridSimilarityScorer:
    """
    Computes hybrid similarity matrix:
        S_hybrid = alpha * S_semantic + (1 - alpha) * S_tfidf
    """

    def __init__(
        self,
        alpha: float = 0.75,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 15000,
        stop_words: Optional[str] = "english"
    ):
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words=self.stop_words
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "HybridSimilarityScorer":
        """Fit TF-IDF vocabulary on background/reference corpus."""
        valid_texts = [t for t in texts if t and t.strip()]
        if valid_texts:
            self.vectorizer.fit(valid_texts)
            self._fitted = True
        return self

    def compute_hybrid_similarity(
        self,
        user_embeddings: np.ndarray,
        source_embeddings: np.ndarray,
        user_texts: List[str],
        source_texts: List[str],
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute pairwise hybrid similarity matrix between user queries and source passages.
        
        Returns:
            np.ndarray of shape (len(user_texts), len(source_texts))
        """
        if alpha is None:
            alpha = self.alpha

        # 1. Semantic Cosine Similarity (Dense MiniLM)
        # Assumes normalized or computes standard cosine similarity
        if user_embeddings.ndim == 1:
            user_embeddings = user_embeddings.reshape(1, -1)
        if source_embeddings.ndim == 1:
            source_embeddings = source_embeddings.reshape(1, -1)

        sim_semantic = cosine_similarity(user_embeddings, source_embeddings)
        sim_semantic = np.clip(sim_semantic, 0.0, 1.0)

        # 2. Lexical Cosine Similarity (Sparse TF-IDF)
        combined_texts = user_texts + source_texts
        try:
            # Fit-transform dynamically on current comparison pair context
            tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            all_matrix = tfidf_vec.fit_transform(combined_texts)
            u_matrix = all_matrix[:len(user_texts)]
            s_matrix = all_matrix[len(user_texts):]
            sim_tfidf = (u_matrix.dot(s_matrix.T)).toarray()
            sim_tfidf = np.clip(sim_tfidf, 0.0, 1.0)
        except ValueError:
            # Fallback if text has no tokens or empty
            sim_tfidf = np.zeros_like(sim_semantic)

        # 3. Hybrid Blend
        sim_hybrid = alpha * sim_semantic + (1.0 - alpha) * sim_tfidf
        return np.clip(sim_hybrid, 0.0, 1.0)
