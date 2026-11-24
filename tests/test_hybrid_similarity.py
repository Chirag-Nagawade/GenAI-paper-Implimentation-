"""
Unit Tests for Hybrid TF-IDF + MiniLM Similarity.
"""
import unittest
import numpy as np
from plagiarism.hybrid_similarity import HybridSimilarityScorer


class TestHybridSimilarity(unittest.TestCase):
    def setUp(self):
        self.scorer = HybridSimilarityScorer(alpha=0.7)
        self.user_texts = [
            "Photosynthesis converts sunlight and water into glucose.",
            "Database management systems store and organize records efficiently."
        ]
        self.source_texts = [
            "Plants use sunlight and water in photosynthesis to generate glucose.",
            "Astronomers detected distant pulsar radio frequencies."
        ]
        # Synthetic unit embeddings (dim 4)
        self.user_embs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        self.source_embs = np.array([
            [0.9, 0.1, 0.0, 0.0],  # similar to user 0
            [0.0, 0.0, 1.0, 0.0]   # unrelated
        ], dtype=np.float32)

    def test_hybrid_computation_shape(self):
        sim = self.scorer.compute_hybrid_similarity(
            self.user_embs, self.source_embs,
            self.user_texts, self.source_texts
        )
        self.assertEqual(sim.shape, (2, 2))
        self.assertTrue(np.all(sim >= 0.0))
        self.assertTrue(np.all(sim <= 1.0))

    def test_hybrid_ranking_agreement(self):
        sim = self.scorer.compute_hybrid_similarity(
            self.user_embs, self.source_embs,
            self.user_texts, self.source_texts,
            alpha=0.7
        )
        # Pair (0, 0) should be substantially higher than (0, 1)
        self.assertGreater(sim[0, 0], sim[0, 1])
        self.assertGreater(sim[0, 0], 0.6)

    def test_alpha_weighting_extreme(self):
        # Pure semantic
        sim_sem = self.scorer.compute_hybrid_similarity(
            self.user_embs, self.source_embs,
            self.user_texts, self.source_texts,
            alpha=1.0
        )
        from sklearn.metrics.pairwise import cosine_similarity
        expected = cosine_similarity(self.user_embs, self.source_embs)[0, 0]
        self.assertAlmostEqual(sim_sem[0, 0], expected, places=3)


if __name__ == "__main__":
    unittest.main()
