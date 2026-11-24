"""
Unit Tests for Sliding-Window Chunking and Mapping.
"""
import unittest
from unittest.mock import patch
import numpy as np
from plagiarism.chunking import create_sliding_windows, map_window_matches_to_sentences


class TestSlidingWindowChunking(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            f"Sentence {i} with distinct technical content and descriptive words."
            for i in range(10)
        ]

    def test_short_document(self):
        short = ["Only one sentence."]
        chunks = create_sliding_windows(short, window_size=3, stride=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].sentence_indices, [0])
        self.assertEqual(chunks[0].text, "Only one sentence.")

    def test_window_overlap_and_strides(self):
        chunks = create_sliding_windows(self.sentences, window_size=3, stride=1)
        self.assertEqual(len(chunks), 8)
        self.assertEqual(chunks[0].sentence_indices, [0, 1, 2])
        self.assertEqual(chunks[1].sentence_indices, [1, 2, 3])
        self.assertEqual(chunks[-1].sentence_indices, [7, 8, 9])

    def test_stride_greater_than_one(self):
        chunks = create_sliding_windows(self.sentences, window_size=3, stride=2)
        self.assertEqual(chunks[0].sentence_indices, [0, 1, 2])
        self.assertEqual(chunks[1].sentence_indices, [2, 3, 4])

    def test_long_individual_sentence_bound(self):
        long_sent = "word " * 250
        chunks = create_sliding_windows([long_sent, "Normal sentence."], window_size=3, max_words_per_window=100)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].sentence_indices, [0])

    def test_window_match_mapping(self):
        chunks = create_sliding_windows(self.sentences, window_size=3, stride=1)
        window_matches = [
            {
                "chunk_id": 0,
                "type": "Semantic Match",
                "score": 85.0,
                "sentence": chunks[0].text,
                "matched": "Source passage text",
                "title": "Test Source",
                "url": "https://example.com/source1"
            },
            {
                "chunk_id": 1,
                "type": "Semantic Match",
                "score": 92.0,  # higher score overlapping on sentence 1 and 2
                "sentence": chunks[1].text,
                "matched": "Source passage text higher",
                "title": "Test Source",
                "url": "https://example.com/source1"
            }
        ]
        mapped = map_window_matches_to_sentences(window_matches, len(self.sentences), chunks)
        mapped_dict = {m["index"]: m for m in mapped}

        # Sentence 0 covered by chunk 0
        self.assertIn(0, mapped_dict)
        self.assertEqual(mapped_dict[0]["score"], 85.0)

        # Sentence 1 covered by chunk 0 (85) and chunk 1 (92) -> should pick 92
        self.assertIn(1, mapped_dict)
        self.assertEqual(mapped_dict[1]["score"], 92.0)

        # Sentence 2 covered by chunk 0 (85) and chunk 1 (92) -> should pick 92
        self.assertIn(2, mapped_dict)
        self.assertEqual(mapped_dict[2]["score"], 92.0)

    def test_sliding_window_fallback_logging(self):
        """Verify that sliding window exceptions log a warning and record diagnostic info without crashing."""
        import app

        user_sents = ["This user paper studies neural learning and algorithmic acceleration models."]
        source_sents = [
            "A related study discusses statistical approaches in computer science.",
            "Another sentence here for context.",
            "Third sentence in the source document."
        ]
        user_emb = np.ones((1, 384), dtype=np.float32)
        source_emb = np.ones((3, 384), dtype=np.float32)
        diagnostics = []

        # Patch create_sliding_windows in app to simulate an unexpected error during window chunking
        with patch("app.create_sliding_windows", side_effect=RuntimeError("Simulated sliding window failure")):
            with self.assertLogs(level="WARNING") as cm:
                matches = app.match_against_source(
                    model=None,
                    sentences=user_sents,
                    user_embeddings=user_emb,
                    source_sentences=source_sents,
                    title="Test Source",
                    url="https://example.com/test",
                    semantic_threshold=0.7,
                    near_threshold=0.9,
                    source_embeddings=source_emb,
                    use_chunking=True,
                    window_size=3,
                    diagnostics=diagnostics
                )

        # 1. Fallback matching executed cleanly without raising exception
        self.assertIsInstance(matches, list)

        # 2. Diagnostic message was recorded in diagnostics list
        self.assertTrue(any("Fallback to standard sentence matching was applied" in d for d in diagnostics))
        self.assertTrue(any("Simulated sliding window failure" in d for d in diagnostics))

        # 3. Warning was logged to standard logger
        self.assertTrue(any("Sliding-window candidate generation failed" in output for output in cm.output))


if __name__ == "__main__":
    unittest.main()
