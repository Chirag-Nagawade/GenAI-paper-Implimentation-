"""
Unit and Integration Tests for Local Reference Database Matching & Embedding Reuse.
"""
import os
import tempfile
import unittest
from unittest import mock
import numpy as np

import app
from plagiarism.reference_search import (
    index_reference_document,
    search_reference_database,
    list_reference_documents,
    delete_reference_document,
    compute_content_hash
)
from storage.sqlite_backend import SQLiteStorageBackend


class DummyEmbeddingModel:
    """Mock embedding model that tracks how many times encode is invoked."""
    def __init__(self, dim=384):
        self.dim = dim
        self.call_count = 0

    def encode(self, sentences, **kwargs):
        self.call_count += 1
        rng = np.random.RandomState(42)
        # Produce normalized vectors
        vecs = rng.randn(len(sentences), self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


class TestLocalReferenceMatching(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.backend = SQLiteStorageBackend(db_path=self.temp_db.name)
        self.backend.init()
        self.model = DummyEmbeddingModel()

        self.sample_text = (
            "Photosynthesis converts sunlight into glucose in green plants. "
            "Chloroplasts contain chlorophyll pigments that absorb light. "
            "Carbon dioxide and water react to form organic carbohydrates. "
            "Oxygen is released as a metabolic byproduct into the atmosphere."
        )

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except OSError:
                pass

    def test_indexing_and_duplicate_prevention(self):
        # Index document 1
        res1 = index_reference_document(
            model=self.model,
            document_name="biology_reference.txt",
            text=self.sample_text,
            storage_backend=self.backend
        )
        self.assertTrue(res1["ok"])
        self.assertIn("doc_id", res1)
        self.assertGreater(res1["total_chunks"], 0)

        # Attempt duplicate upload with identical content
        res2 = index_reference_document(
            model=self.model,
            document_name="duplicate_reference.txt",
            text=self.sample_text,
            storage_backend=self.backend
        )
        self.assertFalse(res2["ok"])
        self.assertIn("Duplicate", res2["reason"])

        # Attempt duplicate upload with identical name
        res3 = index_reference_document(
            model=self.model,
            document_name="biology_reference.txt",
            text="Different text completely about computer operating systems.",
            storage_backend=self.backend
        )
        self.assertFalse(res3["ok"])
        self.assertIn("Duplicate", res3["reason"])

    def test_stored_embedding_reuse_without_reencoding(self):
        # 1. Index document
        index_reference_document(
            model=self.model,
            document_name="biology_reference.txt",
            text=self.sample_text,
            storage_backend=self.backend
        )
        initial_encodes = self.model.call_count
        self.assertGreater(initial_encodes, 0)

        # 2. Search against reference database
        user_sentences = [
            "Photosynthesis converts sunlight into glucose in green plants.",
            "Robotics relies on actuator motion control algorithms."
        ]
        user_embs = self.model.encode(user_sentences)
        encodes_after_user = self.model.call_count

        # Execute search_reference_database
        matches = search_reference_database(
            model=self.model,
            sentences=user_sentences,
            user_embeddings=user_embs,
            storage_backend=self.backend,
            threshold=0.60,
            near=0.90,
            use_hybrid=True,
            hybrid_alpha=0.75
        )

        # Crucial check: model.encode MUST NOT be called for stored reference chunks
        # Call count should remain exactly equal to encodes_after_user
        self.assertEqual(
            self.model.call_count,
            encodes_after_user,
            "Stored reference embeddings should be reused directly from the database without re-encoding!"
        )

    def test_overlapping_match_deduplication_and_sentence_positions(self):
        # Index document
        index_reference_document(
            model=self.model,
            document_name="biology_reference.txt",
            text=self.sample_text,
            storage_backend=self.backend
        )

        user_sentences = [
            "Photosynthesis converts sunlight into glucose in green plants.",
            "Oxygen is released as a metabolic byproduct into the atmosphere."
        ]
        user_embs = self.model.encode(user_sentences)

        matches = search_reference_database(
            model=self.model,
            sentences=user_sentences,
            user_embeddings=user_embs,
            storage_backend=self.backend,
            threshold=0.50,
            near=0.90
        )

        # Check sentence index preservation and deduplication
        indices_matched = [m["index"] for m in matches]
        # Should contain at most one match per sentence per document
        self.assertEqual(len(indices_matched), len(set(indices_matched)))
        for m in matches:
            self.assertIn(m["index"], [0, 1])
            self.assertIn("local-ref://biology_reference.txt", m["url"])
            self.assertIn("[Local Ref] biology_reference.txt", m["title"])

    def test_run_scan_local_reference_integration(self):
        # Index reference document
        index_reference_document(
            model=self.model,
            document_name="biology_reference.txt",
            text=self.sample_text,
            storage_backend=self.backend
        )

        user_text = (
            "Photosynthesis converts sunlight into glucose in green plants. "
            "Chloroplasts contain chlorophyll pigments that absorb light."
        )

        settings = {
            "provider": "auto",
            "semantic_threshold": 0.50,
            "near_threshold": 0.90,
            "max_queries": 2,
            "results_per_query": 2,
            "max_pages": 2,
            "max_sentences": 50,
            "include_local_refs": True,
            "use_hybrid": False,
            "use_chunking": False
        }

        # Mock web search to return empty so only local reference matches
        empty_search = {"results": [], "errors": [], "notes": [], "engines": ["fake"], "provider": "fake",
                        "provider_label": "Fake search", "queries": []}

        with mock.patch.object(app, "search_web_sources", return_value=empty_search), \
             mock.patch.object(app, "get_storage", return_value=self.backend):
            result = app.run_scan(self.model, user_text, "UserDoc.txt", settings, lambda f, m: None)

        self.assertIn(result["status"], ["matches", "no_matches"])
        # Sources list should contain the local reference entry
        local_sources = [s for s in result["sources"] if s.get("kind") == "local_reference"]
        self.assertEqual(len(local_sources), 1)
        self.assertEqual(local_sources[0]["title"], "[Local Ref] biology_reference.txt")


if __name__ == "__main__":
    unittest.main()
