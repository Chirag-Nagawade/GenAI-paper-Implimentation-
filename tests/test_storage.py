"""
Unit Tests for Storage Backend (SQLite Backend & Repository Fallback).
"""
import os
import tempfile
import unittest
import numpy as np

from storage.sqlite_backend import SQLiteStorageBackend
from storage.repository import get_storage


class TestStorageBackend(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.backend = SQLiteStorageBackend(db_path=self.temp_db.name)

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except OSError:
                pass

    def test_save_and_retrieve_scan(self):
        sample_result = {
            "document_name": "TestDoc",
            "scanned_at": "2026-02-01 10:00:00",
            "status": "matches",
            "stats": {
                "total_sentences": 5,
                "exact": 1,
                "near_exact": 1,
                "semantic": 1,
                "none": 2,
                "exact_pct": 40.0,
                "semantic_pct": 20.0,
                "overall_pct": 60.0,
                "verified_sources": 2,
                "sources_with_matches": 1
            },
            "matches": []
        }
        scan_id = self.backend.save_scan(sample_result)
        self.assertIsInstance(scan_id, int)

        history = self.backend.get_scan_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["Scan ID"], scan_id)
        self.assertEqual(history[0]["Document"], "TestDoc")
        self.assertEqual(history[0]["Overall %"], 60.0)

        details = self.backend.get_scan_details(scan_id)
        self.assertIsNotNone(details)
        self.assertEqual(details["document_name"], "TestDoc")

    def test_reference_documents_workflow(self):
        chunks = [
            {"chunk_id": 0, "sentence_indices": [0, 1], "text": "First chunk text", "embedding": [0.1, 0.2, 0.3]},
            {"chunk_id": 1, "sentence_indices": [2, 3], "text": "Second chunk text", "embedding": [0.4, 0.5, 0.6]}
        ]
        doc_id = self.backend.save_reference_document(
            document_name="RefPaper.pdf",
            metadata={"author": "Researcher A"},
            chunks=chunks
        )
        self.assertIsNotNone(doc_id)

        docs = self.backend.get_reference_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["document_name"], "RefPaper.pdf")
        self.assertEqual(docs[0]["chunk_count"], 2)

        retrieved_chunks = self.backend.get_reference_chunks(doc_id=doc_id)
        self.assertEqual(len(retrieved_chunks), 2)
        self.assertEqual(retrieved_chunks[0]["text"], "First chunk text")
        self.assertEqual(retrieved_chunks[0]["embedding"], [0.1, 0.2, 0.3])

        # Delete document
        deleted = self.backend.delete_reference_document(doc_id)
        self.assertTrue(deleted)
        self.assertEqual(len(self.backend.get_reference_documents()), 0)
        self.assertEqual(len(self.backend.get_reference_chunks(doc_id=doc_id)), 0)

    def test_ai_scan_persistence(self):
        sent_results = [
            {"sentence_index": 0, "category": "Likely Human", "prob_ai": 0.12, "confidence": 0.85, "sentence": "Hello world."}
        ]
        summary = {"ai_percentage": 0.0, "overall_classification": "Predominantly human-written text"}
        scan_id = self.backend.save_ai_scan("Essay.docx", sent_results, summary)
        self.assertIsNotNone(scan_id)

        details = self.backend.get_ai_scan_details(scan_id)
        self.assertIsNotNone(details)
        self.assertEqual(details["document_name"], "Essay.docx")
        self.assertEqual(len(details["sentences"]), 1)

    def test_evaluation_persistence(self):
        exp_data = {
            "experiment_id": "test_exp_001",
            "dataset": "Webis-CPC-11",
            "dataset_fingerprint": "abc12345",
            "total_records": 100,
            "total_time_seconds": 2.5,
            "exp1_minilm": {"accuracy": 0.85, "precision": 0.83, "recall": 0.88, "f1_score": 0.854, "roc_auc": 0.91},
            "exp3_hybrid": {"accuracy": 0.89, "precision": 0.87, "recall": 0.91, "f1_score": 0.889, "roc_auc": 0.94}
        }
        exp_id = self.backend.save_evaluation(exp_data)
        self.assertEqual(exp_id, "test_exp_001")

        evals = self.backend.get_evaluations()
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["experiment_id"], "test_exp_001")
        self.assertEqual(evals[0]["dataset_name"], "Webis-CPC-11")
        self.assertAlmostEqual(evals[0]["best_f1"], 0.889)

        details = self.backend.get_evaluation_details("test_exp_001")
        self.assertIsNotNone(details)
        self.assertEqual(details["total_records"], 100)
        self.assertIn("exp3_hybrid", details)


if __name__ == "__main__":
    unittest.main()
