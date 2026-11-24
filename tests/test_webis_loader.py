"""
Unit Tests for Webis-CPC-11 Dataset Loader.
"""
import os
import tempfile
import unittest
from plagiarism.webis_loader import parse_metadata_file, load_webis_dataset, WebisRecord


class TestWebisLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create a mock record: 100-original, 100-paraphrase, 100-metadata
        with open(os.path.join(self.temp_dir.name, "100-original.txt"), "w", encoding="utf-8") as f:
            f.write("Original passage text for record 100.")
        with open(os.path.join(self.temp_dir.name, "100-paraphrase.txt"), "w", encoding="utf-8") as f:
            f.write("Candidate paraphrase text for record 100.")
        with open(os.path.join(self.temp_dir.name, "100-metadata.txt"), "w", encoding="utf-8") as f:
            f.write("HITId: hit_123\nWorkerId: w_456\nWorkTimeInSeconds: 45\nParaphrase: Yes\n")

        # Create a mock non-paraphrase record: 101
        with open(os.path.join(self.temp_dir.name, "101-original.txt"), "w", encoding="utf-8") as f:
            f.write("Original passage text for record 101.")
        with open(os.path.join(self.temp_dir.name, "101-paraphrase.txt"), "w", encoding="utf-8") as f:
            f.write("Candidate non-paraphrase text for record 101.")
        with open(os.path.join(self.temp_dir.name, "101-metadata.txt"), "w", encoding="utf-8") as f:
            f.write("HITId: hit_789\nWorkerId: w_456\nWorkTimeInSeconds: 30\nParaphrase: No\n")

        # Create an invalid record missing original text: 102
        with open(os.path.join(self.temp_dir.name, "102-metadata.txt"), "w", encoding="utf-8") as f:
            f.write("HITId: hit_999\nParaphrase: Yes\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_metadata_file(self):
        meta_path = os.path.join(self.temp_dir.name, "100-metadata.txt")
        meta = parse_metadata_file(meta_path)
        self.assertEqual(meta.get("HITId"), "hit_123")
        self.assertEqual(meta.get("Paraphrase"), "Yes")

    def test_load_webis_dataset(self):
        records, stats = load_webis_dataset(self.temp_dir.name, verbose=False)
        self.assertEqual(stats["valid_records"], 2)
        self.assertEqual(stats["positive_labels"], 1)
        self.assertEqual(stats["negative_labels"], 1)
        self.assertEqual(stats["missing_file_count"], 1)

        self.assertEqual(records[0].record_id, "100")
        self.assertEqual(records[0].label, 1)
        self.assertEqual(records[1].record_id, "101")
        self.assertEqual(records[1].label, 0)

    def test_stratified_sampling_proportion(self):
        """Verify that _sample_records preserves class balance."""
        from plagiarism.webis_loader import _sample_records

        # Create 100 mock records: 60 positive, 40 negative
        pool = [
            WebisRecord(f"pos_{i}", f"orig {i}", f"para {i}", 1) for i in range(60)
        ] + [
            WebisRecord(f"neg_{i}", f"orig {i}", f"para {i}", 0) for i in range(40)
        ]
        sampled = _sample_records(pool, max_records=20, stratified=True, random_seed=42)
        self.assertEqual(len(sampled), 20)
        pos = sum(1 for r in sampled if r.label == 1)
        neg = sum(1 for r in sampled if r.label == 0)
        # 60% of 20 = 12 pos, 8 neg
        self.assertEqual(pos, 12)
        self.assertEqual(neg, 8)


if __name__ == "__main__":
    unittest.main()
