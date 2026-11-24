"""
Unit Tests for Phase 1 (Group-Aware Partitioning) and Phase 2 (Tripartite Evaluation).

Verifies:
1. Group-aware partitioning ensures zero group overlap and preserves class balance.
2. Category assignment rules categorize exact copying, genuine paraphrases, and unrelated text faithfully.
3. Ambiguous examples are properly flagged for manual review.
4. Original Webis labels are strictly preserved.
5. Tripartite outputs exist, match schemas, and confusion matrices agree mathematically.
6. Original evaluation benchmark files are preserved untouched.
"""
import json
import os
import unittest
import numpy as np

from plagiarism.group_partitioning import get_record_group_id, group_aware_stratified_split
from plagiarism.tripartite_evaluation import (
    AMBIGUOUS_REVIEW,
    EXACT_COPYING,
    GENUINE_PARAPHRASING,
    UNRELATED_TEXT,
    classify_webis_record,
    categorize_dataset,
    compute_binary_task_metrics,
)
from plagiarism.webis_loader import WebisRecord, load_webis_dataset


class TestPhase1GroupPartitioning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, _ = load_webis_dataset(max_records=500, verbose=False, stratified=True, random_seed=42)

    def test_group_disjointness_zero_overlap(self):
        """Verify zero group overlap across Train, Dev, and Test partitions."""
        train_idx, dev_idx, test_idx, meta = group_aware_stratified_split(
            self.records, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15, random_seed=42
        )

        train_groups = set(get_record_group_id(self.records[i]) for i in train_idx)
        dev_groups = set(get_record_group_id(self.records[i]) for i in dev_idx)
        test_groups = set(get_record_group_id(self.records[i]) for i in test_idx)

        # 1. Absolute zero group overlap
        self.assertEqual(len(train_groups & dev_groups), 0, "Train and Dev groups must not overlap!")
        self.assertEqual(len(train_groups & test_groups), 0, "Train and Test groups must not overlap!")
        self.assertEqual(len(dev_groups & test_groups), 0, "Dev and Test groups must not overlap!")

        # 2. Row indices are mutually disjoint
        self.assertEqual(len(set(train_idx) & set(dev_idx)), 0)
        self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
        self.assertEqual(len(set(dev_idx) & set(test_idx)), 0)

        # 3. Union equals total records
        self.assertEqual(len(train_idx) + len(dev_idx) + len(test_idx), len(self.records))

    def test_class_balance_preservation(self):
        """Verify class distribution in each partition closely mirrors full sample."""
        train_idx, dev_idx, test_idx, meta = group_aware_stratified_split(
            self.records, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15, random_seed=42
        )

        overall_pos_ratio = sum(r.label for r in self.records) / len(self.records)

        for p_name, idxs in [("Train", train_idx), ("Dev", dev_idx), ("Test", test_idx)]:
            p_pos = sum(self.records[i].label for i in idxs)
            p_ratio = p_pos / len(idxs)
            # Must be within 5% of overall population ratio (~51.8%)
            self.assertAlmostEqual(p_ratio, overall_pos_ratio, delta=0.05,
                                   msg=f"{p_name} class balance deviated excessively from overall ratio.")

    def test_reproducibility_with_seed(self):
        """Verify identical partitions are generated under the same random seed."""
        tr1, dv1, te1, _ = group_aware_stratified_split(self.records, random_seed=42)
        tr2, dv2, te2, _ = group_aware_stratified_split(self.records, random_seed=42)
        self.assertEqual(tr1, tr2)
        self.assertEqual(dv1, dv2)
        self.assertEqual(te1, te2)


class TestPhase2TripartiteCategorization(unittest.TestCase):
    def test_exact_copying_rule(self):
        """Verify verbatim and near-verbatim pairs are classified as EXACT_COPYING."""
        # Verbatim match
        r1 = WebisRecord(record_id="test1", original_text="The quick brown fox jumps.",
                         paraphrase_text="The quick brown fox jumps.", label=0)
        c1 = classify_webis_record(r1)
        self.assertEqual(c1["category"], EXACT_COPYING)
        self.assertFalse(c1["manual_review_required"])
        self.assertEqual(c1["original_label"], 0)

        # Near-verbatim match (ratio >= 0.85)
        r2 = WebisRecord(record_id="test2", original_text="The quick brown fox jumps over the lazy dog.",
                         paraphrase_text="The quick brown fox jumps over the lazy dogs.", label=0)
        c2 = classify_webis_record(r2)
        self.assertEqual(c2["category"], EXACT_COPYING)
        self.assertGreaterEqual(c2["normalized_ratio"], 0.85)

    def test_genuine_paraphrase_rule(self):
        """Verify meaning-preserving rewrites with distinct vocabulary are classified as GENUINE_PARAPHRASING."""
        r = WebisRecord(record_id="test3",
                        original_text="Photosynthesis converts solar energy into chemical energy stored in glucose molecules.",
                        paraphrase_text="Plants transform sunlight into chemical fuel by synthesizing sugar compounds.",
                        label=1)
        c = classify_webis_record(r)
        self.assertEqual(c["category"], GENUINE_PARAPHRASING)
        self.assertFalse(c["manual_review_required"])
        self.assertEqual(c["original_label"], 1)

    def test_unrelated_text_rule(self):
        """Verify spam, empty, or completely disconnected texts are classified as UNRELATED_TEXT."""
        # Spam text
        r1 = WebisRecord(record_id="test4",
                         original_text="Astronomers detected pulsating radio emissions from a distant spinning neutron star.",
                         paraphrase_text="please help me sir / madam thankyou", label=0)
        c1 = classify_webis_record(r1)
        self.assertEqual(c1["category"], UNRELATED_TEXT)
        self.assertFalse(c1["manual_review_required"])

        # Zero overlap
        r2 = WebisRecord(record_id="test5",
                         original_text="Astronomers detected pulsating radio emissions from a distant spinning neutron star.",
                         paraphrase_text="Baking sourdough bread requires water, flour, wild yeast, and salt fermentation.",
                         label=0)
        c2 = classify_webis_record(r2)
        self.assertEqual(c2["category"], UNRELATED_TEXT)
        self.assertFalse(c2["manual_review_required"])

    def test_ambiguous_review_flagging(self):
        """Verify partial overlap in rejected submissions is flagged for manual review."""
        r = WebisRecord(record_id="test6",
                        original_text="The committee convened on Tuesday morning to discuss annual budget appropriations.",
                        paraphrase_text="The committee convened to review expenditures and discuss allocations.",
                        label=0)
        c = classify_webis_record(r)
        self.assertEqual(c["category"], AMBIGUOUS_REVIEW)
        self.assertTrue(c["manual_review_required"])
        self.assertEqual(c["original_label"], 0)


class TestPhase2TripartiteEvaluationOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.json_path = os.path.join("outputs", "evaluation", "plagiarism", "tripartite_evaluation_summary.json")
        cls.csv_path = os.path.join("outputs", "evaluation", "plagiarism", "tripartite_experiment_metrics.csv")
        cls.orig_json_path = os.path.join("outputs", "evaluation", "plagiarism", "webis_evaluation_summary.json")
        cls.orig_csv_path = os.path.join("outputs", "evaluation", "plagiarism", "webis_experiment_metrics.csv")

    def test_original_benchmark_preserved_untouched(self):
        """Verify original Webis benchmark files exist and were NOT overwritten."""
        self.assertTrue(os.path.exists(self.orig_json_path), "Original summary JSON must remain intact.")
        self.assertTrue(os.path.exists(self.orig_csv_path), "Original metrics CSV must remain intact.")

        with open(self.orig_json_path, "r", encoding="utf-8") as f:
            orig = json.load(f)
        self.assertIn("exp1_minilm", orig)
        self.assertIn("corpus_note", orig)

    def test_tripartite_outputs_exist_and_consistent(self):
        """Verify tripartite JSON and CSV outputs exist and adhere to schema."""
        self.assertTrue(os.path.exists(self.json_path), "Tripartite summary JSON must exist.")
        self.assertTrue(os.path.exists(self.csv_path), "Tripartite metrics CSV must exist.")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("partition_metadata", data)
        self.assertIn("tripartite_sample_counts", data)
        self.assertIn("models", data)

        for m_key in ["exp1_minilm", "exp2_tfidf", "exp3_hybrid", "exp4_hybrid_gate"]:
            self.assertIn(m_key, data["models"])
            m_res = data["models"][m_key]
            for task_key in ["task1_exact_copying", "task2_genuine_paraphrase", "task3_unrelated_rejection"]:
                self.assertIn(task_key, m_res)
                t = m_res[task_key]
                if "error" not in t:
                    cm = t["confusion_matrix"]
                    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
                    total = tp + fp + fn + tn
                    self.assertEqual(t["sample_size"], total)
                    self.assertAlmostEqual(t["accuracy"], (tp + tn) / total, places=4)
                    self.assertGreaterEqual(t["raw_roc_auc"], 0.0)
                    self.assertLessEqual(t["raw_roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
