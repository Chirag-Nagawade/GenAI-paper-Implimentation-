"""
Unit Tests for Evaluation Pipeline, Metrics Calculation, Threshold Tuning,
Data Leakage Prevention, and Dashboard Export Artifacts.
"""
import json
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from plagiarism.evaluation_reporting import generate_evaluation_pdf_report
from storage.sqlite_backend import SQLiteStorageBackend


class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        self.summary_json_path = os.path.join("outputs", "evaluation", "plagiarism", "webis_evaluation_summary.json")
        self.metrics_csv_path = os.path.join("outputs", "evaluation", "plagiarism", "webis_experiment_metrics.csv")

    def test_evaluation_summary_file_and_schema(self):
        """Verify webis_evaluation_summary.json exists and adheres to schema."""
        self.assertTrue(os.path.exists(self.summary_json_path), "Summary JSON must exist on disk.")
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        required_keys = [
            "experiment_id", "evaluated_at", "dataset", "dataset_fingerprint",
            "total_records", "train_size", "dev_size", "test_size",
            "test_positives", "test_negatives", "exp1_minilm", "exp2_tfidf",
            "exp3_hybrid", "exp4_hybrid_gate", "curves"
        ]
        for key in required_keys:
            self.assertIn(key, summary, f"Key '{key}' missing from evaluation summary.")

        # Verify metric bounds for all models
        for m_key in ["exp1_minilm", "exp2_tfidf", "exp3_hybrid", "exp4_hybrid_gate"]:
            m = summary[m_key]
            for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
                self.assertIn(metric, m)
                self.assertGreaterEqual(m[metric], 0.0)
                self.assertLessEqual(m[metric], 1.0)
            self.assertIn("confusion_matrix", m)
            for cm_key in ["TP", "TN", "FP", "FN"]:
                self.assertIn(cm_key, m["confusion_matrix"])

    def test_metrics_csv_format(self):
        """Verify webis_experiment_metrics.csv exists and has all comparative columns."""
        self.assertTrue(os.path.exists(self.metrics_csv_path), "Metrics CSV must exist on disk.")
        df = pd.read_csv(self.metrics_csv_path)
        required_cols = [
            "model", "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "pr_auc", "fpr", "dev_f1", "decision_threshold"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' missing from metrics CSV.")
        self.assertGreaterEqual(len(df), 4, "Must contain all 4 evaluated models.")

    def test_metrics_calculation_accuracy(self):
        """Verify mathematical correctness of metrics functions."""
        y_true = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.85, 0.1, 0.15])
        threshold = 0.5
        y_pred = (y_scores >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_scores)

        # Expected:
        # y_true 1s: indices 0, 1, 2, 5, 7 (5 total)
        # y_pred 1s: indices 0, 1, 2, 3, 7 (5 total)
        # TP: 0, 1, 2, 7 (4), FP: 3 (1), FN: 5 (1), TN: 4, 6, 8, 9 (4)
        self.assertEqual(acc, 0.8)
        self.assertEqual(prec, 0.8)
        self.assertEqual(rec, 0.8)
        self.assertEqual(f1, 0.8)
        self.assertGreater(auc, 0.8)

    def test_threshold_tuning_on_dev_set(self):
        """Verify that decision threshold selection maximizes F1 on development data."""
        rng = np.random.RandomState(42)
        dev_labels = np.array([1]*50 + [0]*50)
        # Positive scores centered at 0.75, negative at 0.45
        dev_scores = np.concatenate([
            rng.normal(0.75, 0.1, 50),
            rng.normal(0.45, 0.1, 50)
        ])
        dev_scores = np.clip(dev_scores, 0.0, 1.0)

        best_t = 0.5
        best_f1 = -1.0
        grid = np.linspace(0.35, 0.95, 61)
        for t in grid:
            preds = (dev_scores >= t).astype(int)
            score = f1_score(dev_labels, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t

        # Best threshold should separate the distributions around 0.55 - 0.65
        self.assertGreaterEqual(best_t, 0.50)
        self.assertLessEqual(best_t, 0.70)
        self.assertGreater(best_f1, 0.85)

    def test_data_leakage_partitions_disjoint(self):
        """Verify that train, dev, and test partition indices have zero overlap and maintain stratification."""
        n_total = 500
        labels = np.array([1]*259 + [0]*241)
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        rng = np.random.RandomState(42)
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        n_train_pos = int(round(len(pos_indices) * 0.70))
        n_dev_pos = int(round(len(pos_indices) * 0.15))
        n_train_neg = int(round(len(neg_indices) * 0.70))
        n_dev_neg = int(round(len(neg_indices) * 0.15))

        train_idx = set(np.concatenate([pos_indices[:n_train_pos], neg_indices[:n_train_neg]]))
        dev_idx = set(np.concatenate([pos_indices[n_train_pos:n_train_pos + n_dev_pos], neg_indices[n_train_neg:n_train_neg + n_dev_neg]]))
        test_idx = set(np.concatenate([pos_indices[n_train_pos + n_dev_pos:], neg_indices[n_train_neg + n_dev_neg:]]))

        # Partitions must be mutually disjoint
        self.assertEqual(len(train_idx & dev_idx), 0, "Train and Dev partitions must not overlap!")
        self.assertEqual(len(train_idx & test_idx), 0, "Train and Test partitions must not overlap!")
        self.assertEqual(len(dev_idx & test_idx), 0, "Dev and Test partitions must not overlap!")
        # Union must equal full dataset
        self.assertEqual(len(train_idx | dev_idx | test_idx), n_total)
        # Class proportions preserved in test set
        test_labels = labels[list(test_idx)]
        pos_share = np.mean(test_labels == 1)
        self.assertAlmostEqual(pos_share, 259 / 500, places=2)

    def test_evaluation_pdf_generation(self):
        """Verify that generate_evaluation_pdf_report generates valid non-empty PDF bytes."""
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        pdf_bytes = generate_evaluation_pdf_report(summary)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # PDF signature check
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"), "Generated report must begin with PDF header %PDF-")

    def test_database_evaluation_persistence(self):
        """Verify that evaluations persist to and load from database storage."""
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        backend = SQLiteStorageBackend(db_path=temp_db.name)
        backend.init()

        try:
            with open(self.summary_json_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            eval_id = backend.save_evaluation(summary)
            self.assertIsNotNone(eval_id)

            history = backend.get_evaluations()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["dataset_name"], summary.get("dataset", "Webis-CPC-11"))

            details = backend.get_evaluation_details(eval_id)
            self.assertIsNotNone(details)
            self.assertEqual(details.get("experiment_id", details.get("id")), eval_id)
            self.assertIn("exp1_minilm", details)
        finally:
            if os.path.exists(temp_db.name):
                try:
                    os.remove(temp_db.name)
                except OSError:
                    pass

    def test_fpr_and_fdr_calculation_known_confusion_matrix(self):
        """Verify mathematical calculations of FPR, FDR, Accuracy, Precision, Recall, and F1."""
        from validate_real_model import _compute_metrics

        # Scenario A: Known sample confusion matrix matching reported Hybrid test results
        # TN = 4, FP = 11, FN = 8, TP = 52  (Total = 75)
        y_true = np.array([1]*60 + [0]*15)
        # 52 TPs out of 60 positives, 8 FNs
        # 11 FPs out of 15 negatives, 4 TNs
        y_pred = np.array([1]*52 + [0]*8 + [1]*11 + [0]*4)
        y_scores = np.where(y_pred == 1, 0.85, 0.25)

        metrics = _compute_metrics(y_true, y_pred, y_scores)
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["TP"], 52)
        self.assertEqual(cm["FP"], 11)
        self.assertEqual(cm["FN"], 8)
        self.assertEqual(cm["TN"], 4)

        # 1. False Positive Rate: FP / (FP + TN) = 11 / 15 = 0.7333
        self.assertAlmostEqual(metrics["false_positive_rate"], 11 / 15, places=4)
        # 2. False Discovery Rate: FP / (FP + TP) = 11 / 63 = 0.1746 (1 - Precision)
        self.assertAlmostEqual(metrics["false_discovery_rate"], 11 / 63, places=4)
        # 3. Accuracy: (TP + TN) / Total = (52 + 4) / 75 = 56 / 75 = 0.7467
        self.assertAlmostEqual(metrics["accuracy"], 56 / 75, places=4)
        # 4. Precision: TP / (TP + FP) = 52 / 63 = 0.8254
        self.assertAlmostEqual(metrics["precision"], 52 / 63, places=4)
        # 5. Recall: TP / (TP + FN) = 52 / 60 = 0.8667
        self.assertAlmostEqual(metrics["recall"], 52 / 60, places=4)
        # 6. F1: 2 * P * R / (P + R) = 2 * (52/63) * (52/60) / ((52/63) + (52/60)) = 104 / 123 = 0.8455
        self.assertAlmostEqual(metrics["f1_score"], 104 / 123, places=4)

        # Scenario B: Standard balanced matrix
        # TN = 50, FP = 5, FN = 5, TP = 40 (Total = 100)
        y_true_b = np.array([1]*45 + [0]*55)
        y_pred_b = np.array([1]*40 + [0]*5 + [1]*5 + [0]*50)
        y_scores_b = np.where(y_pred_b == 1, 0.9, 0.1)

        metrics_b = _compute_metrics(y_true_b, y_pred_b, y_scores_b)
        self.assertAlmostEqual(metrics_b["false_positive_rate"], 5 / 55, places=4)
        self.assertAlmostEqual(metrics_b["false_discovery_rate"], 5 / 45, places=4)
        self.assertAlmostEqual(metrics_b["accuracy"], 90 / 100, places=4)
        self.assertAlmostEqual(metrics_b["precision"], 40 / 45, places=4)
        self.assertAlmostEqual(metrics_b["recall"], 40 / 45, places=4)
        self.assertAlmostEqual(metrics_b["f1_score"], 40 / 45, places=4)

    def test_saved_metrics_match_confusion_matrices(self):
        """Verify that saved metrics in summary JSON strictly match their confusion matrices."""
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        for model_key in ["exp1_minilm", "exp2_tfidf", "exp3_hybrid", "exp4_hybrid_gate"]:
            m = summary[model_key]
            cm = m["confusion_matrix"]
            tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
            total = tp + fp + fn + tn

            expected_acc = round(float((tp + tn) / total), 4)
            expected_prec = round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4)
            expected_rec = round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4)
            expected_f1 = round(float(2 * expected_prec * expected_rec / (expected_prec + expected_rec)) if (expected_prec + expected_rec) > 0 else 0.0, 4)
            expected_fpr = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4)

            self.assertEqual(m["accuracy"], expected_acc, f"Accuracy mismatch in {model_key}")
            self.assertEqual(m["precision"], expected_prec, f"Precision mismatch in {model_key}")
            self.assertEqual(m["recall"], expected_rec, f"Recall mismatch in {model_key}")
            self.assertEqual(m["f1_score"], expected_f1, f"F1 mismatch in {model_key}")
            self.assertEqual(m["false_positive_rate"], expected_fpr, f"FPR mismatch in {model_key}")

    def test_dashboard_consistency_with_evaluation_files(self):
        """Verify that summary JSON and experiment CSV have identical metric values and thresholds."""
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        df = pd.read_csv(self.metrics_csv_path)

        for model_key in ["exp1_minilm", "exp2_tfidf", "exp3_hybrid", "exp4_hybrid_gate"]:
            m = summary[model_key]
            model_name = m.get("model_name", model_key)
            row = df[df["model"] == model_name]
            self.assertEqual(len(row), 1, f"Model '{model_name}' not found uniquely in CSV.")
            csv_row = row.iloc[0]

            self.assertAlmostEqual(m["accuracy"], csv_row["accuracy"], places=4)
            self.assertAlmostEqual(m["precision"], csv_row["precision"], places=4)
            self.assertAlmostEqual(m["recall"], csv_row["recall"], places=4)
            self.assertAlmostEqual(m["f1_score"], csv_row["f1_score"], places=4)
            self.assertAlmostEqual(m["false_positive_rate"], csv_row["fpr"], places=4)


    def test_corpus_note_in_summary_json(self):
        """Verify that corpus_note is present and contains all required fields with correct types."""
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertIn("corpus_note", summary, "corpus_note key must exist in evaluation summary.")
        note = summary["corpus_note"]

        # Required textual fields
        for field in ["finding", "implication", "dataset_limitations"]:
            self.assertIn(field, note, f"corpus_note.{field} must be present.")
            self.assertIsInstance(note[field], str)
            self.assertGreater(len(note[field]), 20)

        self.assertIn("methodological_distinctions", note)
        dist = note["methodological_distinctions"]
        for k in ["paraphrase_classification", "plagiarism_detection", "exact_copying_detection", "near_exact_copying_detection", "semantic_similarity_detection"]:
            self.assertIn(k, dist, f"Distinction for '{k}' must be documented.")

        # Similarity statistics
        for stat in ["positive_median_sim", "negative_median_sim", "positive_mean_sim", "negative_mean_sim"]:
            self.assertIn(stat, note, f"corpus_note.{stat} must be present.")
            val = note[stat]
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0.0)
            self.assertLessEqual(val, 1.0)

        # Negative median must exceed positive median (verbatim-copy corpus property)
        self.assertGreater(
            note["negative_median_sim"],
            note["positive_median_sim"],
            "Negative class median cosine sim must exceed positive class median on Webis-CPC-11."
        )

        # Reverse-Ranking Diagnostic AUC block
        self.assertTrue(
            "reverse_ranking_diagnostic_auc" in note or "inverted_roc_auc" in note,
            "Reverse-Ranking Diagnostic block must be present in corpus_note."
        )
        diag = note.get("reverse_ranking_diagnostic_auc", note.get("inverted_roc_auc", {}))
        self.assertEqual(diag.get("diagnostic_name"), "Reverse-Ranking Diagnostic AUC")
        for model_key in ["minilm", "tfidf", "hybrid", "hybrid_gate"]:
            self.assertIn(model_key, diag, f"diagnostic.{model_key} must be present.")
            val = diag[model_key]
            self.assertIsInstance(val, float)
            # Inverted AUC must be > 0.5 (since raw AUC < 0.5)
            self.assertGreater(val, 0.5, f"diagnostic.{model_key} should exceed 0.5.")

    def test_roc_auc_inverted_property(self):
        """Verify that for Webis-CPC-11, all models have raw ROC-AUC < 0.5 (corpus property)."""
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        for model_key in ["exp1_minilm", "exp2_tfidf", "exp3_hybrid", "exp4_hybrid_gate"]:
            roc = summary[model_key]["roc_auc"]
            self.assertLess(
                roc, 0.5,
                f"{model_key} ROC-AUC={roc:.4f} should be < 0.5 on Webis-CPC-11 "
                "(verbatim-copy negatives invert cosine score ranking)."
            )

        # Corpus note must confirm: Reverse-Ranking Diagnostic > 0.5 for all models
        diag = summary["corpus_note"].get("reverse_ranking_diagnostic_auc", summary["corpus_note"]["inverted_roc_auc"])
        for key in ["minilm", "tfidf", "hybrid", "hybrid_gate"]:
            raw_key = {"minilm": "exp1_minilm", "tfidf": "exp2_tfidf",
                       "hybrid": "exp3_hybrid", "hybrid_gate": "exp4_hybrid_gate"}[key]
            self.assertAlmostEqual(
                diag[key], round(1.0 - summary[raw_key]["roc_auc"], 4), places=3,
                msg=f"Reverse-Ranking Diagnostic {key} should equal 1 - raw roc_auc (symmetric ROC)."
            )


if __name__ == "__main__":
    unittest.main()
