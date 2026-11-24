"""
Unit Tests for AI-Generated Text Detection Module.
"""
import unittest
import numpy as np
from ai_detect.features import extract_stylometric_features, extract_feature_matrix, FEATURE_NAMES
from ai_detect.predict import predict_sentence, predict_document


class TestAIDetection(unittest.TestCase):
    def test_feature_vector_dimension(self):
        text = "This is a typical test sentence for stylometric feature extraction."
        vec = extract_stylometric_features(text)
        # Feature count is determined by FEATURE_NAMES (may grow as new features are added)
        n_features = len(FEATURE_NAMES)
        self.assertGreaterEqual(n_features, 33, "Must have at least the original 33 features")
        self.assertEqual(len(vec), n_features)
        self.assertFalse(np.any(np.isnan(vec)))

    def test_feature_matrix(self):
        texts = [
            "First sentence of the document.",
            "Furthermore, second sentence provides transitional discourse markers."
        ]
        mat = extract_feature_matrix(texts)
        n_features = len(FEATURE_NAMES)
        self.assertEqual(mat.shape, (2, n_features))

    def test_predict_sentence_categories(self):
        # Human-like sentence (short, first person)
        res_human = predict_sentence("I went to the store and bought some milk.", None)
        self.assertIn("category", res_human)
        self.assertIn(res_human["category"], ["Likely Human", "Needs Review", "Potentially AI-generated"])
        self.assertIn("prob_ai", res_human)
        self.assertIn("confidence", res_human)

        # AI-characteristic phrasing
        ai_phrasing = "Furthermore, in conclusion, it is crucial to recognize that the multifaceted nature of education facilitates holistic development."
        res_ai = predict_sentence(ai_phrasing, None)
        self.assertIn(res_ai["category"], ["Likely Human", "Needs Review", "Potentially AI-generated"])

    def test_short_sentence_handling(self):
        res_short = predict_sentence("Yes, indeed.", None)
        self.assertTrue(res_short["warning"] != "")

    def test_predict_document(self):
        doc_text = (
            "I wrote this personal reflection based on my childhood experiences.\n"
            "We enjoyed hiking in the nearby mountains every summer.\n\n"
            "Furthermore, it is essential to emphasize that systematic pedagogical frameworks enhance student engagement significantly."
        )
        res = predict_document(doc_text)
        self.assertGreater(res["total_sentences"], 0)
        self.assertIn("summary", res)
        self.assertIn("overall_classification", res["summary"])
        self.assertIn("disclaimer", res["summary"])


if __name__ == "__main__":
    unittest.main()
