"""
Unit Tests for Citation-Aware Plagiarism Analysis.
"""
import unittest
from plagiarism.citations import extract_citations, assess_citation_status


class TestCitationAwareAnalysis(unittest.TestCase):
    def setUp(self):
        self.raw_text = (
            "Deep neural networks require large datasets for effective generalization [12].\n"
            "Recent advancements demonstrate transformer scalability (Vaswani et al., 2017).\n"
            "Uncited claims are common in draft manuscripts.\n"
            "\nReferences\n"
            "[12] LeCun, Y. et al. Deep learning. Nature, 2015.\n"
            "Vaswani et al. Attention is all you need. NeurIPS, 2017.\n"
        )
        self.sentences = [
            "Deep neural networks require large datasets for effective generalization [12].",
            "Recent advancements demonstrate transformer scalability (Vaswani et al., 2017).",
            "Uncited claims are common in draft manuscripts."
        ]

    def test_citation_extraction(self):
        analysis = extract_citations(self.raw_text, self.sentences)
        self.assertTrue(analysis.has_reference_section)
        self.assertEqual(len(analysis.reference_entries), 2)
        self.assertGreaterEqual(analysis.total_citations_found, 2)

        # Sentence 0 has numeric citation [12]
        self.assertIn(0, analysis.citations_by_sentence)
        self.assertEqual(analysis.citations_by_sentence[0][0].raw_marker, "[12]")
        self.assertEqual(analysis.citations_by_sentence[0][0].citation_type, "numeric")

        # Sentence 1 has author-year citation
        self.assertIn(1, analysis.citations_by_sentence)
        self.assertEqual(analysis.citations_by_sentence[1][0].citation_type, "author_year")

        # Sentence 2 has no citation
        self.assertNotIn(2, analysis.citations_by_sentence)

    def test_assess_citation_status_cited(self):
        analysis = extract_citations(self.raw_text, self.sentences)
        status0 = assess_citation_status(0, analysis)
        self.assertEqual(status0["status"], "Potentially cited")
        self.assertIn("[12]", status0["citation_markers"])

    def test_assess_citation_status_uncited(self):
        analysis = extract_citations(self.raw_text, self.sentences)
        status2 = assess_citation_status(2, analysis)
        self.assertEqual(status2["status"], "Potentially uncited")
        self.assertTrue(status2["has_references"])

    def test_assess_citation_status_unclear(self):
        status_none = assess_citation_status(0, None)
        self.assertEqual(status_none["status"], "Citation status unclear")

    def test_citation_disclaimer_does_not_claim_verification(self):
        analysis = extract_citations(self.raw_text, self.sentences)
        # Exact match copy with citation
        match = {"score": 100.0, "type": "Exact Match", "text": self.sentences[0]}
        status0 = assess_citation_status(0, analysis, match)
        self.assertEqual(status0["status"], "Potentially cited")
        # Must clearly state that presence of marker is not verification
        self.assertIn("does NOT verify", status0["note"])
        self.assertIn("manual review", status0["note"].lower())
        self.assertIn("Verbatim copying", status0["note"])

    def test_informal_attribution_phrase_unclear(self):
        analysis = extract_citations(self.raw_text, self.sentences)
        match = {
            "score": 75.0,
            "type": "Semantic Match",
            "text": "According to previous studies, neural networks generalize well."
        }
        # Sentence 2 itself has no formal citation markers
        status2 = assess_citation_status(2, analysis, match)
        self.assertEqual(status2["status"], "Citation status unclear")
        self.assertIn("informal attribution", status2["note"].lower())

    def test_contiguous_passage_citation_unclear(self):
        raw = (
            "Transformers scale exceptionally well (Vaswani et al., 2017).\n"
            "Furthermore, their attention mechanisms capture global dependencies.\n"
        )
        sents = [
            "Transformers scale exceptionally well (Vaswani et al., 2017).",
            "Furthermore, their attention mechanisms capture global dependencies."
        ]
        analysis = extract_citations(raw, sents)
        match1 = {
            "score": 80.0,
            "type": "Semantic Match",
            "text": sents[1]
        }
        # Sentence 1 follows cited sentence 0 with continuation transition 'Furthermore'
        status1 = assess_citation_status(1, analysis, match1)
        self.assertEqual(status1["status"], "Citation status unclear")
        self.assertIn("immediately preceding sentence", status1["note"].lower())

    def test_citation_preservation_before_normalization(self):
        import app
        raw_sentence = "Convolutional filters capture spatial hierarchies [3, 4]."
        # Normalization strips brackets and numbers
        normalized = app.normalize_text(raw_sentence)
        self.assertNotIn("[3, 4]", normalized)
        # extract_citations preserves the raw marker
        analysis = extract_citations(raw_sentence, [raw_sentence])
        self.assertIn(0, analysis.citations_by_sentence)
        self.assertEqual(analysis.citations_by_sentence[0][0].raw_marker, "[3, 4]")


if __name__ == "__main__":
    unittest.main()
