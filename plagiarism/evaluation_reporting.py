"""
PDF Reporting Module for PlagiarismAI Evaluation & Benchmark Performance.

Uses FPDF2 to generate publication-grade performance evaluation reports:
- Dataset overview and sample distribution (Webis-CPC-11)
- Scientific validity and operational scope disclaimers
- Multi-model benchmark metrics comparison (MiniLM, TF-IDF, Hybrid, Lexical Gate)
- Confusion matrices and decision thresholds
- Performance, latency, and throughput statistics
"""
import io
import time
from typing import Any, Dict, List, Optional

from fpdf import FPDF


def _latin(text: str) -> str:
    """Encode string safely for standard Helvetica in FPDF2."""
    if not text:
        return ""
    # Replace special symbols if needed
    text = text.replace("α", "alpha").replace("—", "-").replace("–", "-")
    return text.encode("latin-1", "replace").decode("latin-1")


class EvaluationReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "PlagiarismAI - Benchmark Evaluation & Performance Audit", new_x="LMARGIN", new_y="NEXT", align="R")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Webis-CPC-11 Paraphrase Evaluation Benchmark", align="C")


def generate_evaluation_pdf_report(summary: Dict[str, Any]) -> bytes:
    """
    Generate downloadable PDF report from Webis evaluation summary dictionary.
    """
    pdf = EvaluationReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.alias_nb_pages()
    pdf.add_page()

    exp_id = summary.get("experiment_id", "N/A")
    eval_time = summary.get("evaluated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    dataset_name = summary.get("dataset_name", summary.get("dataset", "Webis-CPC-11"))
    total_records = summary.get("total_records", 0)
    train_size = summary.get("train_size", 0)
    dev_size = summary.get("dev_size", 0)
    test_size = summary.get("test_size", 0)
    test_pos = summary.get("test_positives", 0)
    test_neg = summary.get("test_negatives", 0)
    tot_pos = summary.get("total_positives", 0)
    tot_neg = summary.get("total_negatives", 0)
    train_pos = summary.get("train_positives", 0)
    train_neg = summary.get("train_negatives", 0)
    dev_pos = summary.get("dev_positives", 0)
    dev_neg = summary.get("dev_negatives", 0)
    fingerprint = summary.get("dataset_fingerprint", "N/A")
    total_time = summary.get("total_time_seconds", 0.0)
    enc_time = summary.get("encoding_time_seconds", 0.0)

    # 1. Title & Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 10, "PlagiarismAI Benchmark Evaluation Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, f"Experiment ID: {exp_id}  |  Evaluated: {eval_time}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Scientific Validity Disclaimer Box
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(210, 215, 225)
    pdf.rect(pdf.get_x(), pdf.get_y(), 190, 22, "FD")
    pdf.set_xy(pdf.get_x() + 3, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(180, 40, 20)
    pdf.cell(0, 5, "SCIENTIFIC VALIDITY & EVALUATION SCOPE DISCLAIMER", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 60)
    disclaimer_text = (
        "Webis-CPC-11 paraphrase evaluation - not full web plagiarism accuracy. "
        "Webis-CPC-11 evaluates isolated sentence/passage pairs for paraphrase detection. "
        "In production web scanning, PlagiarismAI operates on multi-sentence documents where the "
        "Evidence Gate verifies contextual adjacency and multi-source dispersion."
    )
    pdf.set_x(pdf.get_x() + 3)
    pdf.multi_cell(184, 4, _latin(disclaimer_text))
    pdf.ln(6)

    # 3. Dataset Profile
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 8, "1. Dataset Distribution & Experimental Setup", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)

    col_w = [38, 38, 38, 38, 38]
    pdf.set_fill_color(230, 235, 245)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(col_w[0], 6, "Total Evaluated", border=1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
    pdf.cell(col_w[1], 6, "Train (70%)", border=1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
    pdf.cell(col_w[2], 6, "Dev (15%)", border=1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
    pdf.cell(col_w[3], 6, "Test (15%)", border=1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
    pdf.cell(col_w[4], 6, "Stratified Balance", border=1, new_x="LMARGIN", new_y="NEXT", align="C", fill=True)

    pdf.set_font("Helvetica", "", 8)
    pdf.cell(col_w[0], 6, f"{total_records} (Pos:{tot_pos}/Neg:{tot_neg})", border=1, new_x="RIGHT", new_y="TOP", align="C")
    pdf.cell(col_w[1], 6, f"{train_size} (P:{train_pos}/N:{train_neg})", border=1, new_x="RIGHT", new_y="TOP", align="C")
    pdf.cell(col_w[2], 6, f"{dev_size} (P:{dev_pos}/N:{dev_neg})", border=1, new_x="RIGHT", new_y="TOP", align="C")
    pdf.cell(col_w[3], 6, f"{test_size} (P:{test_pos}/N:{test_neg})", border=1, new_x="RIGHT", new_y="TOP", align="C")
    pdf.cell(col_w[4], 6, f"Pop: 7859 (51.8%/48.2%)", border=1, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Dataset Fingerprint: {fingerprint} | Timing: Encoding {enc_time:.2f}s, Total {total_time:.2f}s", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 4. Multi-Model Benchmark Metrics Table
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 8, "2. Paraphrase Detection Model Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)

    models_info = [
        ("exp1_minilm", "MiniLM Baseline", summary.get("exp1_minilm", {})),
        ("exp2_tfidf", "TF-IDF Baseline", summary.get("exp2_tfidf", {})),
        ("exp3_hybrid", "Hybrid TF-IDF+MiniLM", summary.get("exp3_hybrid", {})),
        ("exp4_hybrid_gate", "Hybrid + Lexical Gate", summary.get("exp4_hybrid_gate", {}))
    ]

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC", "FPR", "Threshold"]
    widths = [46, 18, 18, 18, 18, 18, 18, 18, 18]

    pdf.set_fill_color(225, 235, 248)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(20, 20, 20)
    for idx, (h, w) in enumerate(zip(headers, widths)):
        is_last = (idx == len(headers) - 1)
        pdf.cell(w, 6, h, border=1, new_x="LMARGIN" if is_last else "RIGHT", new_y="NEXT" if is_last else "TOP", align="C", fill=True)

    pdf.set_font("Helvetica", "", 8)
    for _, name, m_dict in models_info:
        if not m_dict:
            continue
        pdf.cell(widths[0], 6, _latin(name), border=1, new_x="RIGHT", new_y="TOP", align="L")
        pdf.cell(widths[1], 6, f"{m_dict.get('accuracy', 0.0)*100:.1f}%", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(widths[2], 6, f"{m_dict.get('precision', 0.0):.3f}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(widths[3], 6, f"{m_dict.get('recall', 0.0):.3f}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(widths[4], 6, f"{m_dict.get('f1_score', 0.0):.3f}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(widths[5], 6, f"{m_dict.get('roc_auc', 0.0):.3f}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(widths[6], 6, f"{m_dict.get('pr_auc', 0.0):.3f}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(widths[7], 6, f"{m_dict.get('false_positive_rate', 0.0)*100:.1f}%", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(widths[8], 6, f"{m_dict.get('best_threshold', 0.0):.2f}", border=1, new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(6)

    # 5. Confusion Matrices
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 8, "3. Test Set Confusion Matrices", new_x="LMARGIN", new_y="NEXT")

    for _, name, m_dict in models_info:
        cm = m_dict.get("confusion_matrix", {})
        tp = cm.get("TP", 0)
        tn = cm.get("TN", 0)
        fp = cm.get("FP", 0)
        fn = cm.get("FN", 0)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 50, 90)
        pdf.cell(0, 5, _latin(name), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(40, 5, f"TP (Paraphrase Hit): {tp}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(40, 5, f"FP (False Alarm): {fp}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(40, 5, f"FN (Missed Paraphrase): {fn}", border=1, new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(40, 5, f"TN (Correct Rejection): {tn}", border=1, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)

    pdf.ln(4)

    # 6. Methodological Analysis & Contextual Gate Explanation
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 8, "4. Architectural Observations & Deployment Guidelines", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(50, 50, 50)
    obs = (
        "1. Dense vs. Sparse Representation: Dense sentence embeddings (MiniLM-L6-v2) capture subtle semantic "
        "rephrasing and synonym substitution where TF-IDF fails, delivering superior recall on real paraphrases.\n"
        "2. Hybrid Lexical-Semantic Fusion: Blending TF-IDF (alpha=0.75) preserves high semantic recall while "
        "increasing discriminative confidence on structurally similar lexical phrasing.\n"
        "3. Evidence Gate Operational Behavior: In isolated sentence datasets (like Webis pairs), contextual "
        "adjacency cannot fire (no preceding/succeeding sentences exist). In full document web scanning, "
        "the Evidence Gate verifies ordered paragraph adjacency and distinctive shared vocabulary, preventing false "
        "positives on common technical definitions and standard formulas."
    )
    pdf.multi_cell(190, 4.5, _latin(obs))

    # 7. Corpus Note: ROC-AUC explanation and Reverse-Ranking Diagnostic
    corpus_note = summary.get("corpus_note", {})
    if corpus_note:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(24, 43, 73)
        pdf.cell(0, 8, "5. Corpus Diagnostics: Reverse-Ranking Diagnostic AUC & Limitations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(190, 4.5, _latin(corpus_note.get("finding", "")))
        pdf.ln(2)

        distinctions_text = (
            "Conceptual Distinctions:\n"
            "- Paraphrase Classification: Binary decision on whether alternative text conveys equivalent meaning.\n"
            "- Plagiarism Detection: Multi-tier pipeline identifying unauthorized reuse across exact, near, and semantic tiers.\n"
            "- Exact / Near-Exact Copying: 100% token identity or minor lexical edits (SequenceMatcher).\n"
            "- Semantic Similarity: Dense embedding cosine alignment (MiniLM) capturing conceptual rephrasing.\n"
            "Dataset Limitation: Webis-CPC-11 is valuable for paraphrase research, but its negative class is dominated by "
            "rejected verbatim copies. In production plagiarism detection, verbatim copies are positive violations, not negatives."
        )
        pdf.multi_cell(190, 4.5, _latin(distinctions_text))
        pdf.ln(2)

        diag = corpus_note.get("reverse_ranking_diagnostic_auc", corpus_note.get("inverted_roc_auc", {}))
        if diag:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 5, "Reverse-Ranking Diagnostic AUC (Mathematical diagnostic: 1 - Raw ROC-AUC; not an independent detector):",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            adj_line = (
                f"MiniLM Diagnostic: {diag.get('minilm', 0.0):.3f}  |  "
                f"TF-IDF Diagnostic: {diag.get('tfidf', 0.0):.3f}  |  "
                f"Hybrid Diagnostic: {diag.get('hybrid', 0.0):.3f}  |  "
                f"Hybrid+Gate Diagnostic: {diag.get('hybrid_gate', 0.0):.3f}"
            )
            pdf.cell(0, 5, _latin(adj_line), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(190, 4, _latin(
                "Note: Reverse-Ranking Diagnostic AUC is strictly a mathematical diagnostic (1 - Raw ROC-AUC) explaining "
                "the inverted benchmark ranking. It does NOT represent improved model performance or a 62.5% plagiarism detection "
                "accuracy claim. Raw ROC-AUC remains the primary benchmark metric."
            ))

    return bytes(pdf.output())
