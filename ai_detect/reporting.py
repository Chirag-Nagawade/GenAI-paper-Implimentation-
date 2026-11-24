"""
PDF Reporting Module for AI Detection Results.

Uses FPDF2 to generate formal, research-grade AI detection audit reports:
- Document metadata and scan timestamp
- Overall summary statistics (AI %, Human %, Needs Review %)
- Classification overview with confidence scores
- Sentence-by-sentence stylometric predictions
- Detailed explanation of stylometric methodology (2024 paper)
- Transparent limitations, disclaimers, and false-positive caveats
"""
import io
import time
from typing import Any, Dict, List

from fpdf import FPDF


def _latin(text: str) -> str:
    """Encode string safely for standard Helvetica in FPDF2."""
    if not text:
        return ""
    return text.encode("latin-1", "replace").decode("latin-1")


class AIDetectionPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "PlagiarismAI - Sentence-Level AI Detection Report", 0, 0, "R")
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | PlagiarismAI Diagnostic Tool", 0, 0, "C")


def generate_ai_pdf_report(ai_result: Dict[str, Any]) -> bytes:
    """
    Generate downloadable PDF report from AI detection results.
    """
    pdf = AIDetectionPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.alias_nb_pages()
    pdf.add_page()

    doc_name = ai_result.get("document_name", "Submitted Document")
    scanned_at = ai_result.get("scanned_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    summary = ai_result.get("summary", {})
    sentences = ai_result.get("sentences", [])

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 10, "AI-Generated Text Detection Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Document: {_latin(doc_name)}   |   Scan Date: {scanned_at}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Executive Summary Card
    pdf.set_fill_color(245, 247, 250)
    pdf.rect(10, pdf.get_y(), 190, 28, "F")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(24, 43, 73)
    pdf.set_xy(14, pdf.get_y() + 3)
    pdf.cell(0, 7, f"Overall Assessment: {_latin(summary.get('overall_classification', 'Evaluated'))}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    stat_str = (
        f"Total Sentences: {ai_result.get('total_sentences', len(sentences))}  |  "
        f"Potentially AI: {summary.get('ai_count', 0)} ({summary.get('ai_percentage', 0.0)}%)  |  "
        f"Needs Review: {summary.get('review_count', 0)} ({summary.get('review_percentage', 0.0)}%)  |  "
        f"Likely Human: {summary.get('human_count', 0)} ({summary.get('human_percentage', 0.0)}%)"
    )
    pdf.set_x(14)
    pdf.cell(0, 6, stat_str, new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(pdf.get_y() + 8)

    # Key Methodology & Limitations Notice
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(180, 50, 50)
    pdf.cell(0, 7, "Important Methodological Disclaimers & Limitations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(60, 60, 60)
    disclaimers = [
        "1. Probabilistic Estimate: AI detection relies on statistical stylometric distributions and is NOT conclusive proof of authorship.",
        "2. Sentence-Level Variance: Individual short sentences (<10 words) exhibit high natural variance; context assistance is applied.",
        "3. Model Scope: Trained on ChatGPT vs Human data; writing styles of other models or non-native English writers may differ.",
        "4. Independent Verification: Matches must never automatically imply plagiarism or vice-versa."
    ]
    for d in disclaimers:
        pdf.cell(0, 5, _latin(d), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Grouped Sentence Breakdown
    by_category = {
        "Potentially AI-generated": [s for s in sentences if s.get("category") == "Potentially AI-generated"],
        "Needs Review": [s for s in sentences if s.get("category") == "Needs Review"],
        "Likely Human": [s for s in sentences if s.get("category") == "Likely Human"]
    }

    category_colors = {
        "Potentially AI-generated": (200, 30, 30),
        "Needs Review": (200, 140, 20),
        "Likely Human": (30, 140, 50)
    }

    for cat_name in ["Potentially AI-generated", "Needs Review", "Likely Human"]:
        items = by_category[cat_name]
        pdf.set_font("Helvetica", "B", 12)
        r, g, b = category_colors[cat_name]
        pdf.set_text_color(r, g, b)
        pdf.cell(0, 8, f"{cat_name} ({len(items)} sentence(s))", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        if not items:
            pdf.cell(0, 6, "None identified in this document.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            continue

        for it in items[:60]:  # Limit output to prevent massive PDFs
            idx = it.get("sentence_index", "?")
            prob = it.get("prob_ai", 0.0) * 100
            conf = int(it.get("confidence", 0.5) * 100)
            warning = it.get("warning", "")

            pdf.set_font("Helvetica", "B", 8.5)
            line1 = f"Sentence #{idx} [AI Probability: {prob:.1f}%, Confidence: {conf}%]"
            if warning:
                line1 += f" - Notice: {warning}"
            pdf.cell(0, 5, _latin(line1), new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "", 8.5)
            text = it.get("sentence", "")
            # Clean text
            pdf.multi_cell(0, 4.5, _latin(f"\"{text}\""))
            pdf.ln(1.5)

        if len(items) > 60:
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(0, 5, f"... and {len(items) - 60} more sentence(s) omitted for brevity.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    return bytes(pdf.output())
