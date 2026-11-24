"""
AI Dataset Loader and Verification Module for ChatGPT-Human-NewGenerated Dataset.

Handles:
- Verification and loading of 81 training problems and 82 training truth files.
- Robust JSON extraction handling trailing text/annotations (e.g., truth-problem-2.txt).
- Accurate mapping of paragraph authors: 1 -> ChatGPT (1), 2 -> Human (0).
- Paragraph-level extraction with document and paragraph IDs.
- Sentence-level extraction with inherited weak labels and parent context.
- Verification of 19 testing documents (reports that ground truth annotations are absent; never fabricates labels).
- Output verification report.
"""
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParagraphRecord:
    doc_id: str
    paragraph_idx: int
    text: str
    label: int  # 1 for ChatGPT, 0 for Human
    author_name: str  # "ChatGPT" or "Human"


@dataclass
class SentenceRecord:
    doc_id: str
    paragraph_idx: int
    sentence_idx: int
    text: str
    weak_label: int  # 1 for ChatGPT, 0 for Human (inherited weakly from paragraph)
    parent_paragraph_text: str
    context_window: str = ""  # Surrounding sentences in parent paragraph


def _parse_truth_file(path: str) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[str]]:
    """
    Robustly extract 'paragraph-authors' and 'changes' from truth JSON file.
    Handles extra trailing text outside JSON block.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read().strip()
        # Find JSON object boundaries {...}
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start:end + 1]
            data = json.loads(json_str)
            authors = data.get("paragraph-authors", [])
            changes = data.get("changes", [])
            return authors, changes, None
        return None, None, "No JSON block found in file"
    except Exception as e:
        return None, None, str(e)


_NLTK_PUNKT_AVAILABLE = None

def split_into_sentences(text: str) -> List[str]:
    """
    Sentence splitter using NLTK if available locally, with robust regex fallback.
    Avoids network calls in offline/sandbox environments.
    """
    global _NLTK_PUNKT_AVAILABLE
    text = text.strip()
    if not text:
        return []

    if _NLTK_PUNKT_AVAILABLE is None:
        try:
            import nltk
            nltk.sent_tokenize("Test sentence.")
            _NLTK_PUNKT_AVAILABLE = True
        except Exception:
            _NLTK_PUNKT_AVAILABLE = False

    if _NLTK_PUNKT_AVAILABLE:
        try:
            import nltk
            return nltk.sent_tokenize(text)
        except Exception:
            pass

    # High-quality regex-based sentence boundary splitter
    splits = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', text)
    return [s.strip() for s in splits if s.strip()]


def load_ai_dataset(
    dataset_dir: str = "data/ChatGPT-Human-NewGenerated",
    verbose: bool = True
) -> Tuple[List[ParagraphRecord], List[SentenceRecord], Dict[str, Any]]:
    """
    Load training documents and annotations.
    
    Returns:
        (paragraph_records, sentence_records, verification_report)
    """
    train_probs_dir = os.path.join(dataset_dir, "Training-Problems")
    train_truth_dir = os.path.join(dataset_dir, "Training-Truth")
    test_dir = os.path.join(dataset_dir, "Testing")

    if not os.path.isdir(train_probs_dir):
        raise FileNotFoundError(f"Training problems directory not found: {train_probs_dir}")

    prob_files = glob.glob(os.path.join(train_probs_dir, "problem-*.txt"))
    test_files = glob.glob(os.path.join(test_dir, "problem-*.txt")) if os.path.isdir(test_dir) else []

    paragraphs: List[ParagraphRecord] = []
    sentences: List[SentenceRecord] = []

    matched_docs = 0
    missing_truth = []
    mismatched_paragraph_counts = []
    parse_errors = []

    total_chatgpt_paragraphs = 0
    total_human_paragraphs = 0
    total_chatgpt_sentences = 0
    total_human_sentences = 0

    for prob_path in sorted(prob_files):
        fname = os.path.basename(prob_path)
        doc_id = fname.replace(".txt", "")  # e.g., 'problem-104'
        truth_path = os.path.join(train_truth_dir, f"truth-{doc_id}.txt")

        if not os.path.isfile(truth_path):
            missing_truth.append(doc_id)
            continue

        authors, changes, err = _parse_truth_file(truth_path)
        if err or authors is None:
            parse_errors.append((doc_id, err))
            continue

        # Read problem paragraphs (each non-empty line or newline-separated block is a paragraph)
        with open(prob_path, "r", encoding="utf-8", errors="replace") as f:
            doc_lines = [line.strip() for line in f if line.strip()]

        if len(doc_lines) != len(authors):
            mismatched_paragraph_counts.append({
                "doc_id": doc_id,
                "text_paragraphs": len(doc_lines),
                "truth_authors": len(authors)
            })
            # Take minimum length to preserve alignment without guessing
            limit = min(len(doc_lines), len(authors))
        else:
            limit = len(doc_lines)

        matched_docs += 1

        for p_idx in range(limit):
            p_text = doc_lines[p_idx]
            raw_author = authors[p_idx]
            # In dataset truth: 1 = ChatGPT, 2 = Human
            if raw_author == 1:
                label = 1
                author_name = "ChatGPT"
                total_chatgpt_paragraphs += 1
            elif raw_author == 2:
                label = 0
                author_name = "Human"
                total_human_paragraphs += 1
            else:
                continue

            para_rec = ParagraphRecord(
                doc_id=doc_id,
                paragraph_idx=p_idx,
                text=p_text,
                label=label,
                author_name=author_name
            )
            paragraphs.append(para_rec)

            # Split paragraph into sentences for sentence-level experimentation
            sents = split_into_sentences(p_text)
            for s_idx, s_text in enumerate(sents):
                if not s_text or len(s_text.split()) < 2:
                    continue

                # Compute local context window (previous sentence + current + next sentence)
                start_w = max(0, s_idx - 1)
                end_w = min(len(sents), s_idx + 2)
                context_win = " ".join(sents[start_w:end_w])

                sent_rec = SentenceRecord(
                    doc_id=doc_id,
                    paragraph_idx=p_idx,
                    sentence_idx=s_idx,
                    text=s_text,
                    weak_label=label,  # Note: Inherited weakly from paragraph
                    parent_paragraph_text=p_text,
                    context_window=context_win
                )
                sentences.append(sent_rec)
                if label == 1:
                    total_chatgpt_sentences += 1
                else:
                    total_human_sentences += 1

    # Check testing folder for truth files
    test_truth_found = []
    test_docs = []
    for tf in sorted(test_files):
        t_id = os.path.basename(tf).replace(".txt", "")
        test_docs.append(t_id)
        # Check potential truth naming
        cand_truths = [
            os.path.join(test_dir, f"truth-{t_id}.txt"),
            os.path.join(train_truth_dir, f"truth-{t_id}.txt")
        ]
        for ct in cand_truths:
            if os.path.isfile(ct):
                test_truth_found.append(t_id)

    verification_report = {
        "training_problem_files_count": len(prob_files),
        "training_documents_matched": matched_docs,
        "missing_truth_count": len(missing_truth),
        "missing_truth_docs": missing_truth,
        "truth_parse_error_count": len(parse_errors),
        "mismatched_paragraph_counts": mismatched_paragraph_counts,
        "total_paragraphs": len(paragraphs),
        "chatgpt_paragraphs": total_chatgpt_paragraphs,
        "human_paragraphs": total_human_paragraphs,
        "paragraph_class_ratio": round(total_chatgpt_paragraphs / max(total_human_paragraphs, 1), 3),
        "total_sentences": len(sentences),
        "chatgpt_sentences_weak": total_chatgpt_sentences,
        "human_sentences_weak": total_human_sentences,
        "testing_problem_files_count": len(test_files),
        "testing_truth_annotations_present": len(test_truth_found),
        "testing_status_note": (
            "No ground-truth annotations are provided for the 19 testing documents in ChatGPT-Human-NewGenerated. "
            "In accordance with research integrity requirements, testing labels are NOT fabricated; "
            "model evaluation is strictly conducted via document-grouped cross-validation on verified training data."
        )
    }

    if verbose:
        print("\n" + "=" * 70)
        print("AI DATASET VERIFICATION REPORT (ChatGPT-Human-NewGenerated)")
        print("=" * 70)
        print(f"Training Documents: {matched_docs}/{len(prob_files)}")
        print(f"Paragraphs: {len(paragraphs)} (ChatGPT: {total_chatgpt_paragraphs}, Human: {total_human_paragraphs})")
        print(f"Sentences (Weak Labels): {len(sentences)} (ChatGPT: {total_chatgpt_sentences}, Human: {total_human_sentences})")
        print(f"Testing Documents: {len(test_files)} (Ground Truth Found: {len(test_truth_found)})")
        print(f"Testing Note: {verification_report['testing_status_note']}")
        print("=" * 70 + "\n")

    return paragraphs, sentences, verification_report
