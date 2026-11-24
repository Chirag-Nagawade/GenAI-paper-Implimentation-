"""
Citation-Aware Plagiarism Analysis Module.

Preserves and tracks citations before text normalization.
Identifies:
- Numeric citations: [1], [12], [3-5]
- Author-year citations: (Smith et al., 2020), (Johnson, 2019)
- References / Bibliography sections

Associates citations with sentence indices and classifies matches into:
- Potentially cited
- Potentially uncited
- Citation status unclear

Strictly preserves plagiarism similarity scores without unverified alteration.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CitationInfo:
    raw_marker: str
    citation_type: str  # 'numeric' or 'author_year'
    sentence_index: int
    matched_text: str


@dataclass
class DocumentCitationAnalysis:
    total_citations_found: int
    has_reference_section: bool
    reference_entries: List[str]
    citations_by_sentence: Dict[int, List[CitationInfo]] = field(default_factory=dict)


# Regex patterns for citations
NUMERIC_CITATION_PATTERN = re.compile(r'\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]')
AUTHOR_YEAR_PATTERN = re.compile(
    r'\((?:[A-Z][a-zA-Z\'-]+(?:\s+et\s+al\.?)?(?:,?\s*(?:19|20)\d{2}[a-z]?))\)|'
    r'\b[A-Z][a-zA-Z\'-]+\s+et\s+al\.?\s*\((?:19|20)\d{2}[a-z]?\)'
)
REFERENCE_SECTION_HEADER = re.compile(
    r'(?:\n|\A)(?:references|bibliography|works\s+cited)\b',
    re.IGNORECASE
)


def extract_citations(
    raw_text: str,
    extracted_sentences: List[str]
) -> DocumentCitationAnalysis:
    """
    Analyze raw text and extracted sentences to find citations and map them to sentence indices.
    Must be called before citation markers are stripped by text normalization.
    """
    has_ref_section = False
    ref_entries = []

    # Check for Reference section
    ref_match = REFERENCE_SECTION_HEADER.search(raw_text)
    if ref_match:
        has_ref_section = True
        ref_text = raw_text[ref_match.end():]
        ref_lines = [line.strip() for line in ref_text.split("\n") if line.strip()]
        ref_entries = ref_lines[:50]  # Store up to 50 entries

    citations_by_sent: Dict[int, List[CitationInfo]] = {}
    total_citations = 0

    # Map citations across sentences
    for idx, sent in enumerate(extracted_sentences):
        sent_cits = []
        # 1. Numeric citations
        for m in NUMERIC_CITATION_PATTERN.finditer(sent):
            marker = m.group(0).strip()
            sent_cits.append(CitationInfo(
                raw_marker=marker,
                citation_type="numeric",
                sentence_index=idx,
                matched_text=sent
            ))
            total_citations += 1

        # 2. Author-year citations
        for m in AUTHOR_YEAR_PATTERN.finditer(sent):
            marker = m.group(0).strip()
            sent_cits.append(CitationInfo(
                raw_marker=marker,
                citation_type="author_year",
                sentence_index=idx,
                matched_text=sent
            ))
            total_citations += 1

        if sent_cits:
            citations_by_sent[idx] = sent_cits

    return DocumentCitationAnalysis(
        total_citations_found=total_citations,
        has_reference_section=has_ref_section,
        reference_entries=ref_entries,
        citations_by_sentence=citations_by_sent
    )


INFORMAL_ATTRIBUTION_PATTERN = re.compile(
    r'\b(?:according\s+to|as\s+stated\s+by|as\s+reported\s+by|as\s+noted\s+by|'
    r'as\s+described\s+by|as\s+shown\s+by|cited\s+in|et\s+al\b)',
    re.IGNORECASE
)
CONTINUATION_TRANSITIONS = re.compile(
    r'^(?:they\b|the\s+authors?\b|the\s+researchers?\b|furthermore|moreover|in\s+addition|'
    r'additionally|similarly|this\s+study|these\s+results)',
    re.IGNORECASE
)


def assess_citation_status(
    sentence_idx: int,
    citation_analysis: Optional[DocumentCitationAnalysis],
    match: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify a plagiarism match according to citation awareness.

    Distinguishes:
    - Potentially cited: Sentence contains explicit citation marker(s).
      (Does not claim verified attribution; manual review required).
    - Citation status unclear: Sentence contains informal attribution, or adjacent sentence
      carries citation for a contiguous passage, or source appears in references without inline anchor.
    - Potentially uncited: No citation marker or attribution found.
    
    Returns:
        {
            "status": "Potentially cited" | "Potentially uncited" | "Citation status unclear",
            "citation_markers": list of string markers in this sentence,
            "has_references": bool,
            "note": explanation
        }
    """
    if citation_analysis is None:
        return {
            "status": "Citation status unclear",
            "citation_markers": [],
            "has_references": False,
            "note": "Citation analysis was not performed on this document."
        }

    cits = citation_analysis.citations_by_sentence.get(sentence_idx, [])
    markers = [c.raw_marker for c in cits]

    # 1. Direct explicit citation marker on this sentence
    if markers:
        score = match.get("score", 0.0) if match else 0.0
        m_type = match.get("type", "") if match else ""
        verbatim_warning = ""
        if score >= 98.0 or m_type == "Exact Match":
            verbatim_warning = (
                " Caution: Verbatim copying without quotation marks may still constitute "
                "plagiarism even when a citation marker is present."
            )
        return {
            "status": "Potentially cited",
            "citation_markers": markers,
            "has_references": citation_analysis.has_reference_section,
            "note": (
                f"Sentence contains citation marker(s): {', '.join(markers)}. "
                "Note: A citation marker indicates potential attribution, but does NOT verify that the "
                "source was accurately or properly cited; manual review of the cited work is required."
                f"{verbatim_warning}"
            )
        }

    sent_text = ""
    if match and match.get("text"):
        sent_text = match["text"]
    elif sentence_idx in citation_analysis.citations_by_sentence:
        sent_text = citation_analysis.citations_by_sentence[sentence_idx][0].matched_text

    # 2. Check for informal attribution phrasing without formal citation marker
    if sent_text and INFORMAL_ATTRIBUTION_PATTERN.search(sent_text):
        return {
            "status": "Citation status unclear",
            "citation_markers": [],
            "has_references": citation_analysis.has_reference_section,
            "note": (
                "Sentence contains informal attribution phrasing (e.g., 'according to', 'as noted by') "
                "without a formal parseable citation marker. Reviewer verification required."
            )
        }

    # 3. Check if immediately adjacent preceding sentence has an inline citation and this sentence continues it
    prev_cits = citation_analysis.citations_by_sentence.get(sentence_idx - 1, [])
    if prev_cits and sent_text and CONTINUATION_TRANSITIONS.search(sent_text.strip()):
        prev_markers = [c.raw_marker for c in prev_cits]
        return {
            "status": "Citation status unclear",
            "citation_markers": prev_markers,
            "has_references": citation_analysis.has_reference_section,
            "note": (
                f"Sentence lacks an inline citation anchor, but immediately preceding sentence ({sentence_idx - 1}) "
                f"contains citation marker(s): {', '.join(prev_markers)}. Ambiguous whether the citation covers "
                "this continuous passage; reviewer clarification needed."
            )
        }

    # 4. Check if document has bibliography and source title/URL terms appear in reference entries
    if citation_analysis.has_reference_section and match and citation_analysis.reference_entries:
        src_title = (match.get("title") or match.get("url") or "").lower()
        title_words = [w for w in re.split(r'\W+', src_title) if len(w) >= 5 and w not in {"https", "http", "www", "index", "html"}]
        if title_words:
            matched_ref = any(any(tw in ref_entry.lower() for tw in title_words) for ref_entry in citation_analysis.reference_entries)
            if matched_ref:
                return {
                    "status": "Citation status unclear",
                    "citation_markers": [],
                    "has_references": True,
                    "note": (
                        "A potential match for the source title/keywords was identified in the bibliography/references section, "
                        "but this specific sentence lacks an inline citation marker."
                    )
                }

    # 5. Potentially uncited
    if citation_analysis.has_reference_section:
        return {
            "status": "Potentially uncited",
            "citation_markers": [],
            "has_references": True,
            "note": "Matching passage lacks an inline citation, although document includes a references section."
        }

    return {
        "status": "Potentially uncited",
        "citation_markers": [],
        "has_references": False,
        "note": "No citation marker found in the sentence, and no references section was identified."
    }
