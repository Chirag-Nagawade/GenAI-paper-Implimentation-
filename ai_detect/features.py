"""
Stylometric Feature Extraction Module

Based on the methodological foundation of:
"Unveiling ChatGPT Text Using Writing Style" (2024).

Implements 33 interpretable stylometric features across:
- Surface / Length Metrics (word length, sentence length, character counts)
- Lexical Richness & Diversity (TTR, RTTR, CTTR, LogTTR, Simpson's D, Hapax Legomena)
- Punctuation & Orthographic Characteristics (comma, semicolon, dash, quotes, case)
- Syntactic & Function-Word Patterns (pronouns, conjunctions, prepositions, transitions)
- Part-of-Speech Distribution (nouns, verbs, adjectives, adverbs)

Supports both:
1. Pure Sentence/Paragraph Feature Extraction
2. Context-Assisted Feature Extraction (using surrounding window)
"""
import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np

# -----------------------------------------------------------------------------
# Function words & discourse markers common in English and characteristic in LLMs
# -----------------------------------------------------------------------------
FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

FIRST_PERSON_PRONOUNS = {"i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves"}
THIRD_PERSON_PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"}

TRANSITIONAL_CONJUNCTIONS = {
    "furthermore", "moreover", "additionally", "however", "therefore", "consequently",
    "nevertheless", "nonetheless", "conversely", "meanwhile", "thus", "hence", "accordingly",
    "in conclusion", "in summary", "on the other hand", "firstly", "secondly", "finally",
    "in addition", "specifically", "in particular", "for example", "for instance",
    "as a result", "to begin with", "notably", "significantly", "importantly", "overall",
    "ultimately", "essentially", "first", "second", "third", "lastly", "to summarize",
    "in contrast", "on the contrary", "in terms of", "with respect to", "as well as",
    "in order to", "crucial", "essential", "vital", "indispensable", "pivotal", "paramount",
    "delve", "tapestry", "landscape", "beacon", "testament", "realm", "foster", "garner",
    "seamless", "multifaceted", "holistic", "interplay", "underscores", "nuanced", "vibrant",
    "comprehensive", "bolster", "revolutionize", "cornerstone", "imperative", "nuance"
}

FEATURE_NAMES = [
    "f01_avg_word_length",
    "f02_word_count",
    "f03_char_count",
    "f04_type_token_ratio",
    "f05_root_ttr",
    "f06_corrected_ttr",
    "f07_log_ttr",
    "f08_simpsons_d",
    "f09_hapax_ratio",
    "f10_dis_legomena_ratio",
    "f11_punctuation_ratio",
    "f12_comma_density",
    "f13_period_density",
    "f14_semicolon_density",
    "f15_colon_density",
    "f16_question_density",
    "f17_exclamation_density",
    "f18_dash_density",
    "f19_quote_density",
    "f20_uppercase_ratio",
    "f21_digit_ratio",
    "f22_function_word_ratio",
    "f23_pronoun_ratio",
    "f24_first_person_ratio",
    "f25_third_person_ratio",
    "f26_transitional_word_ratio",
    "f27_preposition_density",
    "f28_noun_ratio",
    "f29_verb_ratio",
    "f30_adj_ratio",
    "f31_adv_ratio",
    "f32_sentence_complexity_approx",
    "f33_avg_syllables_per_word",
    "f34_sentence_len_cv",
    "f35_sentence_len_mean",
    "f36_repeated_bigram_ratio",
    "f37_contraction_ratio",
]


# Indices of raw length features (kept for reporting, dropped from the classifier input)
LENGTH_FEATURE_IDX = {1, 2}


def _fast_syllable_count(word: str) -> int:
    """Heuristic syllable counter (vowel groups, silent-e aware)."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 1
    if len(word) <= 3:
        return 1
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    elif word.endswith(("es", "ed")) and count > 1 and not word.endswith(("ted", "ded", "ses", "zes", "ces", "ges")):
        count -= 1
    return max(1, count)


_NLTK_POS_AVAILABLE = None


def _is_nltk_pos_available() -> bool:
    global _NLTK_POS_AVAILABLE
    if _NLTK_POS_AVAILABLE is None:
        try:
            import nltk
            nltk.pos_tag(["test"])
            _NLTK_POS_AVAILABLE = True
        except Exception:
            _NLTK_POS_AVAILABLE = False
    return _NLTK_POS_AVAILABLE


def _extract_pos_tags(words: List[str]) -> Counter:
    """Extract simplified POS tags using NLTK pos_tag or heuristic fallback."""
    if _is_nltk_pos_available():
        try:
            import nltk
            tags = nltk.pos_tag(words)
            cats = Counter()
            for _, tag in tags:
                if tag.startswith("NN"):
                    cats["noun"] += 1
                elif tag.startswith("VB"):
                    cats["verb"] += 1
                elif tag.startswith("JJ"):
                    cats["adj"] += 1
                elif tag.startswith("RB"):
                    cats["adv"] += 1
                elif tag in ("IN", "TO"):
                    cats["prep"] += 1
                elif tag.startswith("PRP"):
                    cats["pron"] += 1
                else:
                    cats["other"] += 1
            return cats
        except Exception:
            pass

    # Heuristic fallback based on common suffixes
    cats = Counter()
    for w in words:
        lw = w.lower()
        if lw.endswith("ly"):
            cats["adv"] += 1
        elif lw.endswith(("tion", "ment", "ness", "ity", "ance", "ence")):
            cats["noun"] += 1
        elif lw.endswith(("ing", "ed")):
            cats["verb"] += 1
        elif lw.endswith(("ful", "able", "ive", "ous", "al", "ic")):
            cats["adj"] += 1
        elif lw in FUNCTION_WORDS:
            cats["prep"] += 1
        else:
            cats["other"] += 1
    return cats


def extract_stylometric_features(
    text: str,
    context_text: Optional[str] = None
) -> np.ndarray:
    """
    Extract the 33 stylometric features from text.
    If context_text is provided, context features augment the representation.
    """
    text = text.strip()
    if not text:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    char_count = len(text)
    raw_words = re.findall(r'\b[A-Za-z0-9\'-]+\b', text)
    word_count = len(raw_words)

    if word_count == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    words_lower = [w.lower() for w in raw_words]
    word_lengths = [len(w) for w in raw_words]
    avg_word_len = sum(word_lengths) / word_count

    # Lexical Diversity & Vocabulary Richness
    vocab = Counter(words_lower)
    types_count = len(vocab)
    ttr = types_count / word_count
    root_ttr = types_count / math.sqrt(word_count)
    corr_ttr = types_count / math.sqrt(2 * word_count)
    log_ttr = (math.log(types_count) / math.log(word_count)) if word_count > 1 and types_count > 0 else 1.0

    # Simpson's Diversity Index D
    if word_count > 1:
        simpsons_d = sum(cnt * (cnt - 1) for cnt in vocab.values()) / (word_count * (word_count - 1))
    else:
        simpsons_d = 0.0

    hapax_count = sum(1 for cnt in vocab.values() if cnt == 1)
    dis_count = sum(1 for cnt in vocab.values() if cnt == 2)
    hapax_ratio = hapax_count / word_count
    dis_ratio = dis_count / word_count

    # Punctuation & Orthographic Patterns
    punct_count = sum(1 for ch in text if ch in string.punctuation)
    punct_ratio = punct_count / char_count if char_count > 0 else 0.0
    comma_density = text.count(",") / word_count
    period_density = text.count(".") / word_count
    semicolon_density = text.count(";") / word_count
    colon_density = text.count(":") / word_count
    question_density = text.count("?") / word_count
    exclamation_density = text.count("!") / word_count
    dash_density = (text.count("-") + text.count("—") + text.count("–")) / word_count
    quote_density = (text.count('"') + text.count("'")) / word_count

    uppercase_count = sum(1 for ch in text if ch.isupper())
    uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0.0
    digit_count = sum(1 for ch in text if ch.isdigit())
    digit_ratio = digit_count / char_count if char_count > 0 else 0.0

    # Function Words & Linguistic Patterns
    func_count = sum(1 for w in words_lower if w in FUNCTION_WORDS)
    func_ratio = func_count / word_count

    first_pers_count = sum(1 for w in words_lower if w in FIRST_PERSON_PRONOUNS)
    third_pers_count = sum(1 for w in words_lower if w in THIRD_PERSON_PRONOUNS)
    pronoun_ratio = (first_pers_count + third_pers_count) / word_count
    first_person_ratio = first_pers_count / word_count
    third_person_ratio = third_pers_count / word_count

    # Transitional & Discourse Markers (characteristic of structured LLM outputs)
    text_lower = text.lower()
    words_set = set(words_lower)
    trans_count = 0
    for marker in TRANSITIONAL_CONJUNCTIONS:
        if " " in marker:
            if re.search(r"\b" + re.escape(marker) + r"\b", text_lower):
                trans_count += 1
        else:
            if marker in words_set:
                trans_count += 1
    trans_ratio = trans_count / max(1.0, float(word_count))

    # POS Distribution
    pos_counts = _extract_pos_tags(raw_words)
    noun_ratio = pos_counts["noun"] / word_count
    verb_ratio = pos_counts["verb"] / word_count
    adj_ratio = pos_counts["adj"] / word_count
    adv_ratio = pos_counts["adv"] / word_count
    prep_density = pos_counts["prep"] / word_count

    # Complexity: clauses / conjunction indicators
    clauses = len(re.findall(r'[,;:]|\b(?:because|although|since|while|which|that|who|whom|whose|where|when|if|unless|whereas)\b', text_lower))
    complexity = clauses / max(1.0, float(word_count))

    # Readability: Syllables per word
    syllables = sum(_fast_syllable_count(w) for w in raw_words)
    avg_syllables = syllables / word_count

    # Burstiness: humans vary sentence length more than LLMs
    sent_lens = [len(x.split()) for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if len(sent_lens) >= 2:
        sent_len_cv = float(np.std(sent_lens) / max(1.0, np.mean(sent_lens)))
    else:
        sent_len_cv = 0.0
    sent_len_mean = float(np.mean(sent_lens)) if sent_lens else float(word_count)

    # Repeated bigrams (LLMs reuse phrasing patterns)
    bigrams = list(zip(words_lower, words_lower[1:]))
    rep_bigram = (1 - len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

    # Contractions (humans use more in informal text)
    contraction_ratio = sum(1 for w in words_lower if "'" in w) / word_count

    features = [
        avg_word_len,
        float(word_count),
        float(char_count),
        ttr,
        root_ttr,
        corr_ttr,
        log_ttr,
        simpsons_d,
        hapax_ratio,
        dis_ratio,
        punct_ratio,
        comma_density,
        period_density,
        semicolon_density,
        colon_density,
        question_density,
        exclamation_density,
        dash_density,
        quote_density,
        uppercase_ratio,
        digit_ratio,
        func_ratio,
        pronoun_ratio,
        first_person_ratio,
        third_person_ratio,
        trans_ratio,
        prep_density,
        noun_ratio,
        verb_ratio,
        adj_ratio,
        adv_ratio,
        complexity,
        avg_syllables,
        sent_len_cv,
        sent_len_mean,
        rep_bigram,
        contraction_ratio,
    ]

    # Context blending: only blend the *rate* features (index >= 3); never blend raw
    # length counts (word_count / char_count), which would be meaningless averages.
    if context_text and len(words_lower) < 15:
        context_feats = extract_stylometric_features(context_text, context_text=None)
        blended = []
        for i, (f, cf) in enumerate(zip(features, context_feats)):
            blended.append(f if i in LENGTH_FEATURE_IDX else 0.6 * f + 0.4 * cf)
        features = blended

    return np.array(features, dtype=np.float32)


def extract_feature_matrix(
    texts: List[str],
    contexts: Optional[List[str]] = None
) -> np.ndarray:
    """Extract feature matrix (N, len(FEATURE_NAMES)) for a list of texts."""
    matrix = []
    for i, t in enumerate(texts):
        ctx = contexts[i] if contexts is not None else None
        matrix.append(extract_stylometric_features(t, context_text=ctx))
    return np.array(matrix, dtype=np.float32)
