"""
Stylometric Rule-Based AI Scorer
=================================
A zero-shot, training-free scorer that estimates P(AI-generated) purely from
writing style signals that reliably separate ChatGPT/LLM text from human text.

Key signals used (each contributing to a logit):
  + High average syllables per word  (AI uses polysyllabic vocabulary)
  + High transitional marker ratio   (AI: furthermore, consequently, moreover…)
  + Low first-person pronoun ratio   (AI rarely uses "I", "my", "we")
  + Low contraction ratio            (AI rarely uses don't, can't, it's)
  + Low sentence-length variability  (AI produces uniform sentence lengths)
  + High avg sentence length         (AI sentences are longer)
  + Low function-word ratio          (AI uses dense content words)
  + Presence of AI-attractor words   (ChatGPT lexicon signature)
  + Absence of human-voice markers   (casual speech, hedges, emotions)
  + Passive voice ratio              (AI uses passive constructions more)
  + AI structural patterns           (definition phrases, listing patterns)

Returns a probability in [0, 1]. Values are carefully calibrated so that:
  - Clear ChatGPT output: 0.75-0.99
  - Ambiguous / borderline:  0.40-0.65
  - Clear human informal text: 0.02-0.25
"""
from __future__ import annotations

import re
import math
from typing import List

import numpy as np

from ai_detect.features import (
    extract_stylometric_features,
    FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# AI attractor vocabulary – words / phrases strongly associated with LLM output
# ---------------------------------------------------------------------------
_AI_LEXICON = {
    # Formal discourse markers (very strong AI signals)
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "nevertheless", "nonetheless", "accordingly", "thus", "hence",
    "in conclusion", "in summary", "notably", "significantly", "importantly",
    "overall", "ultimately", "essentially", "specifically", "particularly",
    "henceforth", "heretofore", "aforementioned", "notwithstanding",
    "subsequently", "concurrently", "simultaneously",
    # LLM buzzwords (overused by ChatGPT)
    "indispensable", "pivotal", "paramount", "multifaceted", "holistic",
    "seamless", "tapestry", "landscape", "beacon", "testament", "realm",
    "foster", "garner", "nuanced", "vibrant", "comprehensive", "bolster",
    "revolutionize", "cornerstone", "imperative", "leverage", "empower",
    "facilitate", "encompass", "underscore", "highlight", "utilize",
    "implement", "demonstrate", "ensure", "enhance", "optimize", "integrate",
    "innovative", "cutting-edge", "state-of-the-art", "robust", "dynamic",
    "diverse", "crucial", "essential", "vital", "fundamental", "critical",
    "significant", "substantial", "considerable", "remarkable", "notable",
    "evident", "apparent", "profound", "unprecedented",
    "sophisticated", "intricate", "multifarious", "multidimensional",
    "paradigm", "framework", "ecosystem", "synergy", "methodology",
    "perspective", "dimension", "aspect", "component", "element",
    "encompassing", "pertaining", "constitute", "comprising",
    "pivotal", "instrumental", "overarching", "underlying", "inherent",
    "ubiquitous", "pervasive", "prevalent", "emerging", "evolving",
    "transformative", "revolutionary", "groundbreaking", "pioneering",
    "sustainable", "equitable", "inclusive", "accessible", "scalable",
    "innovative", "disruptive", "exponential", "trajectory", "catalyst",
    "stakeholder", "proactive", "synergistic", "iterative", "streamline",
    "bandwidth", "deliverable", "actionable", "impactful",
    # Informational/educational AI patterns
    "refers to", "is defined as", "can be defined", "is characterized by",
    "is widely used", "is commonly used", "is often used",
    "plays a crucial", "plays an important", "plays a significant",
    "it is important", "it is worth noting", "it should be noted",
    "as a result", "in order to", "with respect to", "in terms of",
    "in addition to", "as well as", "in contrast to", "on the other hand",
    "in conclusion", "in summary", "to summarize", "to conclude",
    "has been shown", "research has shown", "studies have shown",
    "is associated with", "is known as", "has been", "have been",
    "one of the most", "one of the key", "one of the primary",
    "it is essential", "it is crucial", "it is important to note",
    "in recent years", "over the years", "throughout history",
    "in the context of", "with the advent of", "in today's",
    "primary driver", "key factor", "major factor", "critical role",
    "wide range", "wide variety", "broad range", "numerous",
    "encompasses", "encompass", "involves", "include",
    # Passive and impersonal constructions
    "is used to", "are used to", "can be used", "can be applied",
    "is considered", "are considered", "is believed", "is thought",
    "it can be argued", "it can be seen", "it is clear",
    "has led to", "have led to", "has resulted in",
    "is composed of", "consists of", "is made up of",
    # Academic / expository pattern
    "for example", "for instance", "such as", "including",
    "in particular", "specifically", "namely",
    "however", "although", "while", "whereas", "despite",
    "in contrast", "on the contrary",
}

# ---------------------------------------------------------------------------
# Human voice markers – words strongly associated with human, casual writing
# ---------------------------------------------------------------------------
_HUMAN_LEXICON = {
    # First-person / personal (strong human signals)
    "i", "me", "my", "myself", "i'm", "i've", "i'd", "i'll",
    "we", "us", "our", "we're", "we've", "we'd", "we'll",
    # Informal / emotional
    "actually", "honestly", "literally", "basically", "seriously", "totally",
    "pretty", "really", "kinda", "sorta", "anyway", "anyways",
    "yeah", "yep", "nope", "nah", "ok", "okay",
    "hey", "hi", "hello", "thanks", "thank you",
    "lol", "haha", "omg", "wtf", "imo", "tbh", "ngl",
    # Contractions (separately checked via contraction_ratio feature)
    "don't", "doesn't", "didn't", "can't", "won't", "wouldn't", "couldn't",
    "shouldn't", "haven't", "hasn't", "hadn't", "isn't", "aren't", "wasn't",
    "weren't", "it's", "that's", "there's", "here's", "what's", "who's",
    "gonna", "gotta", "wanna", "lemme", "gimme",
    # Hedges / uncertainty
    "maybe", "perhaps", "probably", "possibly", "might", "guess", "think",
    "feel", "felt", "thought", "heard", "saw", "noticed",
    # Casual discourse
    "so", "like", "just", "even", "stuff", "things", "thing",
    "got", "get", "went", "came", "said", "told", "asked",
    "super", "awesome", "cool", "great", "bad", "weird", "nice",
}

# ---------------------------------------------------------------------------
# Passive voice patterns (AI overuses passive constructions)
# ---------------------------------------------------------------------------
_PASSIVE_PATTERN = re.compile(
    r'\b(?:is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE
)

# ---------------------------------------------------------------------------
# AI structural patterns (definition/expository sentences typical of LLMs)
# ---------------------------------------------------------------------------
_AI_STRUCT_PATTERNS = [
    # Definition patterns
    r'\b\w+\s+(?:is|are)\s+(?:a|an|the)\s+\w+',   # "X is a Y"
    r'\brefers\s+to\b',                              # "refers to"
    r'\bis\s+defined\s+as\b',                        # "is defined as"
    r'\bknown\s+as\b',                               # "known as"
    r'\bconsists?\s+of\b',                           # "consists of"
    r'\bcomprises?\b',                               # "comprises"
    # Enumeration patterns (AI loves lists)
    r'\bincluding\b',                                # "including X, Y, Z"
    r'\bsuch\s+as\b',                               # "such as X, Y"
    r'\bnamely\b',                                   # "namely X"
    r'\b(?:first|second|third|finally|lastly)\b',   # "First...Second...Finally"
    # Citation patterns (AI cites studies/research without real sources)
    r'\bstudies\s+have\s+(?:shown|demonstrated|found)\b',
    r'\bresearch\s+(?:has|have)\s+(?:shown|demonstrated|found|indicated)\b',
    r'\baccording\s+to\b',
    r'\bevidence\s+suggests\b',
    r'\bit\s+is\s+(?:widely|commonly|generally)\s+(?:accepted|known|believed|recognized)\b',
]
_AI_STRUCT_RE = [re.compile(p, re.IGNORECASE) for p in _AI_STRUCT_PATTERNS]


def _idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


# Pre-computed feature indices
_IDX_AVG_WORD_LEN  = _idx("f01_avg_word_length")
_IDX_FUNC_RATIO    = _idx("f22_function_word_ratio")
_IDX_FIRST_PERS    = _idx("f24_first_person_ratio")
_IDX_TRANS_RATIO   = _idx("f26_transitional_word_ratio")
_IDX_SYLLABLES     = _idx("f33_avg_syllables_per_word")
_IDX_SENT_CV       = _idx("f34_sentence_len_cv")
_IDX_SENT_MEAN     = _idx("f35_sentence_len_mean")
_IDX_CONTR_RATIO   = _idx("f37_contraction_ratio")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _lexicon_hits(text_lower: str, words_set: set, lexicon: set) -> int:
    """Count how many items from lexicon appear in the text."""
    hits = 0
    for item in lexicon:
        if " " in item:
            if item in text_lower:
                hits += 1
        else:
            if item in words_set:
                hits += 1
    return hits


def rule_score(text: str) -> float:
    """
    Returns P(AI-generated) in [0, 1] based purely on stylometric rules.

    This is NOT a trained classifier — it uses hand-calibrated weights
    derived from known differences between LLM and human writing style.
    """
    if not text or not text.strip():
        return 0.5

    feats = extract_stylometric_features(text)
    words = re.findall(r"\b[A-Za-z'-]+\b", text.lower())
    words_set = set(words)
    text_lower = text.lower()
    word_count = max(1, len(words))

    # -----------------------------------------------------------------------
    # 1. Vocabulary-level signals
    # -----------------------------------------------------------------------
    ai_hits  = _lexicon_hits(text_lower, words_set, _AI_LEXICON)
    hu_hits  = _lexicon_hits(text_lower, words_set, _HUMAN_LEXICON)

    # Normalise by text length (per 100 words)
    ai_rate  = ai_hits  / word_count * 100
    hu_rate  = hu_hits  / word_count * 100

    logit = 0.0

    # AI vocab signal (each hit per 100 words adds logit; saturates)
    logit += min(ai_rate * 0.35, 3.0)
    # Human vocab signal (each hit is a strong negative indicator)
    logit -= min(hu_rate * 0.60, 3.5)

    # -----------------------------------------------------------------------
    # 2. Stylometric feature signals
    # -----------------------------------------------------------------------
    # Transitional marker density  (AI: high)
    trans = float(feats[_IDX_TRANS_RATIO])
    logit += min(trans * 22.0, 2.0)

    # First-person ratio  (AI: near-zero)
    first_pers = float(feats[_IDX_FIRST_PERS])
    if first_pers < 0.005:
        logit += 0.9
    elif first_pers < 0.03:
        logit += 0.1
    elif first_pers < 0.07:
        logit -= 0.7
    else:
        logit -= 1.8

    # Contraction ratio  (AI: near-zero; humans: higher)
    contr = float(feats[_IDX_CONTR_RATIO])
    if contr < 0.005:
        logit += 0.6
    elif contr > 0.025:
        logit -= 1.2
    elif contr > 0.01:
        logit -= 0.4

    # Average syllables per word  (AI: higher, > 1.7 is unusual for human casual)
    syl = float(feats[_IDX_SYLLABLES])
    # Reference: human casual ~1.4, human academic ~1.6, AI ~1.8-2.5
    logit += max(min((syl - 1.55) * 2.8, 2.5), -1.0)

    # Sentence length variability (CV): AI produces uniform sentences
    cv = float(feats[_IDX_SENT_CV])
    # Human CV ~ 0.30+; AI CV < 0.20 → positive logit
    logit += max(min((0.28 - cv) * 4.5, 1.8), -1.5)

    # Average sentence length  (AI tends to write longer sentences)
    sent_mean = float(feats[_IDX_SENT_MEAN])
    if sent_mean > 28:
        logit += 0.8
    elif sent_mean > 22:
        logit += 0.5
    elif sent_mean > 18:
        logit += 0.2
    elif sent_mean < 12:
        logit -= 0.5

    # Function word ratio (AI: fewer function words relative to content)
    func = float(feats[_IDX_FUNC_RATIO])
    # Human ~ 0.50; AI ~ 0.35-0.45
    logit += max(min((0.47 - func) * 3.5, 1.2), -1.0)

    # -----------------------------------------------------------------------
    # 3. Passive voice patterns (AI overuses passive)
    # -----------------------------------------------------------------------
    passive_matches = len(_PASSIVE_PATTERN.findall(text))
    passive_rate = passive_matches / word_count * 100
    logit += min(passive_rate * 0.25, 1.0)

    # -----------------------------------------------------------------------
    # 4. AI structural patterns (definition/expository phrases)
    # -----------------------------------------------------------------------
    struct_hits = sum(1 for r in _AI_STRUCT_RE if r.search(text_lower))
    logit += min(struct_hits * 0.18, 1.5)

    # -----------------------------------------------------------------------
    # 5. Convert to probability and return
    # -----------------------------------------------------------------------
    # logit=0 → 0.50; logit=3 → 0.95; logit=-3 → 0.05
    return float(_sigmoid(logit))


def rule_score_batch(texts: List[str]) -> np.ndarray:
    """Vectorised wrapper — returns np.ndarray of shape (N,)."""
    return np.array([rule_score(t) for t in texts], dtype=np.float64)
