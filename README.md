# PlagiarismAI

An end-to-end, dual-pipeline academic document verification system featuring two strictly isolated analysis engines:

| Module | Core Capability | Underpinning Architecture |
| :--- | :--- | :--- |
| **Module A – Multi-Tier Plagiarism Verification** | Exact, near-exact, and semantic plagiarism detection with citation awareness and false-positive suppression | SHA-256 Hashing, `difflib.SequenceMatcher`, Sentence-BERT (`all-MiniLM-L6-v2`), Hybrid TF-IDF, Context Alignment & IDF-Weighted Lexical Coverage Gates |
| **Module B – Stylometric AI Text Attribution** | Paragraph and sentence-level AI text detection with confidence scoring and interpretability breakdown | 37-dimensional stylometric feature extractor (35 active features), XGBoost classifier, sparse lexical n-gram model, zero-shot heuristic rule engine, and adaptive cascading fusion |

> **Academic & Ethical Disclaimer:** Plagiarism similarity percentages quantify textual and contextual overlap; they do not constitute definitive legal proof of academic dishonesty. AI attribution predictions represent calibrated probabilistic estimates and should serve as advisory evidence to guide manual editorial review.

---

## Table of Contents

1. [Architecture & System Design](#architecture--system-design)
2. [Key Features](#key-features)
3. [Plagiarism Detection Pipeline](#plagiarism-detection-pipeline)
4. [AI Text Detection Pipeline](#ai-text-detection-pipeline)
5. [Installation & Setup](#installation--setup)
6. [Quick Start & Execution Modes](#quick-start--execution-modes)
7. [Configuration (.env)](#configuration-env)
8. [Experimental Benchmarks & Datasets](#experimental-benchmarks--datasets)
9. [Running Tests](#running-tests)
10. [Project Structure](#project-structure)
11. [System Invariants & Limitations](#system-invariants--limitations)

---

## Architecture & System Design

PlagiarismAI is built on an **Independent Dual-Pipeline Principle**: the plagiarism scanning workflow and the AI stylometric engine operate on completely separated execution paths with **zero score sharing or pollution**.

```
PlagiarismAI
│
├── Streamlit Web UI (app.py) & Headless / Colab Engine (plagiarism_ai_colab.py)
│
├── Module A: Plagiarism Detection Pipeline
│   ├── Pre-processing: citations.py (Numeric [1] & Author-Year (Smith, 2020) preservation)
│   ├── Segmentation: chunking.py (Sliding-window 3-sentence chunks, 180-word token budget)
│   ├── Candidate Retrieval:
│   │   ├── Local Reference DB: reference_search.py (Pre-computed 384-dim MiniLM vectors)
│   │   └── Real-time Web Search: Tavily API, Brave API, DuckDuckGo search + scraping
│   ├── 3-Tier Hierarchical Matching Cascade:
│   │   ├── Tier 1: Exact Hash & Normalized Match (100% verbatim)
│   │   ├── Tier 2: Near-Exact Lexical Match (difflib SequenceMatcher >= 0.90)
│   │   └── Tier 3: Dense Semantic Vector Match (Sentence-BERT all-MiniLM-L6-v2 >= 0.70)
│   │               [Optional: hybrid_similarity.py (alpha * MiniLM + (1-alpha) * TF-IDF)]
│   ├── Contextual Evidence Gate (False Positive Mitigation):
│   │   ├── Sequential Context Gate (Adjacency alignment at similarity >= 0.60)
│   │   ├── Lexical Distinctiveness Gate (Distinctive IDF token coverage >= 0.25, shared >= 2)
│   │   └── Common-Source Suppression (Candidates in >= 3 sources -> uncounted Topical Similarity)
│   └── Invariant Metric Aggregator (summarize_matches(): One-status-per-sentence)
│
├── Module B: Stylometric AI Text Attribution Pipeline
│   ├── Pre-processing: Paragraph chunking (40 to 220 words, short paragraph merging)
│   ├── Feature Extraction: features.py (37 stylometric, syntactic, and structural metrics)
│   │   └── Feature Filtering: Drop raw length features (f02, f03) -> 35 active features
│   ├── Primary Local Detection Ensemble (predict.py):
│   │   ├── Model A: XGBoost Classifier on 35 StandardScaled stylometric features (P_style)
│   │   ├── Model B: Sparse Lexical N-Gram Model (P_lex)
│   │   │   └── Model Blend: P_model = 0.25 * P_style + 0.75 * P_lex
│   │   └── Model C: Stylometric Heuristic Scorer (stylometric_scorer.py -> P_rule)
│   ├── Cascading Adaptive Fusion Gate:
│   │   ├── P_rule >= 0.60  ==> max(P_model, P_rule)  [Catches OOD ChatGPT text]
│   │   ├── P_rule <= 0.20 & P_model <= 0.35 ==> min(P_model, P_rule) [Protects human text]
│   │   ├── P_model >= 0.75 ==> P_model               [Trusts confident model]
│   │   └── Default         ==> 0.30 * P_model + 0.70 * P_rule
│   ├── Sentence Attribution: Sentences inherit paragraph context probabilities
│   ├── Decision Bands: <= 0.35 Human | 0.35-0.52 Review Band | >= 0.52 AI-Generated
│   └── Optional Cloud Comparator: sapling_api.py (External API with offline fallback)
│
└── Storage Abstraction Layer
    ├── storage/sqlite_backend.py (Zero-config local database: plagiarism_history.db)
    └── storage/mongo_backend.py  (Optional enterprise backend via MONGODB_URI)
```

---

## Key Features

- **Multi-Tier Similarity Cascade:** Classifies plagiarized segments into Exact, Near-Exact, and Semantic categories.
- **Sequential Context Alignment:** Verifies if surrounding sentences also match in chronological order to identify multi-sentence passage copying.
- **IDF-Weighted Distinctive Coverage Gate:** Rejects false semantic positives by requiring candidate pairs to share rare, domain-specific terminology.
- **Common-Source Suppression:** Prevents standard academic definitions and ubiquitous clichéd phrases from inflating similarity percentages by classifying them as uncounted *Topical Similarity*.
- **Token-Budget Sliding Windows:** Overlapping 3-sentence windows constrained to $\le 180$ words prevent silent truncation inside transformer attention layers.
- **FAISS-Free Local Vector Indexing:** Pre-computes 384-dimensional dense vectors and executes batch cosine distance calculations using pure NumPy operations.
- **35-Feature Stylometric Engine:** Evaluates vocabulary richness (TTR, Hapax Legomena, Simpson's D), discourse transition density, sentence complexity, and burstiness without raw token count bias.
- **Zero-Shot Heuristic Safeguard:** Catches novel, out-of-distribution LLM phrasing via an adaptive cascading fusion rule engine.
- **Full Headless & Colab Support:** A standalone script (`plagiarism_ai_colab.py`) allows zero-setup headless execution on Google Colab or remote Linux environments.
- **Annotated PDF Reporting:** Generates comprehensive PDF audit reports with highlighted text, source attributions, and metric breakdowns.

---

## Plagiarism Detection Pipeline

### 1. Ingestion & Citation Extraction (`plagiarism/citations.py`)
Before normalization, regex engines extract citations to distinguish attributed quotes from unacknowledged copying:
- Numeric citations: `\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]` (e.g., `[1]`, `[3-5]`)
- Author-year citations: `(Smith et al., 2020)`
- Bibliography headers: `References`, `Bibliography`, `Works Cited`

### 2. Multi-Tier Matching Cascade (`app.py`, `plagiarism/hybrid_similarity.py`)
1. **Tier 1 (Exact):** Normalized SHA-256 hash and string equality ($100\%$ verbatim match).
2. **Tier 2 (Near-Exact):** Ratcliff/Obershelp Gestalt pattern matching via `difflib.SequenceMatcher` ($\text{ratio} \ge 0.90$).
3. **Tier 3 (Dense Semantic):** Cosine similarity over normalized `sentence-transformers/all-MiniLM-L6-v2` embeddings ($\text{sim} \ge 0.70$).
   - Optional Hybrid Blend:
     $$S_{\text{hybrid}} = \alpha \cdot S_{\text{semantic}} + (1 - \alpha) \cdot S_{\text{tfidf}} \quad (\alpha = 0.75)$$

### 3. Contextual Evidence Gate
Candidates passing Tier 3 are confirmed as **Semantic Matches** only if they meet either:
- **Context Gate:** Sentence $i \pm 1$ matches source passage $j \pm 1$ in sequence with $\text{sim} \ge 0.60$.
- **Distinctive Lexical Gate:**
  $$C(Q, R) = \frac{\sum_{w \in W_{\text{shared}}} \text{IDF}(w)}{\sum_{w \in W_{\text{distinctive}}(Q)} \text{IDF}(w)} \ge 0.25 \quad \text{and} \quad |W_{\text{shared}}| \ge 2$$
  *(where $W_{\text{distinctive}}$ are content tokens occurring in $\le 10\%$ of scan sentences).*
- **Suppression:** Matches appearing in $\ge 3$ distinct sources are flagged as *Topical Similarity* and **omitted from the overall score**.

### 4. Non-Inflation Aggregation
Every sentence is assigned strictly **one winning status**:
$$\text{Priority: } \text{EXACT} > \text{NEAR} > \text{SEMANTIC} > \text{NONE}$$
$$\text{Overall Plagiarism \%} = \frac{N_{\text{exact}} + N_{\text{near}} + N_{\text{semantic}}}{N_{\text{total}}} \times 100$$

---

## AI Text Detection Pipeline

### 1. 37-Dimensional Stylometric Analysis (`ai_detect/features.py`)
Extracts quantitative stylistic signatures across four core categories:
- **Lexical Diversity:** Type-Token Ratio (TTR), Root TTR, Corrected TTR, Log TTR, Simpson's $D$, Hapax Legomena ratio, Dis Legomena ratio.
- **Syntactic & Discourse Markers:** Transitional connectives (*furthermore, moreover, consequently*), function-word ratios, pronoun distributions, and POS distributions (nouns, verbs, adjectives, adverbs).
- **Punctuation & Orthography:** Comma density, semicolon density, colon density, dash ratio, quotation density, uppercase ratio.
- **Rhythm & Burstiness:** Sentence length coefficient of variation ($\text{CV} = \sigma / \mu$), mean sentence length, syllables per word, contraction ratio (*don't, can't*).

> **Length Bias Mitigation:** Features $f_{02}$ (`word_count`) and $f_{03}$ (`char_count`) are explicitly dropped, leaving **35 active features** to ensure predictions are invariant to text length.

### 2. Dual-Engine Local Classification (`ai_detect/predict.py`)
1. **Model Blend ($P_{\text{model}}$):**
   $$P_{\text{model}} = 0.25 \cdot P_{\text{style}} + 0.75 \cdot P_{\text{lex}}$$
   where $P_{\text{style}}$ is generated by an XGBoost model on scaled stylometric vectors, and $P_{\text{lex}}$ is derived from a sparse lexical n-gram model.
2. **Heuristic Rule Engine ($P_{\text{rule}}$):**
   Evaluates AI attractor vocabulary, conversational absence, and structural symmetry, mapping signals to probability via a calibrated sigmoid function:
   $$P_{\text{rule}} = \frac{1}{1 + e^{-\text{logit}}}$$

### 3. Cascading Adaptive Fusion Gate
Reconciles the statistical model with the rule engine to prevent false negatives on unseen generator models:
- **Rule Confident AI ($P_{\text{rule}} \ge 0.60$):** $P_{\text{final}} = \max(P_{\text{model}}, P_{\text{rule}})$
- **Conservative Human Agreement ($P_{\text{rule}} \le 0.20 \text{ and } P_{\text{model}} \le 0.35$):** $P_{\text{final}} = \min(P_{\text{model}}, P_{\text{rule}})$
- **Model Confident AI ($P_{\text{model}} \ge 0.75$):** $P_{\text{final}} = P_{\text{model}}$
- **Uncertain Zone:** $P_{\text{final}} = 0.30 \cdot P_{\text{model}} + 0.70 \cdot P_{\text{rule}}$

### 4. Decision Thresholds & Document Aggregation
- **Sentence Inheritance:** Sentences inherit their parent paragraph's calibrated score ($40 \le \text{words} \le 220$).
- **Document Probability:** Word-weighted average across all chunks $\bar{P}_{\text{doc}} = \frac{\sum W_k P_k}{\sum W_k}$.
- **Thresholds:**
  - $\bar{P} \le 0.35$: **Likely Human**
  - $0.35 < \bar{P} < 0.52$: **Needs Review (Review Band)**
  - $\bar{P} \ge 0.52$: **Potentially AI-generated**
  - Total document $< 50$ words: Flagged as **Inconclusive** due to insufficient sample size.

*(Note: An optional cloud comparator using the Sapling API is available in `ai_detect/sapling_api.py`; if disabled or offline, the local 35-feature ensemble operates as the primary self-contained detector).*

---

## Installation & Setup

### Requirements
- Python 3.9, 3.10, 3.11, or 3.12
- Windows, Linux, or macOS

### 1. Clone the Repository
```bash
git clone https://github.com/Chirag-Nagawade/GenAI-paper-Implimentation-.git
cd GenAI-paper-Implimentation-
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start & Execution Modes

### Mode 1: Interactive Streamlit Web Application
```bash
# 1. Initialize environment variables
cp .env.example .env

# 2. Launch Streamlit
streamlit run app.py
```
The application will launch at `http://localhost:8501`.

### Mode 2: Headless Standalone Script (Google Colab / Server)
To run scans without a GUI or Streamlit server:
```bash
python plagiarism_ai_colab.py
```
This script includes integrated keyless academic search, sliding-window chunking, offline tripartite benchmarks, and automatic generation of `plagiarism_audit_report.pdf`.

---

## Configuration (.env)

Configure your API keys and storage backends in `.env`:

```ini
# --- Web Search (For Online Plagiarism Discovery) ---
TAVILY_API_KEY=tvly-...           # Recommended: https://tavily.com
BRAVE_API_KEY=BSA...              # Alternative: https://brave.com/search/api
# Free fallback: Keyless DuckDuckGo / Wikipedia API is enabled automatically

# --- Storage Configuration ---
# Default: SQLite (zero-setup, stored locally at plagiarism_history.db)
MONGODB_URI=                      # Optional: mongodb://localhost:27017
MONGODB_DB_NAME=plagiarism_ai

# --- HuggingFace Offline Mode (Optional) ---
# Set to 1 if deploying to an isolated server without internet access
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0
```

---

## Experimental Benchmarks & Datasets

### 1. Webis-CPC-11 Tripartite Plagiarism Benchmark
Evaluated on **500 passage pairs (481 unique source groups)** partitioned **group-disjointly** (seed 42) into **349 train / 75 dev / 76 test pairs**:

| Task | Configuration | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Task 1: Paraphrase vs. Distractor** | Raw MiniLM | 0.6508 | 1.0000 | 0.7885 | 0.7861 |
| | Sparse TF-IDF | 0.6406 | 1.0000 | 0.7810 | 0.7550 |
| | **Hybrid (MiniLM + TF-IDF)** | **0.6508** | **1.0000** | **0.7885** | **0.7847** |
| **Task 2: Near-Paraphrase vs. Distractor** | Raw MiniLM | 0.5897 | 0.9583 | 0.7297 | 0.6576 |
| **Task 3: Original vs. Near-Paraphrase** | Raw MiniLM | 0.5789 | 0.9167 | 0.7097 | 0.6277 |

### 2. Independent HC3 AI Detection Evaluation ($N = 2,611$)
Evaluated on **2,611 independent paragraphs** across multiple academic domains (1,539 AI-generated, 1,072 Human-written):

| Metric | Score / Value | Evaluation Breakdown |
| :--- | :---: | :--- |
| **Macro F1 Score** | **0.787** | Balanced performance across human and AI classes |
| **AI Recall (Sensitivity)** | **98.4%** | 1,514 out of 1,539 AI texts correctly identified |
| **Human Recall (Specificity)**| **56.1%** | 632 out of 1,072 Human texts correctly verified |
| **ROC-AUC** | **0.8300** | Area under the Receiver Operating Characteristic curve |
| **Review Band Rate** | **3.06%** | 80 borderline samples appropriately flagged for manual review |

---

## Running Tests

Execute the unit and integration test suite:

```bash
python -m unittest discover -s tests -v
```

### Module-Specific Tests:
```bash
python -m unittest tests.test_chunking -v
python -m unittest tests.test_hybrid_similarity -v
python -m unittest tests.test_citations -v
python -m unittest tests.test_semantic_evidence -v
python -m unittest tests.test_ai_detect -v
python -m unittest tests.test_tripartite_evaluation -v
```

---

## Project Structure

```
PlagiarismAI-main/
├── app.py                          Main multi-tab Streamlit application
├── plagiarism_ai_colab.py          Standalone headless / Google Colab script
├── validate_real_model.py          Benchmark validation & diagnostic runner
├── system_architecture.html        Interactive system architecture diagram
├── requirements.txt                Python package dependencies
├── .env.example                    Template for environment variables
├── .gitignore                      Git ignore configuration
│
├── paper/                          Research manuscript
│   └── PlagiarismAI_IEEE_Paper.docx Academic research paper
│
├── plagiarism/                     Multi-Tier Plagiarism Detection Module
│   ├── chunking.py                 Sliding-window segmentation with token budgeting
│   ├── citations.py                Numeric and author-year citation extraction
│   ├── hybrid_similarity.py        Hybrid TF-IDF + Sentence-BERT similarity scorer
│   ├── reference_search.py         Local reference database indexer (pure NumPy)
│   ├── group_partitioning.py       Group-disjoint partition splitter
│   ├── tripartite_evaluation.py    Tripartite Webis benchmark evaluator
│   └── webis_loader.py             Webis-CPC-11 dataset loader
│
├── ai_detect/                      Stylometric AI Text Detection Module
│   ├── features.py                 37 stylometric feature extractor (35 active)
│   ├── stylometric_scorer.py       Zero-shot heuristic rule engine & AI attractor lexicon
│   ├── predict.py                  Adaptive cascading fusion gate & document aggregator
│   ├── lexical.py                  N-gram lexical richness and vocabulary models
│   ├── dataset_loader.py           Dataset ingestion and preparation
│   ├── sapling_api.py              Optional cloud comparator client with offline fallback
│   └── reporting.py                PDF audit report generator
│
├── storage/                        Storage Abstraction Layer
│   ├── sqlite_backend.py           Default SQLite implementation
│   ├── mongo_backend.py            Optional MongoDB implementation
│   ├── repository.py               Factory method get_storage()
│   └── migration.py                Database migration utilities
│
├── outputs/evaluation/             Experimental Results & Evaluation Artifacts
│   ├── plagiarism/                 Webis ROC/PR curves, thresholds, metrics CSVs
│   ├── phase_5_2c/                 HC3 independent evaluation reports
│   └── phase_5_2d/                 Domain-level error analysis reports
│
├── data/                           Benchmark Datasets
│   ├── Webis-CPC-11/               Webis paraphrase corpus
│   └── ChatGPT-Human-NewGenerated/ Human vs. ChatGPT training corpora
│
└── tests/                          Unit and Integration Test Suite
```

---

## System Invariants & Limitations

1. **Dual-Pipeline Separation:** Plagiarism similarity percentages never influence AI attribution probability, and AI detection results never alter plagiarism similarity scores.
2. **One-Status-Per-Sentence:** Plagiarism metrics never double-count sentences matching multiple sources; each sentence receives exactly one mutually exclusive status ($\text{Exact} > \text{Near} > \text{Semantic} > \text{None}$).
3. **Evidence Gating vs. Raw Similarity:** Cosine similarity above 0.70 is treated as a hypothesis, not proof. Matches lacking sequential context or distinctive token coverage are downgraded to uncounted *Topical Similarity*.
4. **Length Sensitivity in AI Detection:** Texts shorter than 50 words cannot support robust stylometric statistical modeling; such inputs are designated as *Inconclusive*.
5. **Human False Positive Rate:** Highly structured or formulaic technical writing (e.g., standard definitions in finance or computer science) can exhibit low burstiness and trigger elevated AI probability scores, emphasizing the necessity of editorial review.
