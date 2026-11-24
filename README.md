# PlagiarismAI

A Streamlit-based document analysis system with two independent modules:

| Module | Capability |
|---|---|
| **A – Plagiarism Detection** | Exact, near-exact, semantic, hybrid TF-IDF + MiniLM, sliding-window chunking, local reference database, citation-aware |
| **B – AI-Generated Text Detection** | Sentence-level XGBoost classifier, 33 stylometric features, confidence estimates, highlighted results |

> **Disclaimer:** Similarity scores are not proof of plagiarism. AI-detection results are probabilistic and are not proof of AI authorship. Always review results manually.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Features](#features)
6. [Datasets](#datasets)
7. [Benchmark Results](#benchmark-results)
8. [Running Tests](#running-tests)
9. [CLI Tools](#cli-tools)
10. [Project Structure](#project-structure)
11. [Limitations](#limitations)

---

## Architecture

```
app.py  (Streamlit UI – 6 tabs)
│
├── Module A – Plagiarism Detection
│   ├── Exact matching
│   ├── Near-exact (difflib.SequenceMatcher)
│   ├── Semantic (Sentence-BERT all-MiniLM-L6-v2 + cosine similarity)
│   ├── Evidence Gate (IDF-weighted lexical evidence)
│   ├── plagiarism/chunking.py      – Sliding-window passage chunking
│   ├── plagiarism/hybrid_similarity.py – TF-IDF + MiniLM blended score
│   ├── plagiarism/citations.py     – Citation extraction & status
│   ├── plagiarism/reference_search.py – Local reference database
│   └── plagiarism/webis_loader.py  – Webis-CPC-11 dataset loader
│
├── Module B – AI-Generated Text Detection
│   ├── ai_detect/dataset_loader.py – ChatGPT-Human-NewGenerated loader
│   ├── ai_detect/features.py       – 33 stylometric features
│   ├── ai_detect/train.py          – XGBoost + GroupKFold training
│   ├── ai_detect/predict.py        – Sentence & document predictions
│   └── ai_detect/reporting.py      – PDF report generation
│
└── Storage
    ├── storage/sqlite_backend.py   – Default (zero-config)
    └── storage/mongo_backend.py    – Optional (set MONGODB_URI)
```

---

## Installation

**Requirements:** Python 3.9+

```powershell
python -m pip install -r requirements.txt
```

The first run downloads:
- Sentence-BERT model: `all-MiniLM-L6-v2` (~90 MB, cached by HuggingFace)
- NLTK punkt tokenizer (for sentence splitting)

---

## Quick Start

```powershell
# 1. Copy and edit the environment file
Copy-Item .env.example .env
# (Add at least one search API key – see Configuration)

# 2. Train the AI detection model (one-time, ~20 s)
python -m ai_detect.train

# 3. Launch the app
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Configuration

Edit `.env` (never commit it):

```ini
# --- Web Search (at least one required for online source discovery) ---
TAVILY_API_KEY=tvly-...           # https://tavily.com (recommended)
BRAVE_API_KEY=BSA...              # https://brave.com/search/api (alternative)
# DuckDuckGo is used as a free fallback if neither key is set

# --- Storage (MongoDB optional; SQLite is the default) ---
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=plagiarism_ai

# --- HuggingFace offline mode (set if no internet on deployment host) ---
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

---

## Features

### Tab 1 – Scan Document
Upload a PDF, DOCX, or TXT file and run plagiarism detection.

| Setting | Description |
|---|---|
| Similarity threshold | Minimum cosine similarity to flag a match |
| Search providers | Tavily, Brave, DuckDuckGo |
| Include Local Refs | Search the local reference document database |
| Sliding-window chunking | Splits text into overlapping windows for passage-level context |
| Hybrid TF-IDF + MiniLM | Blends TF-IDF and MiniLM scores (α = 0.75 MiniLM by default) |

Results include:
- Overall similarity percentage
- Exact / near-exact / semantic match breakdown
- Citation status per match (Cited / Uncited / Unclear)
- Source URLs and evidence snippets
- Downloadable PDF report

### Tab 2 – AI Detection
Upload a document and get sentence-level AI-generation predictions.

- Three categories: **Likely Human** / **Needs Review** / **Potentially AI-generated**
- Confidence score per sentence
- Color-coded highlighted text
- Downloadable PDF report with per-sentence table

> **Important:** Train the model first with `python -m ai_detect.train`. The trained model is saved to `outputs/models/ai_detection/`.

### Tab 3 – Reference Library
Upload documents to the local reference database. Future scans will compare against them.

- Upload PDF, DOCX, or TXT
- Index into chunks with MiniLM embeddings
- List and delete indexed documents

### Tab 4 – Compare Documents
Upload two documents and compare them directly.

### Tab 5 – History
Browse previous scan results stored in SQLite (or MongoDB).

### Tab 6 – About
Benchmark results, methodology, and known limitations.

---

## Datasets

### Webis-CPC-11 (Plagiarism / Paraphrase Benchmark)

Location: `data/Webis-CPC-11/`

| Property | Count |
|---|---:|
| Total records | 7,859 |
| Positive (paraphrase) | 4,067 |
| Negative (not paraphrase) | 3,792 |

Each record has three files: `{id}-original.txt`, `{id}-paraphrase.txt`, `{id}-metadata.txt`.

> **Limitation:** This is a paraphrase-recognition dataset, not a full document-level plagiarism benchmark. Results reflect paraphrase detection, not the entire online search pipeline.

### ChatGPT-Human-NewGenerated (AI Detection)

Location: `data/ChatGPT-Human-NewGenerated/`

| Folder | Contents |
|---|---|
| Training-Problems | 81 mixed documents |
| Training-Truth | Paragraph-level Human / ChatGPT annotations |
| Testing | 19 documents (no ground-truth labels) |

Based on: *Unveiling ChatGPT Text Using Writing Style* (2024).

> **Limitation:** Paragraph-level annotations are the primary ground truth. Sentence labels are inherited from their paragraph (weak supervision). Testing documents have no annotations; test metrics are not reported.

---

## Benchmark Results

### Plagiarism Detection – Webis-CPC-11 (1,500 samples, 225 test)

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Exp 1 – Raw MiniLM | 73.8% | 73.2% | 100.0% | 0.845 | 0.550 |
| Exp 2 – MiniLM + Lexical | 73.8% | 75.3% | 94.4% | 0.838 | 0.478 |
| Exp 3 – Hybrid TF-IDF + MiniLM | 73.8% | 73.2% | 100.0% | 0.845 | 0.546 |

> The low ROC-AUC reflects the fact that MiniLM was not trained specifically for paraphrase discrimination on this dataset. The high recall / F1 is partly due to dataset imbalance and threshold tuning. These numbers are **NOT** an accuracy measurement of the full online detection pipeline.

### AI Detection – ChatGPT-Human-NewGenerated (Document-grouped 5-fold CV)

| Evaluation Unit | Ground Truth | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Paragraph (primary) | ✅ Verified | 79.7% ± 5.1% | 0.803 ± 0.054 | 0.874 |
| Sentence (weak labels) | ⚠️ Inherited | 74.7% | 0.702 | 0.823 |

**Model:** XGBoost, 33 stylometric features, GroupKFold(n_splits=5) on 391 paragraphs / 1,499 sentences.

---

## Running Tests

```powershell
python -m unittest discover -s tests -v
# Expected: 61 tests, 0 failures, 1 skipped — OK
```

The skipped test requires a real database path (`PLAGIARISM_DB_PATH` env var).

### Individual test modules

```powershell
python -m unittest tests.test_chunking -v
python -m unittest tests.test_hybrid_similarity -v
python -m unittest tests.test_citations -v
python -m unittest tests.test_storage -v
python -m unittest tests.test_ai_detect -v
python -m unittest tests.test_webis_loader -v
python -m unittest tests.test_semantic_evidence -v
python -m unittest tests.test_semantic_reporting -v
python -m unittest tests.test_validation_script_smoke -v
```

---

## CLI Tools

### Train AI detection model

```powershell
python -m ai_detect.train
# Outputs: outputs/models/ai_detection/xgboost_sentence_detector.joblib
#          outputs/models/ai_detection/feature_scaler.joblib
#          outputs/models/ai_detection/model_metadata.json
```

### Evaluate plagiarism model on Webis-CPC-11

```powershell
# Smoke test only (uses fake embedder, fast)
python validate_real_model.py --smoke

# Full Webis evaluation (1500 records, requires model download ~1 min)
python validate_real_model.py --webis --webis-samples 1500

# Full dataset (7859 records, ~10 min)
python validate_real_model.py --webis
```

Output files are saved to `outputs/evaluation/plagiarism/`.

### Diagnose a saved scan

```powershell
python diagnose_sentence.py
```

Requires a scan to be saved in the database first. Exports annotated pages to `diag_pages/`.

### Migrate SQLite → MongoDB

```powershell
python -m storage.migration --sqlite-path plagiarism_history.db --verify
```

Options:

| Flag | Description |
|---|---|
| `--sqlite-path` | Path to SQLite database (default: `plagiarism_history.db`) |
| `--verify` | Re-read migrated documents and confirm counts |
| `--dry-run` | Print what would be migrated without writing |

---

## Project Structure

```
PlagiarismAI-main/
├── app.py                      Main Streamlit application
├── diagnose_sentence.py        Scan diagnostic tool
├── validate_real_model.py      Webis evaluation + smoke test
├── requirements.txt
├── .env.example
├── .gitignore
│
├── plagiarism/                 Plagiarism module
│   ├── __init__.py
│   ├── webis_loader.py         Webis-CPC-11 dataset loader
│   ├── chunking.py             Sliding-window chunking
│   ├── hybrid_similarity.py    Hybrid TF-IDF + MiniLM scorer
│   ├── citations.py            Citation extraction & status
│   └── reference_search.py    Local reference database search
│
├── ai_detect/                  AI detection module
│   ├── __init__.py
│   ├── dataset_loader.py       ChatGPT-Human-NewGenerated loader
│   ├── features.py             33 stylometric features
│   ├── train.py                XGBoost training pipeline
│   ├── predict.py              Sentence & document prediction
│   └── reporting.py            PDF report generation
│
├── storage/                    Storage abstraction layer
│   ├── __init__.py
│   ├── base.py                 Abstract StorageBackend
│   ├── sqlite_backend.py       SQLite implementation (default)
│   ├── mongo_backend.py        MongoDB implementation (optional)
│   ├── repository.py           Factory: get_storage()
│   └── migration.py            SQLite → MongoDB migration CLI
│
├── tests/
│   ├── support.py              Shared utilities, FakeEmbedder
│   ├── test_semantic_evidence.py
│   ├── test_semantic_reporting.py
│   ├── test_validation_script_smoke.py
│   ├── test_chunking.py
│   ├── test_hybrid_similarity.py
│   ├── test_citations.py
│   ├── test_storage.py
│   ├── test_ai_detect.py
│   └── test_webis_loader.py
│
├── data/
│   ├── Webis-CPC-11/           Paraphrase benchmark dataset
│   └── ChatGPT-Human-NewGenerated/  AI detection dataset
│
└── outputs/                    Generated at runtime (git-ignored)
    ├── models/ai_detection/    Trained model artifacts
    └── evaluation/plagiarism/  Webis evaluation results
```

---

## Limitations

1. **Paraphrase ≠ plagiarism pipeline accuracy.** Webis-CPC-11 measures paraphrase detection only. Online detection accuracy depends on search API quality and source availability.

2. **AI detection is probabilistic.** Stylometric features cannot prove authorship. Results are for review assistance only.

3. **Sentence labels are noisy.** Sentence-level AI detection uses paragraph-inherited labels (weak supervision). Paragraph-level metrics are primary.

4. **No test ground truth for AI detection.** The 19 ChatGPT-Human testing documents have no annotations. Test-set metrics are not reported.

5. **Search API keys required** for online source discovery (Tavily / Brave). DuckDuckGo fallback has rate limits.

6. **MongoDB is optional.** SQLite is the default and requires no configuration.

7. **Plagiarism ≠ AI-generated.** The two modules are independent. A plagiarism match does not imply AI generation and vice versa.
