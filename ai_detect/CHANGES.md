# ai_detect - what was changed and why

## Root causes of wrong results
1. Production model trained on weak sentence labels (every sentence inherits its paragraph's label, short sentences carry little style signal) -> now trained on verified paragraph labels.
2. predict.py clipped scaled features to [-2.5, 2.5]; training never clipped -> removed.
3. predict.py blended the model 50/50 with hand-written keyword rules (common words like "popular", "different", "including", substring phrases like "is a", "created by") -> removed; model score is used directly.
4. Arbitrary +0.08 "burstiness" boost on every sentence -> removed; burstiness is now a real trained feature (f34).
5. Hand-picked thresholds 0.40/0.55 and scale_pos_weight skewing probabilities -> thresholds tuned from out-of-fold paragraph predictions (AI threshold at <=5% false-positive rate); probabilities isotonic-calibrated.
6. Raw word_count / char_count were model inputs (length shortcut) and were averaged with context -> dropped from the classifier.
7. Syllable counter bug (stripped final consonants) and substring matching for transition phrases -> fixed.
8. NLTK POS vs heuristic POS fallback gives different features; train and deploy with the same setup (install nltk + averaged_perceptron_tagger + punkt).

## Still needed on your side
- Retrain: `python -m ai_detect.train` (old .joblib files are incompatible: feature count changed 33 -> 35 model columns).
- Evaluate on text from a DIFFERENT source than the training set; the current dataset is one narrow domain.
- Texts under ~25 words are marked "Needs Review" by design.

# v3 (second round)
- train.py / predict.py rewritten as a paragraph-level ENSEMBLE: stylometric GBM + lexical word/char n-gram logistic regression (lexical.py), blend weight chosen from out-of-fold predictions.
- Documents are scored in paragraph-sized chunks (short paragraphs merged to >=40 words, long ones split) instead of sentence by sentence; sentences inherit their chunk's probability so reporting.py is unchanged.
- Training prints AUC for each model and accuracy / false-positive rate by text length (<30, 30-80, >80 words).
- Optional perplexity features (perplexity.py): `train_ai_detector(use_perplexity=True)`, needs `pip install torch transformers`. Not tested in my sandbox (no torch/network).
- Retrain required: `python -c "from ai_detect.train import train_ai_detector as t; t()"` (artifact is now detector_bundle.joblib).

# v4
- train_ai_detector(extra_csv=["file.csv", ...]): add more labelled data. CSV columns: text, label (0/human, 1/ai/chatgpt), optional group.
- When extra data is given, training prints a CROSS-SOURCE check (train on A, test on B, and back). This shows whether the model learned AI style or just the topics/vocabulary of one dataset.
- Metrics now also reported per source.
- `python -m ai_detect.check "text"` or `--file x.txt` for a quick manual test.

# v5
- prepare_hc3.py: downloads HC3 from Hugging Face and writes hc3.csv (balanced, human/ChatGPT answers to the same question share a group). Run: python -m ai_detect.prepare_hc3

# v6
- train_ai_detector(target_fpr=0.02): the "Potentially AI" threshold is now set so only ~2% of human text is flagged (was 5%). Prints thresholds and AI recall. Use target_fpr=0.01 for stricter academic use, 0.05 for the old behaviour.
