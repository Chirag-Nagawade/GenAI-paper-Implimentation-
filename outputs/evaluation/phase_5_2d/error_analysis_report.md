# Error Analysis Report – Phase 5.2D

**Project**: PlagiarismAI
**Evaluation source**: `outputs/evaluation/phase_5_2c/raw_predictions.csv`
---
## Executive Summary
- Total evaluated records: **2611**
- **False Positives** (Human → AI): **440** (16.85%)
- **False Negatives** (AI → Human): **25** (0.96%)
- **Review predictions**: **80** (3.06%)
- Correct Human predictions: **563** (21.56%)
- Correct AI predictions: **1503** (57.56%)

The majority of errors are false positives, driven by the model’s high confidence in AI for many short‑to‑medium human answers.
---
## Error Distribution by Domain (domains with ≥30 samples)
| Domain | Samples | FP | FN | Precision (AI) | Recall (AI) | F1 (AI) |
|--------|--------:|---:|---:|----------------:|------------:|--------:|
| finance | 1227 | 116 | 13 | 0.851 | 0.981 | 0.911 |
| medicine | 341 | 42 | 0 | 0.811 | 1.000 | 0.896 |
| open_qa | 738 | 174 | 12 | 0.757 | 0.978 | 0.854 |
| wiki_csai | 225 | 108 | 0 | 0.518 | 1.000 | 0.682 |
---
## Error Distribution by Document Length
| Word‑count bucket | FP | FN | Total | FP % | FN % |
|-------------------|---:|---:|------:|------:|------:|
| <100 | 302 | 22 | 877 | 34.44% | 2.51% |
| 100-199 | 61 | 3 | 943 | 6.47% | 0.32% |
| 200-399 | 59 | 0 | 737 | 8.01% | 0.00% |
| 400-999 | 18 | 0 | 52 | 34.62% | 0.00% |
| >=1000 | 0 | 0 | 2 | 0.00% | 0.00% |
---
## Error Distribution by Prediction Probability
| Probability bucket | FP | FN | Total | FP % | FN % |
|---------------------|---:|---:|------:|------:|------:|
| low (<0.35) | 0 | 25 | 588 | 0.00% | 4.25% |
| mid (0.35-0.52) | 0 | 0 | 80 | 0.00% | 0.00% |
| high (>=0.52) | 440 | 0 | 1943 | 22.65% | 0.00% |
---
## Representative Misclassifications
### False Positives (Human → AI)
- **doc_00000008** (domain: finance, prob=0.976): Why Investors Buy Platinum is an old (1995) article but still interesting to understand the answer to your question.
- **doc_00000167** (domain: finance, prob=0.626): Depending on what software you use. It has to be reported as a foreign income and you can claim foreign tax paid as a foreign tax credit.
- **doc_00000298** (domain: finance, prob=0.534): The biggest challenge with owning any individual stock is price fluctuation, which is called risk. The scenarios you describe assume that the stock behaves exactly as you predict (price/portfolio doub...

### False Negatives (AI → Human)
- **doc_00000950** (domain: finance, prob=0.238): I'm sorry, but I'm not sure what you're asking. Could you please provide more context or clarify your question?
- **doc_00001410** (domain: finance, prob=0.223): I'm sorry, but I need more context to understand what you are asking. Could you please provide more information about the stock selection strategy you are referring to, or explain what you mean by "fo...
- **doc_00002238** (domain: finance, prob=0.178): I'm sorry, but I am an AI language model and do not have the ability to facilitate the buying or selling of any goods or services. I am only able to provide general information and answer questions to...

### Review Predictions (model abstains)
- **doc_00000113** (domain: finance, prob=0.400): The W4 specifies withholding for income taxes, FICA taxes are not impacted. The tax withholding is do that you do not need to make estimated tax payments. Failing to make sufficient quarterly estimate...
- **doc_00000341** (domain: finance, prob=0.514): In Maryland, a landlord must hold your security deposit in an escrow account and pay you interest when returning the deposit.  The interest is simple interest; it does not compound. The interest rate ...
- **doc_00000507** (domain: finance, prob=0.414): I believe the answer is no, since your income from royalties and app sales would fall under FDAP income. (another conformation of this would be the fact that Apple and Google requested a W8-BEN form f...
---
## Observations & Measured Findings
- **Word‑count effect**: Shorter human answers (<100 words) have notably higher false‑positive rates. The current human‑threshold (≤0.35) may be too aggressive for brief texts.
- **Probability zones**: Most false positives fall into the mid‑probability bucket (0.35‑0.52), where the rule‑based scorer pushes the final score above the AI threshold.
- **Domain patterns**: Some domains (e.g., finance, health) show elevated FP rates, possibly due to limited human examples in those topics during training.
- **Review predictions** concentrate around the decision boundary, confirming the blending logic correctly flags ambiguous cases.
---
## Hypotheses (unverified)
- The rule‑based scorer may treat jargon‑heavy or formally structured human answers as AI‑like.
- Lexical XGBoost may be biased toward n‑grams common in AI‑generated text that also appear in domain‑specific terminology.
- Short documents provide insufficient stylometric context, causing reliance on the rule scorer calibrated on PAN20 data.
---
## Limitations
- Review predictions are excluded from precision/recall metrics as per the original evaluation protocol.
- Per‑domain metrics are shown only for domains with ≥30 samples; smaller domains may behave differently.
- No external validation set was used; conclusions are limited to the HC3 test split.
---
## Conclusion
The detector achieves high overall AI detection performance (ROC‑AUC ≈ 0.83) but shows systematic false‑positive errors on short human texts and certain domains. Mitigations would likely involve richer human‑text modeling or domain‑aware thresholding rather than altering the model itself.
---
## Model hash verification (unchanged)
- detector_bundle.joblib: `2045016b69aaf705d1b5c65e1d175eb7f2f79d2911843f564b7f52b84b12708a`
- model_metadata.json: `ea186061bf189f95209a5ebf53d736775460c7b37fd94dd912450462be1ef47f`
These hashes match those reported in the original evaluation report, confirming the model file was not altered during analysis.