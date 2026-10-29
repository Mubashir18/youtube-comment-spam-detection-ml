# YouTube Comment Spam Detection: Robustness of a Linear Text Classifier

Course project, *Machine Learning and System Security* (HSE University, 2026).

This project builds a spam detector for YouTube comments and then stress-tests it. The question is not only "how accurate is it?" but "what happens when an attacker or the data changes?"

## Results at a glance

| Model (test set, n = 352) | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline: TF-IDF + logistic regression (threshold 0.5) | 0.952 | 0.827 | 0.885 | 0.963 |
| Improved: + class weights, SMOTE, threshold 0.446 | 0.842 | **0.952** | 0.894 | 0.963 |

The improved model trades precision for recall: missing a spam link is treated as worse than flagging a legitimate comment.

## Stress tests (improved model, same threshold)

| Test | Setting | Recall | Precision | Balanced acc. |
|---|---|---:|---:|---:|
| Clean | – | 0.952 | 0.842 | 0.895 |
| Character noise | 10% of characters replaced | 0.887 | 0.745 | 0.805 |
| Character noise | 30% of characters replaced | 0.887 | 0.560 | 0.626 |
| Vocabulary drift | 50% of spam words swapped for synonyms | 0.911 | 0.841 | 0.877 |
| Obfuscation | spaces removed | 1.000 | 0.493 | 0.530 |
| Obfuscation | leetspeak (a→4, e→3, …) | 0.958 | 0.492 | 0.528 |
| Obfuscation | spaces → underscores + leetspeak | 1.000 | 0.477 | 0.500 |

**Main finding.** The model is fairly robust to synonym drift, but simple obfuscation breaks it. High recall under obfuscation is misleading: balanced accuracy falls to about 0.50, meaning the model flags almost every comment as spam. Word-level TF-IDF cannot see through these transformations; character n-grams or normalization would be the next step.

## Method

- **Data:** [UCI YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection+dataset) (CC BY 4.0). 1,956 rows, 196 exact duplicates removed, 1,760 used (919 ham, 841 spam). Author, ID and date columns are dropped.
- **Split:** stratified 80/20 (1,408 train / 352 test), `random_state=42`.
- **No leakage:** TF-IDF is fit on the training set only; SMOTE is applied to training vectors only; the test set is never resampled.
- **Threshold:** chosen from the precision–recall curve to maximize F1 subject to recall ≥ 0.90.
- **Perturbations** are applied to every test comment (spam and ham alike), then scored with the trained model unchanged.

## Repository layout

```
run_all.py                 end-to-end pipeline: train, evaluate, stress-test, write report
download_data.py           fetches the UCI dataset
verify_project.py          checks that saved artifacts match recomputed metrics
test_reproducibility.py    reruns training with fixed seeds
src/                       preprocessing, models, evaluation
notebooks/                 EDA, baseline vs. improvement, error analysis, robustness
artifacts/                 metrics table, plots, error examples, stress-test CSVs
Report.pdf                 full written report
```

## How to run

Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python run_all.py
python verify_project.py
```

## AI usage

GitHub Copilot Chat was used for code edits and report drafting. All metrics were recomputed with fixed seeds and checked against the saved artifacts.
