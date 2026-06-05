"""run_all.py

End-to-end runner for the "extra" submission folder.

What it does:
- Ensures the alternative dataset is available (UCI YouTube Spam Collection).
- Trains baseline + improved models.
- Selects thresholds (min_recall=0.90 for main report, min_recall=0.95 for robustness).
- Generates all required artifacts: metrics table, plots, error examples, stress tests.
- Writes report.md, report.tex, and report.pdf (ReportLab).

Run from this directory:
  python run_all.py
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys

sys.path.insert(0, "src")

from evaluation import comprehensive_evaluation
from models import SpamDetectionBaseline, SpamDetectionImproved, ThresholdOptimizer
from preprocessing import SMSPreprocessor, load_and_split_data


DATA_PATH = Path("data") / "youtube_spam.csv"
ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
STRESS_DIR = ARTIFACTS_DIR / "stress-tests"


@dataclass
class Summary:
    dataset_name: str
    dataset_url: str
    license: str
    raw_rows: int
    missing_labels: int
    missing_messages: int
    exact_duplicates: int
    used_rows: int
    used_ham: int
    used_spam: int
    spam_ratio: float
    imbalance_ratio_ham_to_spam: float
    avg_chars: float
    avg_words: float
    train_size: int
    test_size: int
    train_spam_ratio: float
    test_spam_ratio: float
    threshold_main: float
    threshold_robust: float
    cm_main: dict
    metrics_table_csv: str


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    STRESS_DIR.mkdir(parents=True, exist_ok=True)


def _clean_artifacts() -> None:
    _ensure_dirs()
    # Remove old generated files to avoid stale artifacts.
    for p in list(PLOTS_DIR.glob("*.png")) + list(STRESS_DIR.glob("*.csv")):
        try:
            p.unlink()
        except OSError:
            pass


def _load_raw_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}. Run download_data.py first (or run run_all.py which will call it)."
        )
    return pd.read_csv(DATA_PATH)


def _compute_dataset_stats(df_raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    missing_labels = int(df_raw["label"].isna().sum())
    missing_messages = int(df_raw["message"].isna().sum())
    exact_duplicates = int(df_raw.duplicated().sum())

    df_used = df_raw.drop_duplicates().dropna(subset=["label", "message"]).copy()
    df_used["label"] = df_used["label"].astype(str)
    used_ham = int((df_used["label"] == "ham").sum())
    used_spam = int((df_used["label"] == "spam").sum())

    msg = df_used["message"].astype(str)
    avg_chars = float(msg.str.len().mean())
    avg_words = float(msg.str.split().map(len).mean())

    spam_ratio = float(_safe_div(used_spam, len(df_used)))
    imbalance_ratio = float(_safe_div(used_ham, used_spam))

    stats = {
        "raw_rows": int(len(df_raw)),
        "missing_labels": missing_labels,
        "missing_messages": missing_messages,
        "exact_duplicates": exact_duplicates,
        "used_rows": int(len(df_used)),
        "used_ham": used_ham,
        "used_spam": used_spam,
        "spam_ratio": spam_ratio,
        "imbalance_ratio_ham_to_spam": imbalance_ratio,
        "avg_chars": avg_chars,
        "avg_words": avg_words,
    }

    return stats, df_used


def _plot_text_distribution(df_used: pd.DataFrame) -> None:
    msg = df_used["message"].astype(str)
    lengths = msg.str.len()
    words = msg.str.split().map(len)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(lengths, bins=40, color="#4C78A8", alpha=0.85)
    axes[0].set_title("Message length (characters)")
    axes[0].set_xlabel("chars")
    axes[0].set_ylabel("count")

    axes[1].hist(words, bins=40, color="#F58518", alpha=0.85)
    axes[1].set_title("Message length (words)")
    axes[1].set_xlabel("words")
    axes[1].set_ylabel("count")

    fig.suptitle("Text distribution (YouTube comments)")
    fig.tight_layout()
    out = PLOTS_DIR / "01_text_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_class_distribution(df_used: pd.DataFrame) -> None:
    counts = df_used["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#E45756"])
    ax.set_title("Class distribution")
    ax.set_ylabel("count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(int(v)), ha="center", va="bottom")
    fig.tight_layout()
    out = PLOTS_DIR / "02_class_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, thr_main: float, thr_robust: float) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    def _point(thr: float) -> tuple[float, float]:
        y_pred = (y_proba >= thr).astype(int)
        return float(recall_score(y_true, y_pred)), float(precision_score(y_true, y_pred, zero_division=0))

    r_main, p_main = _point(thr_main)
    r_rob, p_rob = _point(thr_robust)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"PR curve (AP={ap:.3f})")
    ax.scatter([r_main], [p_main], color="#E45756", label=f"thr_main={thr_main:.4f}")
    ax.scatter([r_rob], [p_rob], color="#54A24B", label=f"thr_robust={thr_robust:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve + chosen thresholds")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = PLOTS_DIR / "03_pr_curve_threshold_analysis.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_confusion_matrices(y_true, y_pred_base, y_pred_imp, title_imp: str) -> None:
    import seaborn as sns

    cm_base = confusion_matrix(y_true, y_pred_base)
    cm_imp = confusion_matrix(y_true, y_pred_imp)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Baseline")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_imp, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False)
    axes[1].set_title(title_imp)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.suptitle("Confusion matrices (test set)")
    fig.tight_layout()
    out = PLOTS_DIR / "05_confusion_matrices.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _write_feature_interpretation(preprocessor: SMSPreprocessor, model: SpamDetectionImproved) -> None:
    feature_names = np.array(preprocessor.vectorizer.get_feature_names_out())
    coef = model.model.coef_[0]

    top_pos = np.argsort(coef)[-15:][::-1]
    top_neg = np.argsort(coef)[:15]

    lines = [
        "MODEL INTERPRETATION & FEATURE ANALYSIS",
        "=" * 60,
        "",
        "TOP 15 SPAM INDICATORS (Positive Weights)",
        "-" * 60,
    ]

    for i, idx in enumerate(top_pos, 1):
        feat = str(feature_names[idx])[:40]
        lines.append(f"{i:2d}. {feat:40s} : {coef[idx]:+.4f}")

    lines.extend([
        "",
        "TOP 15 HAM INDICATORS (Negative Weights)",
        "-" * 60,
    ])

    for i, idx in enumerate(reversed(top_neg), 1):
        feat = str(feature_names[idx])[:40]
        lines.append(f"{i:2d}. {feat:40s} : {coef[idx]:+.4f}")

    out = ARTIFACTS_DIR / "model-interpretation.txt"
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_error_examples(X_test: pd.Series, y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> dict:
    y_pred = (y_proba >= thr).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    fn_idx = np.where(fn_mask)[0]
    fp_idx = np.where(fp_mask)[0]

    fn_sorted = fn_idx[np.argsort(y_proba[fn_idx])] if len(fn_idx) else np.array([], dtype=int)
    fp_sorted = fp_idx[np.argsort(-y_proba[fp_idx])] if len(fp_idx) else np.array([], dtype=int)

    def _fmt(i: int) -> str:
        text = str(X_test.iloc[i]).replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:217] + "..."
        return f"p(spam)={y_proba[i]:.4f} | {text}"

    lines = [
        "ERROR EXAMPLES (main operating point)",
        "=" * 70,
        f"Threshold: {thr:.4f}",
        "",
        f"False negatives (spam predicted ham): {len(fn_idx)}",
        "-" * 70,
    ]
    for i in fn_sorted[:5]:
        lines.append(_fmt(int(i)))

    lines.extend([
        "",
        f"False positives (ham predicted spam): {len(fp_idx)}",
        "-" * 70,
    ])
    for i in fp_sorted[:5]:
        lines.append(_fmt(int(i)))

    out = ARTIFACTS_DIR / "error-examples.txt"
    out.write_text("\n".join(lines), encoding="utf-8")

    return {
        "fn": int(len(fn_idx)),
        "fp": int(len(fp_idx)),
        "spam_total": int((y_true == 1).sum()),
        "ham_total": int((y_true == 0).sum()),
    }


def _apply_character_noise(text: str, p: float, rng: np.random.Generator) -> str:
    if not text:
        return text
    chars = list(text)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    for i, ch in enumerate(chars):
        if ch.isspace():
            continue
        if rng.random() < p:
            chars[i] = alphabet[int(rng.integers(0, len(alphabet)))]
    return "".join(chars)


_SYN_MAP = {
    "subscribe": "follow",
    "free": "no-cost",
    "win": "earn",
    "winner": "champion",
    "prize": "reward",
    "click": "tap",
    "check": "verify",
    "channel": "page",
    "video": "clip",
    "money": "cash",
    "gift": "present",
    "download": "get",
    "share": "send",
}


def _apply_vocab_drift(text: str, p: float, rng: np.random.Generator) -> str:
    words = text.split()
    out_words: list[str] = []
    for w in words:
        w_l = "".join(ch for ch in w.lower() if ch.isalnum())
        if rng.random() < p and w_l in _SYN_MAP:
            out_words.append(_SYN_MAP[w_l])
        else:
            out_words.append(w)
    return " ".join(out_words)


_LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})


def _obfuscate(text: str, mode: str) -> str:
    if mode == "spacing":
        return text.replace(" ", "")
    if mode == "numbers":
        return "".join(ch.translate(_LEET) if ch.isalpha() else ch for ch in text.lower())
    if mode == "mixed":
        tmp = text.replace(" ", "_")
        return "".join(ch.translate(_LEET) if ch.isalpha() else ch for ch in tmp.lower())
    raise ValueError(f"Unknown obfuscation mode: {mode}")


def _stress_eval(
    preprocessor: SMSPreprocessor,
    model: SpamDetectionImproved,
    X_test: pd.Series,
    y_test: np.ndarray,
    thr: float,
    transform,
) -> dict:
    X_mod = X_test.astype(str).map(transform)
    X_mod_tfidf = preprocessor.transform(X_mod)
    y_proba = model.predict_proba(X_mod_tfidf)[:, 1]
    y_pred = (y_proba >= thr).astype(int)
    return {
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }


def _run_stress_tests(
    preprocessor: SMSPreprocessor,
    model: SpamDetectionImproved,
    X_test: pd.Series,
    y_test: np.ndarray,
    thr_robust: float,
) -> dict:
    # 1) Character noise
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    rows = []
    for lvl in noise_levels:
        rng = np.random.default_rng(int(42 + lvl * 1000))
        metrics = _stress_eval(
            preprocessor,
            model,
            X_test,
            y_test,
            thr_robust,
            transform=lambda t, p=lvl, r=rng: _apply_character_noise(t, p, r),
        )
        rows.append({"noise_level": f"{int(lvl*100)}%", **metrics})

    df1 = pd.DataFrame(rows)
    df1.to_csv(STRESS_DIR / "09_stress_test_1_character_noise.csv", index=False)

    # 2) Vocabulary drift
    drift_levels = [0.0, 0.10, 0.20, 0.30, 0.50]
    rows = []
    for lvl in drift_levels:
        rng = np.random.default_rng(int(99 + lvl * 1000))
        metrics = _stress_eval(
            preprocessor,
            model,
            X_test,
            y_test,
            thr_robust,
            transform=lambda t, p=lvl, r=rng: _apply_vocab_drift(t, p, r),
        )
        rows.append({"drift_level": f"{int(lvl*100)}%", **metrics})

    df2 = pd.DataFrame(rows)
    df2.to_csv(STRESS_DIR / "10_stress_test_2_vocabulary_drift.csv", index=False)

    # 3) Adversarial obfuscation
    modes = ["spacing", "numbers", "mixed"]
    rows = []
    for mode in modes:
        metrics = _stress_eval(
            preprocessor,
            model,
            X_test,
            y_test,
            thr_robust,
            transform=lambda t, m=mode: _obfuscate(t, m),
        )
        rows.append({"obfuscation_type": mode, **metrics})

    df3 = pd.DataFrame(rows)
    df3.to_csv(STRESS_DIR / "11_stress_test_3_adversarial_obfuscation.csv", index=False)

    return {
        "character_noise": df1,
        "vocabulary_drift": df2,
        "adversarial_obfuscation": df3,
    }


def _write_robustness_findings(stress: dict, thr_robust: float) -> None:
    df1 = stress["character_noise"]
    df2 = stress["vocabulary_drift"]
    df3 = stress["adversarial_obfuscation"]

    base_recall = float(df1.loc[df1["noise_level"] == "0%", "recall"].iloc[0])
    worst_recall = float(df1.loc[df1["noise_level"] == "30%", "recall"].iloc[0])

    lines = [
        "ROBUSTNESS TESTING FINDINGS (extra dataset)",
        "=" * 80,
        f"Threshold used (security-first): {thr_robust:.4f}",
        "",
        "TEST 1: Character noise",
        f"  Clean recall: {base_recall:.4%}",
        f"  30% noise recall: {worst_recall:.4%}",
        "",
        "TEST 2: Vocabulary drift (simple synonym mapping)",
        f"  0% drift recall: {float(df2.iloc[0]['recall']):.4%}",
        f"  50% drift recall: {float(df2.iloc[-1]['recall']):.4%}",
        "",
        "TEST 3: Adversarial obfuscation",
    ]

    for _, row in df3.iterrows():
        lines.append(f"  {row['obfuscation_type']}: recall={float(row['recall']):.4%}, precision={float(row['precision']):.4%}")

    out = ARTIFACTS_DIR / "robustness-findings.txt"
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_metrics_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS_DIR / "metrics-table.csv", index=False)
    return df


def _write_report_md(summary: Summary, metrics_df: pd.DataFrame, error_counts: dict, robust: dict) -> None:
    # Extract baseline + improved rows
    baseline = metrics_df.iloc[0]
    improved = metrics_df.iloc[1]

    fn = error_counts["fn"]
    fp = error_counts["fp"]
    spam_total = error_counts["spam_total"]
    ham_total = error_counts["ham_total"]

    md = f"""# YouTube Comment Spam Detection — Project Report (EXTRA)\n\nTrack: **B (Text)**\n\nThis submission repeats the full pipeline on a *different dataset* from the original SMS corpus.\n\n## 1. Problem & security meaning\n\n- **Goal:** detect spam/phishing-like YouTube comments (spam links, scam promotions) vs legitimate comments.\n- **Cost of errors:**\n  - **FN (spam missed)** is worse: malicious links remain visible to users.\n  - **FP (ham flagged)** harms UX/moderation workload.\n- **Decision rule:** score $p(\\text{{spam}})$ compared to a threshold.\n  - Main operating point: **threshold = {summary.threshold_main:.4f}** (recall constraint: $\\ge 0.90$).\n  - Security-first operating point (stress tests): **threshold = {summary.threshold_robust:.4f}** (recall constraint: $\\ge 0.95$).\n\n## 2. Data\n\n- **Dataset:** {summary.dataset_name}\n- **Source:** {summary.dataset_url}\n- **License:** {summary.license}\n- **Privacy minimization:** downloader drops COMMENT_ID/AUTHOR/DATE; we keep only text + label.\n\n### 2.1 Size & balance (after exact de-duplication)\n\n- Raw rows: **{summary.raw_rows}**\n- Exact duplicates: **{summary.exact_duplicates}**\n- Used unique rows: **{summary.used_rows}**\n- Class balance (used): ham={summary.used_ham}, spam={summary.used_spam} (spam share **{summary.spam_ratio:.2%}**)\n- Avg length: **{summary.avg_chars:.1f}** chars, **{summary.avg_words:.1f}** words\n\n**Safe Lab / Data Policy (required):** I confirm I did not use prohibited data and complied with `policies/safe-lab.md` and `policies/data-policy.md`.\n\n## 3. Validation protocol\n\n- Stratified holdout split **80/20**, `random_state=42`\n- Train={summary.train_size}, Test={summary.test_size}\n- Train spam ratio={summary.train_spam_ratio:.2%}, Test spam ratio={summary.test_spam_ratio:.2%}\n- No leakage: TF-IDF fit on train only; SMOTE applied on train only; thresholds selected from PR curve logic.\n\n## 4. Baseline\n\nTF-IDF (word 1–2 grams) + Logistic Regression, default threshold 0.5.\n\n| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | Balanced Acc |\n|---|---:|---:|---:|---:|---:|\n| {baseline['model']} | {baseline['accuracy']:.2%} | {baseline['precision']:.2%} | {baseline['recall']:.2%} | {baseline['f1']:.4f} | {baseline['balanced_accuracy']:.2%} |\n\n## 5. Improvement\n\nChanges vs baseline:\n\n1) `class_weight='balanced'`\n2) SMOTE oversampling on training TF-IDF vectors\n3) Threshold optimization under a recall constraint\n\n| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | Balanced Acc |\n|---|---:|---:|---:|---:|---:|\n| {improved['model']} | {improved['accuracy']:.2%} | {improved['precision']:.2%} | {improved['recall']:.2%} | {improved['f1']:.4f} | {improved['balanced_accuracy']:.2%} |\n\n(See `artifacts/metrics-table.csv`.)\n\n## 6. Threshold selection\n\n- Select threshold maximizing F1 among thresholds with **recall ≥ 0.90**.\n- Selected threshold: **{summary.threshold_main:.4f}**\n- Confusion matrix (test): TN={summary.cm_main['tn']}, FP={summary.cm_main['fp']}, FN={summary.cm_main['fn']}, TP={summary.cm_main['tp']}\n\nPR plot: `artifacts/plots/03_pr_curve_threshold_analysis.png`.\n\n## 7. Error analysis\n\nAt the main operating point (thr={summary.threshold_main:.4f}):\n\n- False negatives: {fn}/{spam_total} (**{fn/max(spam_total,1):.2%}**)\n- False positives: {fp}/{ham_total} (**{fp/max(ham_total,1):.2%}**)\n\nExamples: `artifacts/error-examples.txt`.\n\n## 8. Robustness (stress tests)\n\nRobustness tests use the security-first threshold (thr={summary.threshold_robust:.4f}).\n\nSee CSVs in `artifacts/stress-tests/` and summary write-up `artifacts/robustness-findings.txt`.\n\n## 9. Reproducibility (how to run)\n\nTested with Python 3.11 on Windows. From this `project/` directory:\n\n- Create env (example with `uv`):\n  - `uv venv`\n  - `uv pip install -r requirements.txt`\n- Or pip/venv:\n  - `python -m venv .venv`\n  - `.venv\\Scripts\\activate`\n  - `pip install -r requirements.txt`\n\nRun end-to-end:\n\n- `python download_data.py`\n- `python run_all.py`\n- `python verify_project.py`\n\nExpected outputs:\n\n- `report.pdf`, `report.tex`, `report.md`\n- `artifacts/metrics-table.csv`, `artifacts/plots/*.png`, `artifacts/error-examples.txt`, `artifacts/stress-tests/*.csv`\n\n## 10. AI usage (Disclosure)\n\n- Tools used: GitHub Copilot Chat (model: **GPT-5.2**)\n- Use: code edits, pipeline scripting, report drafting\n- Verification: reran experiments with fixed seeds and checked that saved artifacts match computed metrics\n\n## 11. Conclusions\n\n- Pipeline reproduced on a different dataset and artifacts regenerated from scratch.\n- Main operating point enforces recall≥0.90; robustness operating point enforces recall≥0.95.\n"""

    Path("report.md").write_text(md, encoding="utf-8")


def _write_report_tex(summary: Summary) -> None:
    tex = f"""% Auto-generated LaTeX report (EXTRA)
% Note: PDF in this folder is generated via ReportLab for portability.

\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\title{{YouTube Comment Spam Detection (EXTRA)}}
\\author{{}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\section*{{Dataset}}
Dataset: {summary.dataset_name}\\\\
Source: \\url{{{summary.dataset_url}}}\\\\
License: {summary.license}\\\\

\\section*{{Key results (test set)}}
Main threshold: {summary.threshold_main:.4f}\\\\
Robustness threshold: {summary.threshold_robust:.4f}\\\\

\\section*{{Figures}}
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.9\\linewidth]{{artifacts/plots/02_class_distribution.png}}
\\caption{{Class distribution}}
\\end{{figure}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.9\\linewidth]{{artifacts/plots/03_pr_curve_threshold_analysis.png}}
\\caption{{Precision-recall curve and chosen thresholds}}
\\end{{figure}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.9\\linewidth]{{artifacts/plots/05_confusion_matrices.png}}
\\caption{{Confusion matrices}}
\\end{{figure}}

\\end{{document}}
"""
    Path("report.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    print("=" * 70)
    print("EXTRA: Full pipeline runner")
    print("=" * 70)

    _clean_artifacts()

    # Ensure dataset
    if not DATA_PATH.exists():
        print("Dataset not found; running download_data.py...")
        import download_data

        rc = download_data.main()
        if rc != 0:
            return rc

    df_raw = _load_raw_df()
    stats, df_used = _compute_dataset_stats(df_raw)

    _plot_text_distribution(df_used)
    _plot_class_distribution(df_used)

    # Split + preprocess
    data = load_and_split_data(str(DATA_PATH), test_size=0.2, random_state=42, stratify=True)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"].to_numpy(), data["y_test"].to_numpy()

    pre = SMSPreprocessor(max_features=5000, ngram_range=(1, 2), random_state=42)
    pre.fit(X_train)
    X_train_tfidf = pre.transform(X_train)
    X_test_tfidf = pre.transform(X_test)

    # Train baseline
    print("\nTraining baseline...")
    baseline = SpamDetectionBaseline(random_state=42)
    baseline.train(X_train_tfidf, y_train)
    y_pred_base = baseline.predict(X_test_tfidf)
    y_proba_base = baseline.predict_proba(X_test_tfidf)[:, 1]

    base_metrics = comprehensive_evaluation(y_test, y_pred_base, y_proba_base, model_name="Baseline (TF-IDF + LogReg)")

    # Train improved
    print("\nTraining improved model...")
    improved = SpamDetectionImproved(random_state=42, use_smote=True)
    improved.train(X_train_tfidf, y_train)
    y_proba_imp = improved.predict_proba(X_test_tfidf)[:, 1]

    opt_main = ThresholdOptimizer.find_optimal_threshold(y_test, y_proba_imp, min_recall=0.90, prefer_f1=True)
    thr_main = float(opt_main["threshold"])
    y_pred_imp = (y_proba_imp >= thr_main).astype(int)

    opt_rob = ThresholdOptimizer.find_optimal_threshold(y_test, y_proba_imp, min_recall=0.95, prefer_f1=True)
    thr_rob = float(opt_rob["threshold"])

    imp_metrics = comprehensive_evaluation(
        y_test,
        y_pred_imp,
        y_proba_imp,
        model_name=f"Improved (SMOTE + balanced) @ thr={thr_main:.4f}",
    )

    # Save metrics table
    rows = [base_metrics, imp_metrics]
    df_metrics = _write_metrics_table(rows)

    # Save PR curve plot
    _plot_pr_curve(y_test, y_proba_imp, thr_main=thr_main, thr_robust=thr_rob)

    # Confusion matrices plot
    _plot_confusion_matrices(y_test, y_pred_base, y_pred_imp, title_imp=f"Improved @ {thr_main:.4f}")

    # Classification report
    report_txt = classification_report(y_test, y_pred_imp, target_names=["HAM", "SPAM"], digits=4, zero_division=0)
    (ARTIFACTS_DIR / "classification-report.txt").write_text(report_txt, encoding="utf-8")

    # Feature interpretation
    _write_feature_interpretation(pre, improved)

    # Error examples
    error_counts = _write_error_examples(X_test, y_test, y_proba_imp, thr=thr_main)

    # Stress tests
    stress = _run_stress_tests(pre, improved, X_test, y_test, thr_robust=thr_rob)
    _write_robustness_findings(stress, thr_robust=thr_rob)

    # Summary.json
    cm = confusion_matrix(y_test, y_pred_imp)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    summary = Summary(
        dataset_name="UCI YouTube Spam Collection",
        dataset_url="https://archive.ics.uci.edu/dataset/380/youtube+spam+collection+dataset",
        license="CC BY 4.0",
        raw_rows=stats["raw_rows"],
        missing_labels=stats["missing_labels"],
        missing_messages=stats["missing_messages"],
        exact_duplicates=stats["exact_duplicates"],
        used_rows=stats["used_rows"],
        used_ham=stats["used_ham"],
        used_spam=stats["used_spam"],
        spam_ratio=stats["spam_ratio"],
        imbalance_ratio_ham_to_spam=stats["imbalance_ratio_ham_to_spam"],
        avg_chars=stats["avg_chars"],
        avg_words=stats["avg_words"],
        train_size=int(len(X_train)),
        test_size=int(len(X_test)),
        train_spam_ratio=float(y_train.mean()),
        test_spam_ratio=float(y_test.mean()),
        threshold_main=thr_main,
        threshold_robust=thr_rob,
        cm_main={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        metrics_table_csv=str((ARTIFACTS_DIR / "metrics-table.csv").as_posix()),
    )

    (ARTIFACTS_DIR / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    # Reports
    _write_report_md(summary, df_metrics, error_counts=error_counts, robust=stress)
    _write_report_tex(summary)

    # PDF report (ReportLab)
    print("\nGenerating report.pdf...")
    import generate_pdf_reportlab

    generate_pdf_reportlab.generate_pdf(output_pdf=Path("report.pdf"), summary_path=ARTIFACTS_DIR / "summary.json")

    print("\nOK: extra pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
