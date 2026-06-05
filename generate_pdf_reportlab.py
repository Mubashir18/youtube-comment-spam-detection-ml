#!/usr/bin/env python3
"""generate_pdf_reportlab.py

Generates `report.pdf` for the "extra" submission using ReportLab.

Inputs (produced by run_all.py):
- artifacts/summary.json
- artifacts/metrics-table.csv
- artifacts/error-examples.txt
- artifacts/stress-tests/*.csv
- artifacts/plots/*.png

This is intentionally data-driven (no hard-coded metrics) to avoid mismatches.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_pdf(output_pdf: Path | str = "report.pdf", summary_path: Path | str = "artifacts/summary.json") -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
            Image,
        )

        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "Report generation requires reportlab and pandas. "
            "Install with: pip install reportlab pandas"
        ) from e

    output_pdf = Path(output_pdf)
    summary_path = Path(summary_path)

    artifacts_dir = Path("artifacts")
    metrics_path = artifacts_dir / "metrics-table.csv"
    errors_path = artifacts_dir / "error-examples.txt"
    stress_dir = artifacts_dir / "stress-tests"
    plots_dir = artifacts_dir / "plots"

    summary = _load_json(summary_path)
    metrics_df = pd.read_csv(metrics_path)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=18,
        alignment=1,
    )
    heading1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8,
    )
    heading2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
        title="Extra Submission Report",
        author=str(summary.get("author", "")),
    )

    elements: list = []

    # Title
    elements.append(Paragraph("YouTube Comment Spam Detection — Extra Submission", title_style))
    elements.append(Paragraph(f"Dataset: {summary['dataset_name']} (license: {summary['license']})", body))
    elements.append(Paragraph(f"Source: {summary['dataset_url']}", body))
    elements.append(Spacer(1, 0.2 * inch))

    # Summary block
    # Expect row 0=baseline, row 1=improved
    perf_row = metrics_df.iloc[1] if len(metrics_df) > 1 else metrics_df.iloc[0]
    thr_main = float(summary["threshold_main"])
    elements.append(Paragraph("1. Executive Summary", heading1))
    elements.append(
        Paragraph(
            (
                f"We train a TF-IDF + Logistic Regression classifier to detect spam comments. "
                f"At the main operating point (threshold={thr_main:.4f}, min_recall=0.90), "
                f"test precision={float(perf_row['precision']):.2%}, recall={float(perf_row['recall']):.2%}, F1={float(perf_row['f1']):.4f}."
            ),
            body,
        )
    )

    # Dataset stats
    elements.append(Paragraph("2. Data", heading1))
    data_table = [
        ["Metric", "Value"],
        ["Raw rows", f"{summary['raw_rows']}"],
        ["Exact duplicates", f"{summary['exact_duplicates']}"],
        ["Rows used (after dedup/dropna)", f"{summary['used_rows']}"],
        ["Ham / Spam (used)", f"{summary['used_ham']} / {summary['used_spam']}"],
        ["Spam share", f"{float(summary['spam_ratio']):.2%}"],
        ["Train size", f"{summary['train_size']}"],
        ["Test size", f"{summary['test_size']}"],
        ["Avg length (chars)", f"{float(summary['avg_chars']):.1f}"],
        ["Avg words", f"{float(summary['avg_words']):.1f}"],
    ]
    t = Table(data_table, colWidths=[220, 250])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    elements.append(t)
    elements.append(Spacer(1, 0.2 * inch))

    # Metrics table
    elements.append(Paragraph("3. Metrics", heading1))
    metrics_rows = [metrics_df.columns.tolist()] + metrics_df.values.tolist()
    mt = Table(metrics_rows, repeatRows=1)
    mt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(mt)
    elements.append(Spacer(1, 0.2 * inch))

    # Plots page
    elements.append(PageBreak())
    elements.append(Paragraph("4. Plots", heading1))

    for name in [
        "01_text_distribution.png",
        "02_class_distribution.png",
        "03_pr_curve_threshold_analysis.png",
        "05_confusion_matrices.png",
    ]:
        p = plots_dir / name
        if p.exists():
            elements.append(Paragraph(name.replace("_", " ").replace(".png", ""), heading2))
            elements.append(Image(str(p), width=6.5 * inch, height=4.0 * inch))
            elements.append(Spacer(1, 0.15 * inch))

    # Error examples
    elements.append(PageBreak())
    elements.append(Paragraph("5. Error Examples", heading1))
    if errors_path.exists():
        from xml.sax.saxutils import escape
        lines = errors_path.read_text(encoding="utf-8").splitlines()
        # Keep it short in PDF (full file is in artifacts)
        snippet = "<br/>".join(escape(line) for line in lines[:40])
        elements.append(Paragraph(snippet, body))
    else:
        elements.append(Paragraph("artifacts/error-examples.txt not found.", body))

    # Robustness
    elements.append(PageBreak())
    elements.append(Paragraph("6. Robustness Stress Tests", heading1))

    stress_files = [
        ("Character noise", stress_dir / "09_stress_test_1_character_noise.csv"),
        ("Vocabulary drift", stress_dir / "10_stress_test_2_vocabulary_drift.csv"),
        ("Adversarial obfuscation", stress_dir / "11_stress_test_3_adversarial_obfuscation.csv"),
    ]

    for title, path in stress_files:
        elements.append(Paragraph(title, heading2))
        if path.exists():
            df = pd.read_csv(path)
            rows = [df.columns.tolist()] + df.values.tolist()
            st = Table(rows, repeatRows=1)
            st.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            elements.append(st)
        else:
            elements.append(Paragraph(f"Missing: {path}", body))
        elements.append(Spacer(1, 0.15 * inch))

    # Compliance + AI usage
    elements.append(PageBreak())
    elements.append(Paragraph("7. Policies & AI Usage (Required)", heading1))
    elements.append(
        Paragraph(
            "Safe Lab / Data Policy: I confirm I did not use prohibited data and complied with policies/safe-lab.md and policies/data-policy.md.",
            body,
        )
    )
    elements.append(
        Paragraph(
            "AI Usage: GitHub Copilot Chat (GPT-5.2) was used for code editing, debugging, and report drafting. All metrics and artifacts were verified by re-running the pipeline with fixed seeds.",
            body,
        )
    )

    doc.build(elements)
    return output_pdf


def main() -> int:
    out = generate_pdf("report.pdf", "artifacts/summary.json")
    print(f"OK: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
