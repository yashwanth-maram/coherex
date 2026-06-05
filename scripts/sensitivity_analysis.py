"""
sensitivity_analysis.py
========================
Phase 7 — Window-Size Sensitivity Analysis

Runs compute_roc.run() on an existing results.csv for three different
window sizes by re-running evaluate_dataset.py for each window.

Usage:
    python scripts/sensitivity_analysis.py

Produces:
    data/evaluation/results_w100.csv
    data/evaluation/results_w150.csv
    data/evaluation/results_w300.csv
    data/evaluation/roc_curve_w100.png
    data/evaluation/roc_curve_w150.png
    data/evaluation/roc_curve_w300.png
    data/evaluation/sensitivity_report.txt
"""

import os
import sys
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.compute_roc import run as roc_run, print_report

EVAL_DIR    = os.path.join(ROOT, "data", "evaluation")
WINDOW_SIZES = [100, 150, 300]

PYTHON = sys.executable


def run_eval(window_size: int) -> str:
    """Run evaluate_dataset.py for a given window, return path to output CSV."""
    out_csv = os.path.join(EVAL_DIR, f"results_w{window_size}.csv")
    cmd = [
        PYTHON, os.path.join(ROOT, "scripts", "evaluate_dataset.py"),
        "--window", str(window_size),
        "--output", out_csv,
    ]
    print(f"\n{'='*60}")
    print(f"Running evaluate_dataset.py with window={window_size} …")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] evaluate_dataset.py failed for window={window_size}")
    return out_csv


def main():
    summary_rows = []

    for w in WINDOW_SIZES:
        csv_path = run_eval(w)
        tag = f"w{w}"
        result = roc_run(csv_path, threshold=0.60, tag=tag)
        print_report(result, threshold=0.60)
        if result:
            summary_rows.append((w, result["roc_auc"], result["metrics"]))

    # ── Summary comparison plot ─────────────────────────────────────────────
    if len(summary_rows) > 1:
        windows = [r[0] for r in summary_rows]
        aucs    = [r[1] for r in summary_rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar([str(w) for w in windows], aucs, color="#2196F3",
                      edgecolor="white", linewidth=1.5)
        for bar, auc_v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{auc_v:.4f}", ha="center", va="bottom", fontsize=12)
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel("Window Size (frames)", fontsize=12)
        ax.set_ylabel("AUC", fontsize=12)
        ax.set_title("Window-Size Sensitivity Analysis", fontsize=14, fontweight="bold")
        ax.axhline(y=0.90, color="green",  linestyle="--", alpha=0.6, label="AUC=0.90")
        ax.axhline(y=0.80, color="orange", linestyle="--", alpha=0.6, label="AUC=0.80")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(EVAL_DIR, "sensitivity_auc_comparison.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n  Comparison bar chart saved → {out_path}")

    # ── Text report ────────────────────────────────────────────────────────
    report_path = os.path.join(EVAL_DIR, "sensitivity_report.txt")
    with open(report_path, "w") as f:
        f.write("CoheRex — Window-Size Sensitivity Report\n")
        f.write("=" * 60 + "\n\n")
        for w, auc_v, cm in summary_rows:
            f.write(f"Window = {w} frames\n")
            f.write(f"  AUC       : {auc_v:.4f}\n")
            f.write(f"  Accuracy  : {cm['accuracy']:.4f}\n")
            f.write(f"  Precision : {cm['precision']:.4f}\n")
            f.write(f"  Recall    : {cm['recall']:.4f}\n")
            f.write(f"  F1        : {cm['f1']:.4f}\n")
            f.write(f"  TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}\n\n")

        if summary_rows:
            aucs = [r[1] for r in summary_rows]
            swing = max(aucs) - min(aucs)
            f.write(f"AUC swing across window sizes: {swing:.4f}\n")
            if swing < 0.05:
                f.write("→ Model is STABLE across window sizes.\n")
            else:
                f.write("→ Model shows sensitivity to window size. Aggregator may be unstable.\n")

    print(f"\n  Sensitivity report saved → {report_path}")

    # ── Print final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 60)
    for w, auc_v, _ in summary_rows:
        print(f"  Window {w:>4} frames : AUC = {auc_v:.4f}")
    if summary_rows:
        aucs = [r[1] for r in summary_rows]
        print(f"\n  AUC swing : {max(aucs) - min(aucs):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
