"""
ablation_study.py
=================
Phase 8 — Agent-Level Ablation Study

Runs evaluate_dataset.py with progressively richer agent configurations,
then computes ROC for each and plots a comparison.

Configurations tested:
    1. motion             — MotionAgent only
    2. motion_continuity  — Motion + Continuity
    3. motion_crowd       — Motion + Crowd
    4. full               — All three agents (full fusion)

Usage:
    python scripts/ablation_study.py

Produces:
    data/evaluation/results_ablation_<config>.csv  (4 files)
    data/evaluation/roc_curve_ablation_<config>.png (4 files)
    data/evaluation/ablation_roc_comparison.png
    data/evaluation/ablation_report.txt
"""

import os
import sys
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.compute_roc import run as roc_run, print_report, load_csv
from sklearn.metrics import roc_curve, auc

EVAL_DIR = os.path.join(ROOT, "data", "evaluation")
PYTHON   = sys.executable

CONFIGS = [
    ("motion",             "Motion only"),
    ("motion_continuity",  "Motion + Continuity"),
    ("motion_crowd",       "Motion + Crowd"),
    ("full",               "Full fusion"),
]

COLOURS = ["#E53935", "#FB8C00", "#43A047", "#1E88E5"]


def run_eval(agents_mode: str) -> str:
    out_csv = os.path.join(EVAL_DIR, f"results_ablation_{agents_mode}.csv")
    cmd = [
        PYTHON, os.path.join(ROOT, "scripts", "evaluate_dataset.py"),
        "--agents", agents_mode,
        "--output", out_csv,
    ]
    print(f"\n{'='*60}")
    print(f"Ablation: agents={agents_mode}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] evaluate_dataset.py failed for agents={agents_mode}")
    return out_csv


def main():
    summary_rows = []

    # Collect individual ROC data for comparison plot
    roc_data = []

    for mode, label in CONFIGS:
        csv_path = run_eval(mode)
        tag = f"ablation_{mode}"
        result = roc_run(csv_path, threshold=0.60, tag=tag)
        print_report(result, threshold=0.60)

        if result:
            summary_rows.append((label, result["roc_auc"], result["metrics"]))

            # Re-load for overlay plot
            _, scores, labels = load_csv(csv_path)
            if len(scores) >= 4:
                tamper_scores = 1.0 - np.array(scores)
                fpr, tpr, _ = roc_curve(labels, tamper_scores)
                roc_data.append((fpr, tpr, result["roc_auc"], label))

    # ── ROC overlay comparison ─────────────────────────────────────────────
    if roc_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", lw=1,
                label="Random")
        for (fpr, tpr, roc_auc, lbl), col in zip(roc_data, COLOURS):
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"{lbl}  (AUC={roc_auc:.4f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
        ax.set_title("CoheRex Ablation Study — ROC Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(EVAL_DIR, "ablation_roc_comparison.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n  Ablation ROC comparison saved → {out_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────
    if summary_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        xlabels = [r[0] for r in summary_rows]
        aucs    = [r[1] for r in summary_rows]
        bars = ax.bar(xlabels, aucs, color=COLOURS[:len(summary_rows)],
                      edgecolor="white", linewidth=1.5)
        for bar, auc_v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{auc_v:.4f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("AUC", fontsize=12)
        ax.set_title("Ablation Study — AUC by Agent Config", fontsize=13, fontweight="bold")
        ax.axhline(y=0.90, color="green",  linestyle="--", alpha=0.5, label="0.90")
        ax.axhline(y=0.80, color="orange", linestyle="--", alpha=0.5, label="0.80")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=10, ha="right")
        fig.tight_layout()
        out_path = os.path.join(EVAL_DIR, "ablation_auc_bar.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Ablation AUC bar chart saved → {out_path}")

    # ── Text report ───────────────────────────────────────────────────────
    report_path = os.path.join(EVAL_DIR, "ablation_report.txt")
    with open(report_path, "w") as f:
        f.write("CoheRex — Ablation Study Report\n")
        f.write("=" * 60 + "\n\n")
        for label, auc_v, cm in summary_rows:
            f.write(f"Config: {label}\n")
            f.write(f"  AUC       : {auc_v:.4f}\n")
            f.write(f"  Accuracy  : {cm['accuracy']:.4f}\n")
            f.write(f"  Precision : {cm['precision']:.4f}\n")
            f.write(f"  Recall    : {cm['recall']:.4f}\n")
            f.write(f"  F1        : {cm['f1']:.4f}\n\n")

        # Agent contribution analysis
        f.write("\nAgent Contribution Analysis:\n")
        aucs_by_config = {label: auc_v for label, auc_v, _ in summary_rows}
        base = aucs_by_config.get("Motion only", 0.0)
        full = aucs_by_config.get("Full fusion", 0.0)

        for label_check, label_name in [
            ("Motion + Continuity", "ContinuityAgent"),
            ("Motion + Crowd",      "CrowdAgent"),
        ]:
            v = aucs_by_config.get(label_check, None)
            if v is not None and base > 0:
                delta = v - base
                f.write(f"  {label_name}: AUC delta vs Motion-only = {delta:+.4f}\n")
                if delta > 0.01:
                    f.write(f"    → {label_name} HELPS (adds {delta:.4f} AUC)\n")
                elif delta < -0.01:
                    f.write(f"    → {label_name} HURTS (costs {abs(delta):.4f} AUC)\n")
                else:
                    f.write(f"    → {label_name} is NEUTRAL\n")

    print(f"\n  Ablation report saved → {report_path}")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 60)
    for label, auc_v, _ in summary_rows:
        print(f"  {label:<25} : AUC = {auc_v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
