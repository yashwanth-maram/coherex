"""
forensic_self_audit.py
=======================
Systematic verification of baseline evaluation assumptions.

Checks:
  1. ROC Inversion: Confirms that tamper_score (1 - integrity) is used.
  2. Class Means: Computes mean(authentic) vs mean(tampered).
  3. Visual Overlap Check: Shows min/max/mean distributions.

Usage:
    python scripts/forensic_self_audit.py [--csv path]
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "evaluation", "results.csv")

def run_audit(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 0 = authentic, 1 = tampered
    auth_df = df[df['label'] == 0]
    tamp_df = df[df['label'] == 1]

    print("\n" + "="*60)
    print("  FORENSIC SELF-AUDIT: CLASS MEANS")
    print("="*60)
    
    m_auth = auth_df['integrity_score'].mean()
    m_tamp = tamp_df['integrity_score'].mean()
    
    print(f"  Authentic Mean Integrity : {m_auth:.6f}")
    print(f"  Tampered Mean Integrity  : {m_tamp:.6f}")
    print("-" * 60)
    
    if m_tamp > m_auth:
        print("  [CRITICAL] INVERSION DETECTED")
        print("  Tampered videos have HIGHER integrity than authentic.")
        print("  This means your agents are NOT reacting or are reacting inversely.")
    else:
        diff = m_auth - m_tamp
        print(f"  Separation (Auth - Tamp) : {diff:.6f}")
        if diff < 0.05:
            print("  [WARNING] WEAK SEPARATION")
            print("  The difference between classes is negligible.")
            
    print("\n" + "="*60)
    print("  DISTRIBUTION OVERVIEW")
    print("="*60)
    print(df.groupby('label')['integrity_score'].describe())
    
    print("\n" + "="*60)
    print("  ROC INVERSION TEST (TOP 5 SAMPLES)")
    print("="*60)
    df['tamper_score'] = 1.0 - df['integrity_score']
    print(df[['video_name', 'label', 'integrity_score', 'tamper_score']].head())
    
    print("\n" + "="*60)
    print("  VERDICT")
    print("="*60)
    if m_tamp >= m_auth:
        print("  FAIL: The model is technically blind to your tampering.")
        print("  Proceed to Phase 3 & 4 (Isolate Agents & Remove Aggregation).")
    else:
        print("  PASS: The model has the correct 'scent', but it is too weak/too noisy.")
        print("  Proceed to Phase 6 (Sensitivity Tuning).")

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    run_audit(csv_arg)
