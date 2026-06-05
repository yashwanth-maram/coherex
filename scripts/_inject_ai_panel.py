"""Inject AI meta-classifier panel into frontend/app.py."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP  = os.path.join(ROOT, "frontend", "app.py")

AI_PANEL_LINES = [
    "\n",
    "    # -- AI Meta-Classifier Panel ----------------------------------------------\n",
    "    st.markdown('<p class=\"sec-hdr\">&#129302; AI Decision Confidence (Random Forest Meta-Classifier)</p>',\n",
    "                unsafe_allow_html=True)\n",
    "\n",
    "    ai = report.get(\"ai_classifier\", {})\n",
    "    hf = report.get(\"hybrid_features\", {})\n",
    "\n",
    "    if ai and ai.get(\"available\"):\n",
    "        conf       = ai[\"confidence\"]\n",
    "        ai_verdict = ai[\"verdict\"]\n",
    "\n",
    "        verdict_color = {\n",
    "            \"AUTHENTIC\":  CLR_HIGH,\n",
    "            \"SUSPICIOUS\": CLR_MOD,\n",
    "            \"TAMPERED\":   CLR_COMP,\n",
    "        }.get(ai_verdict, \"#e6edf3\")\n",
    "\n",
    "        ai1, ai2, ai3 = st.columns(3)\n",
    "\n",
    "        with ai1:\n",
    "            st.markdown(\n",
    "                f'<div class=\"metric-card\">'\n",
    "                f'<div class=\"metric-lbl\">AI Verdict</div>'\n",
    "                f'<div class=\"metric-val\" style=\"font-size:1.1rem;color:{verdict_color}\">'\n",
    "                f'{ai_verdict}</div></div>',\n",
    "                unsafe_allow_html=True)\n",
    "\n",
    "        with ai2:\n",
    "            bar_w   = int(conf * 100)\n",
    "            bar_clr = CLR_COMP if conf >= 0.65 else CLR_MOD if conf >= 0.40 else CLR_HIGH\n",
    "            st.markdown(\n",
    "                f'<div class=\"metric-card\">'\n",
    "                f'<div class=\"metric-lbl\">Tamper Probability</div>'\n",
    "                f'<div class=\"metric-val\" style=\"color:{bar_clr}\">{conf:.1%}</div>'\n",
    "                f'<div style=\"background:#1e293b;border-radius:4px;height:6px;margin-top:8px;\">'\n",
    "                f'<div style=\"background:{bar_clr};width:{bar_w}%;height:6px;border-radius:4px;\"></div>'\n",
    "                f'</div></div>',\n",
    "                unsafe_allow_html=True)\n",
    "\n",
    "        with ai3:\n",
    "            phys_s = report[\"mean_score\"]\n",
    "            agree  = (conf >= 0.5 and phys_s < 0.7) or (conf < 0.5 and phys_s >= 0.7)\n",
    "            agree_txt   = \"Agreement\" if agree else \"Divergent\"\n",
    "            agree_icon  = \"&#9989;\"   if agree else \"&#9888;\"\n",
    "            agree_color = CLR_HIGH    if agree else CLR_MOD\n",
    "            st.markdown(\n",
    "                f'<div class=\"metric-card\">'\n",
    "                f'<div class=\"metric-lbl\">Physics vs AI</div>'\n",
    "                f'<div class=\"metric-val\" style=\"font-size:1rem;color:{agree_color}\">'\n",
    "                f'{agree_icon} {agree_txt}</div>'\n",
    "                f'<div style=\"font-size:0.68rem;color:#7d8590;margin-top:4px;\">'\n",
    "                f'AI {conf:.2f} | Physics {phys_s:.2f}</div></div>',\n",
    "                unsafe_allow_html=True)\n",
    "\n",
    "        if hf:\n",
    "            _rows = \"\".join([\n",
    "                f'<div style=\"display:flex;justify-content:space-between;'\n",
    "                f'padding:3px 0;border-bottom:1px solid #1e293b32;\">'\n",
    "                f'<span style=\"color:#7d8590;font-size:0.76rem;\">{_k}</span>'\n",
    "                f'<span style=\"color:#e6edf3;font-size:0.76rem;font-weight:600;\">{_v}</span></div>'\n",
    "                for _k, _v in [\n",
    "                    (\"Optical Flow Variance\",  f'{hf.get(\"flow_var_mean\", 0):.4f}'),\n",
    "                    (\"SSIM Drift (mean)\",       f'{hf.get(\"ssim_mean\", 1.0):.4f}'),\n",
    "                    (\"Accel. Kurtosis\",         f'{hf.get(\"accel_kurtosis\", 0):.4f}'),\n",
    "                    (\"Max MCV\",                 f'{hf.get(\"max_mcv\", 0):.4f}'),\n",
    "                    (\"Burst Length (frames)\",   str(hf.get(\"burst_length\", 0))),\n",
    "                    (\"Anomaly Density\",         f'{hf.get(\"anomaly_density\", 0):.4f}'),\n",
    "                ]\n",
    "            ])\n",
    "            st.markdown(\n",
    "                f'<div style=\"margin-top:12px;padding:14px 18px;background:#111827;'\n",
    "                f'border:1px solid #1e293b;border-radius:10px;\">'\n",
    "                f'<div style=\"font-size:0.68rem;color:#7d8590;text-transform:uppercase;'\n",
    "                f'letter-spacing:.1em;margin-bottom:10px;\">Hybrid Feature Signal</div>'\n",
    "                f'{_rows}</div>',\n",
    "                unsafe_allow_html=True)\n",
    "    else:\n",
    "        st.info(\n",
    "            \"**AI classifier not trained yet** -- physics score is your forensic signal.\\n\\n\"\n",
    "            \"To enable AI Decision Confidence:\\n\\n\"\n",
    "            \"```\\n\"\n",
    "            \"python scripts/evaluate_dataset.py\\n\"\n",
    "            \"python scripts/train_meta_classifier.py\\n\"\n",
    "            \"```\\n\\n\"\n",
    "            \"Restart the dashboard after training.\"\n",
    "        )\n",
    "\n",
]

AI_PANEL_TEXT = "".join(AI_PANEL_LINES)

with open(APP, "r", encoding="utf-8") as f:
    content = f.read()

# Find the Synced Video Player anchor
ANCHOR = "    # \u2500\u2500 Synced Video Player"
idx = content.find(ANCHOR)
if idx == -1:
    # Try plain dashes
    ANCHOR = "    # -- Synced Video Player"
    idx = content.find(ANCHOR)

if idx == -1:
    print("ERROR: anchor not found!")
    # Print lines around character 29000-29400 to diagnose
    chunk = content[29000:29400]
    for i, line in enumerate(chunk.split("\n")):
        print(f"  {i}: {repr(line)}")
else:
    new_content = content[:idx] + AI_PANEL_TEXT + content[idx:]
    with open(APP, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"SUCCESS injected at index {idx}  new_size={len(new_content)}")
