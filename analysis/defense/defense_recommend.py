"""
Defensive Performance — Recommend Variables
Variables: total_runs, range_runs, arm_runs, dp_runs,
           fielding_runs_prevented, outs_above_average, arm_ss
File: combined-SS-all_years_UPDATED.xlsx (same directory)
Output plots: def_rec_1_redundancy_heatmap.png, def_rec_2_independence.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────
FILE = "../../combined-SS-all_years_UPDATED.xlsx"

df_raw = pd.read_excel(FILE, header=None)
df = df_raw.iloc[1:].copy()
df.columns = df_raw.iloc[1].tolist()
df = df.iloc[1:].reset_index(drop=True)

DEF_VARS = ["total_runs", "range_runs", "arm_runs", "dp_runs",
            "fielding_runs_prevented", "outs_above_average", "arm_ss"]

for v in DEF_VARS:
    df[v] = pd.to_numeric(df[v], errors="coerce")

df_clean = df[["name", "year"] + DEF_VARS].dropna().reset_index(drop=True)
X = df_clean[DEF_VARS]
print(f"Dataset: {len(df_clean)} complete observations\n")

# ── Redundancy check ──────────────────────────────────────────
corr = X.corr()

print("Redundancy check — correlation with outs_above_average:")
print(f"  {'Variable':<30}  {'r':>6}  Decision")
print("  " + "-" * 55)
for v in DEF_VARS:
    if v == "outs_above_average":
        print(f"  {'outs_above_average':<30}  {'anchor':>6}  ✓ keep (primary metric)")
        continue
    r = corr.loc[v, "outs_above_average"]
    if abs(r) > 0.9:
        decision = "✗ drop — REDUNDANT"
    elif abs(r) < 0.5:
        decision = "✓ keep — independent"
    else:
        decision = "↔ keep — moderate overlap"
    print(f"  {v:<30}  {r:+.3f}  {decision}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KEEP
  ✓  outs_above_average      anchor — best single defensive metric
  ✓  arm_ss                  independent arm strength signal
  ✓  arm_runs                arm value in run terms (different signal)
  ✓  dp_runs                 double-play turning ability

  DROP (redundant with outs_above_average)
  ✗  total_runs              r ≈ 0.97
  ✗  range_runs              r ≈ 1.00
  ✗  fielding_runs_prevented r ≈ 1.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ── Plot 1: redundancy heatmap ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

ax.set_xticks(range(len(DEF_VARS)))
ax.set_yticks(range(len(DEF_VARS)))
ax.set_xticklabels(DEF_VARS, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(DEF_VARS, fontsize=9)

for i in range(len(DEF_VARS)):
    for j in range(len(DEF_VARS)):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Defensive Variables — Redundancy Heatmap", fontsize=12)
plt.tight_layout()
plt.savefig("defense_recommendation_redundancy_heatmap.png", dpi=150)
plt.close()
print("Saved: defense_recommendation_redundancy_heatmap.png")

# ── Plot 2: independence from OAA bar chart ───────────────────
keep_vars = ["outs_above_average", "arm_ss", "arm_runs", "dp_runs"]
drop_vars = ["total_runs", "range_runs", "fielding_runs_prevented"]
all_review = keep_vars + drop_vars

r_with_oaa   = [abs(corr.loc[v, "outs_above_average"]) if v != "outs_above_average" else 1.0
                for v in all_review]
independence = [1.0 if v == "outs_above_average" else 1 - abs(corr.loc[v, "outs_above_average"])
                for v in all_review]
bar_colors   = ["#3266ad" if v in keep_vars else "#E24B4A" for v in all_review]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(all_review, independence, color=bar_colors, alpha=0.85)

for bar, ind, r_val, v in zip(bars, independence, r_with_oaa, all_review):
    if v == "outs_above_average":
        label = "anchor metric"
    else:
        label = f"r with OAA = {r_val:.2f}"
    ax.text(ind + 0.01, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9)

ax.set_xlim(0, 1.45)
ax.set_xlabel("Independence from outs_above_average  (1 = fully independent)")
ax.set_title("Defensive Variables — Independence Check", fontsize=12)
ax.axvline(0.1, color="#E24B4A", linestyle="--", linewidth=0.8, alpha=0.6,
           label="< 0.1 = effectively redundant")

import matplotlib.patches as mpatches
keep_p = mpatches.Patch(color="#3266ad", label="Recommended: keep")
drop_p = mpatches.Patch(color="#E24B4A", label="Recommended: drop")
ax.legend(handles=[keep_p, drop_p], fontsize=9)
plt.tight_layout()
plt.savefig("defense_recommendation_independence.png", dpi=150)
plt.close()
print("Saved: defense_recommendation_independence.png")
