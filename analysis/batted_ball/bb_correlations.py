"""
Batted Ball Profile — Correlations
Variables: gb_rate, air_rate, fb_rate, ld_rate, pu_rate, pull_rate, straight_rate
File: combined-SS-all_years_UPDATED.xlsx
Output plots: bb_corr_1_heatmap.png, bb_corr_2_pairwise_bars.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────
FILE = "../../combined-SS-all_years_UPDATED.xlsx"

df_raw = pd.read_excel(FILE, header=None)
df = df_raw.iloc[1:].copy()
df.columns = df_raw.iloc[1].tolist()
df = df.iloc[1:].reset_index(drop=True)

BB_VARS = ["gb_rate", "air_rate", "fb_rate", "ld_rate", "pu_rate",
           "pull_rate", "straight_rate"]

for v in BB_VARS:
    df[v] = pd.to_numeric(df[v], errors="coerce")

df_clean = df[["name", "year"] + BB_VARS].dropna().reset_index(drop=True)
X = df_clean[BB_VARS]
print(f"Dataset: {len(df_clean)} complete observations\n")

# ── Pearson correlation matrix ────────────────────────────────
corr = X.corr(method="pearson")
print("Pearson Correlation Matrix:")
print(corr.round(3).to_string())

# ── P-value matrix ────────────────────────────────────────────
p_matrix = pd.DataFrame(index=BB_VARS, columns=BB_VARS, dtype=float)
for c1 in BB_VARS:
    for c2 in BB_VARS:
        if c1 == c2:
            p_matrix.loc[c1, c2] = 0.0
        else:
            _, p = stats.pearsonr(X[c1], X[c2])
            p_matrix.loc[c1, c2] = round(p, 4)

print("\nP-value Matrix:")
print(p_matrix.to_string())

# ── Pairwise summary ──────────────────────────────────────────
rows = []
for i, v1 in enumerate(BB_VARS):
    for v2 in BB_VARS[i+1:]:
        r, p = stats.pearsonr(X[v1], X[v2])
        flag = "REDUNDANT" if abs(r) > 0.9 else "moderate" if abs(r) > 0.5 else "weak"
        rows.append({"var1": v1, "var2": v2, "r": round(r, 3), "p": round(p, 4), "note": flag})
summary = pd.DataFrame(rows).sort_values("r", key=abs, ascending=False)

print("\nPairwise summary (sorted by |r|):")
print(summary.to_string(index=False))

# ── Plot 1: heatmap ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 10}, ax=ax)
ax.set_title("Batted Ball Profile — Correlation Heatmap", fontsize=13, pad=12)
plt.xticks(rotation=40, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("bb_corr_1_heatmap.png", dpi=150)
plt.close()
print("\nSaved: bb_corr_1_heatmap.png")

# ── Plot 2: pairwise bar chart ────────────────────────────────
colors = summary["r"].apply(
    lambda r: "#3266ad" if abs(r) > 0.9 else "#1D9E75" if abs(r) > 0.5 else "#888780"
)
labels = (summary["var1"] + "  vs  " + summary["var2"]).tolist()
n_bars = len(labels)
fig_height = max(10, n_bars * 0.55)
fig, ax = plt.subplots(figsize=(13, fig_height))

bars = ax.barh(range(n_bars), summary["r"].values,
               color=colors.values, alpha=0.85, height=0.6)
ax.set_yticks(range(n_bars))
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlim(-1.1, 1.6)
ax.set_xlabel("Pearson r", fontsize=11)
ax.set_title("Batted Ball Profile — Pairwise Correlations", fontsize=13, pad=12)

for bar, (_, row) in zip(bars, summary.iterrows()):
    xpos  = row["r"] + 0.03 if row["r"] >= 0 else row["r"] - 0.03
    align = "left" if row["r"] >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"r={row['r']:+.2f}  [{row['note']}]",
            va="center", ha=align, fontsize=8.5)

blue_p = mpatches.Patch(color="#3266ad", label="|r| > 0.9 (redundant)")
teal_p = mpatches.Patch(color="#1D9E75", label="|r| 0.5-0.9 (moderate)")
gray_p = mpatches.Patch(color="#888780", label="|r| < 0.5 (weak)")
ax.legend(handles=[blue_p, teal_p, gray_p], fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("bb_corr_2_pairwise_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: bb_corr_2_pairwise_bars.png")
