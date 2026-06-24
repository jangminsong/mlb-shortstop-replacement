"""
Running Ability / Speed — PCA
Variables: hp_to_1b, sprint_speed
File: combined-SS-all_years_UPDATED.xlsx (same directory)
Output plots: spd_pca_1_explained_variance.png, spd_pca_2_loadings.png, spd_pca_3_scores.png

Note: With only 2 variables PCA is essentially a rotation.
PC1 captures the shared speed signal; PC2 the small residual difference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

SPD_VARS = ["hp_to_1b", "sprint_speed"]

for v in SPD_VARS:
    df[v] = pd.to_numeric(df[v], errors="coerce")

df_clean = df[["name", "year"] + SPD_VARS].dropna().reset_index(drop=True)
X = df_clean[SPD_VARS]
print(f"Dataset: {len(df_clean)} complete observations\n")

# ── PCA ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_sc)
df_clean["PC1"] = pcs[:, 0]
df_clean["PC2"] = pcs[:, 1]

loadings = pd.DataFrame(
    pca.components_.T,
    index=SPD_VARS,
    columns=["PC1", "PC2"]
)

pct1 = round(pca.explained_variance_ratio_[0] * 100, 1)
pct2 = round(pca.explained_variance_ratio_[1] * 100, 1)

print("Explained variance:")
print(f"  PC1: {pct1}%  — overall speed (both vars load similarly)")
print(f"  PC2: {pct2}%  — residual difference between the two measures\n")

print("Loadings:")
print(loadings.round(4))

print("\nInterpretation:")
print("  PC1 loads near-equally on both vars → composite speed score")
print("  PC2 separates players where sprint_speed and hp_to_1b disagree")

print("\nTop 10 by PC1 (highest composite speed):")
top10 = df_clean.nlargest(10, "PC1")[["name", "year", "sprint_speed", "hp_to_1b", "PC1", "PC2"]]
print(top10.round(3).to_string(index=False))

# ── Plot 1: explained variance ────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["PC1", "PC2"], pca.explained_variance_ratio_ * 100,
       color=["#1D9E75", "#888780"], alpha=0.85, edgecolor="white")
for i, ev in enumerate(pca.explained_variance_ratio_):
    ax.text(i, ev * 100 + 0.5, f"{ev*100:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("Explained variance (%)")
ax.set_title("Speed PCA — Explained Variance", fontsize=12)
plt.tight_layout()
plt.savefig("1_running_pca_explained_variance.png", dpi=150)
plt.close()
print("\nSaved: 1_running_pca_explained_variance.png")

# ── Plot 2: loadings ──────────────────────────────────────────
x_pos = np.arange(2)
width = 0.35
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x_pos - width/2, loadings["PC1"], width,
       label=f"PC1 ({pct1}% var)", color="#1D9E75", alpha=0.85)
ax.bar(x_pos + width/2, loadings["PC2"], width,
       label=f"PC2 ({pct2}% var)", color="#3266ad", alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(SPD_VARS, fontsize=11)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Loading")
ax.set_title("Speed PCA — Loadings", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("2_running_pca_loadings.png", dpi=150)
plt.close()
print("Saved: 2_running_pca_loadings.png")

# ── Plot 3: PC1 scores (ranked player speed composite) ────────
ranked = df_clean.sort_values("PC1", ascending=True).tail(25)  # top 25

fig, ax = plt.subplots(figsize=(8, 8))
colors = ["#1D9E75" if v > 0 else "#888780" for v in ranked["PC1"]]
ax.barh(ranked["name"].str.split(",").str[0] + " (" + ranked["year"].astype(str) + ")",
        ranked["PC1"], color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("PC1 score (composite speed — higher = faster)")
ax.set_title("Speed PCA — Top 25 Players by PC1 Score", fontsize=12)
plt.tight_layout()
plt.savefig("3_running_pca_scores.png", dpi=150)
plt.close()
print("Saved: 3_running_pca_scores.png")
