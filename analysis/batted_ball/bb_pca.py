"""
Batted Ball Profile — PCA
Variables: gb_rate, air_rate, fb_rate, ld_rate, pu_rate, pull_rate, straight_rate
File: combined-SS-all_years_UPDATED.xlsx
Output plots: bb_pca_1_explained_variance.png, bb_pca_2_loadings.png, bb_pca_3_biplot.png
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

BB_VARS = ["gb_rate", "air_rate", "fb_rate", "ld_rate", "pu_rate",
           "pull_rate", "straight_rate"]

for v in BB_VARS:
    df[v] = pd.to_numeric(df[v], errors="coerce")

df_clean = df[["name", "year"] + BB_VARS].dropna().reset_index(drop=True)
X = df_clean[BB_VARS]
print(f"Dataset: {len(df_clean)} complete observations\n")

# ── PCA ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

pca = PCA(n_components=len(BB_VARS))
pcs = pca.fit_transform(X_sc)

loadings = pd.DataFrame(
    pca.components_.T,
    index=BB_VARS,
    columns=[f"PC{i+1}" for i in range(len(BB_VARS))]
)

print("Explained variance per component:")
for i, ev in enumerate(pca.explained_variance_ratio_):
    cum = pca.explained_variance_ratio_[:i+1].sum()
    print(f"  PC{i+1}: {ev*100:.1f}%  (cumulative: {cum*100:.1f}%)")

print("\nLoadings (PC1-PC4):")
print(loadings.iloc[:, :4].round(3))

# ── Plot 1: scree + cumulative ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(range(1, len(BB_VARS)+1), pca.explained_variance_ratio_ * 100,
            color="#3266ad", alpha=0.85, edgecolor="white")
for i, ev in enumerate(pca.explained_variance_ratio_):
    axes[0].text(i+1, ev*100+0.5, f"{ev*100:.1f}%", ha="center", fontsize=8)
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Explained variance (%)")
axes[0].set_title("Scree Plot", fontsize=12)
axes[0].set_xticks(range(1, len(BB_VARS)+1))

cum = np.cumsum(pca.explained_variance_ratio_) * 100
axes[1].plot(range(1, len(BB_VARS)+1), cum, "o-", color="#3266ad", linewidth=2)
axes[1].axhline(80, color="#E24B4A", linestyle="--", linewidth=1, label="80% threshold")
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative variance (%)")
axes[1].set_title("Cumulative Explained Variance", fontsize=12)
axes[1].set_xticks(range(1, len(BB_VARS)+1))
axes[1].legend(fontsize=9)

plt.suptitle("Batted Ball Profile PCA — Explained Variance", fontsize=13)
plt.tight_layout()
plt.savefig("bb_pca_1_explained_variance.png", dpi=150)
plt.close()
print("\nSaved: bb_pca_1_explained_variance.png")

# ── Plot 2: loadings bar (PC1 & PC2) ─────────────────────────
pct1 = round(pca.explained_variance_ratio_[0] * 100, 1)
pct2 = round(pca.explained_variance_ratio_[1] * 100, 1)

x_pos = np.arange(len(BB_VARS))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x_pos - width/2, loadings["PC1"], width,
       label=f"PC1 ({pct1}% var)", color="#3266ad", alpha=0.85)
ax.bar(x_pos + width/2, loadings["PC2"], width,
       label=f"PC2 ({pct2}% var)", color="#BA7517", alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(BB_VARS, rotation=30, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Loading")
ax.set_title("Batted Ball Profile PCA — Loadings (PC1 & PC2)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("bb_pca_2_loadings.png", dpi=150)
plt.close()
print("Saved: bb_pca_2_loadings.png")

# ── Plot 3: biplot ────────────────────────────────────────────
df_clean["PC1"] = pcs[:, 0]
df_clean["PC2"] = pcs[:, 1]

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_clean["PC1"], df_clean["PC2"],
           alpha=0.5, color="#888780", s=45, edgecolors="white", linewidths=0.3,
           label="Players")

top6 = df_clean.nlargest(6, "PC1")
for _, row in top6.iterrows():
    ax.annotate(row["name"].split(",")[0],
                (row["PC1"], row["PC2"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, color="#0C447C")

scale = 3.0
for var, (l1, l2) in zip(BB_VARS, zip(loadings["PC1"], loadings["PC2"])):
    ax.annotate("", xy=(l1*scale, l2*scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#E24B4A", lw=1.2))
    ax.text(l1*scale*1.1, l2*scale*1.1, var,
            fontsize=8, color="#E24B4A", ha="center")

ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
ax.set_xlabel(f"PC1 ({pct1}% variance)", fontsize=10)
ax.set_ylabel(f"PC2 ({pct2}% variance)", fontsize=10)
ax.set_title("Batted Ball Profile PCA — Biplot", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.2, linewidth=0.5)
plt.tight_layout()
plt.savefig("bb_pca_3_biplot.png", dpi=150)
plt.close()
print("Saved: bb_pca_3_biplot.png")
