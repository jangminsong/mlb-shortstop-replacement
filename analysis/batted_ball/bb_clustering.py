"""
Batted Ball Profile — Clustering
Variables: gb_rate, air_rate, fb_rate, ld_rate, pu_rate, pull_rate, straight_rate
File: combined-SS-all_years_UPDATED.xlsx
Output plots: bb_clust_1_silhouette.png, bb_clust_2_scatter.png, bb_clust_3_means.png
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

# ── Scale & PCA (for scatter plot) ───────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_sc)
df_clean["PC1"] = pcs[:, 0]
df_clean["PC2"] = pcs[:, 1]

pct1 = round(pca.explained_variance_ratio_[0] * 100, 1)
pct2 = round(pca.explained_variance_ratio_[1] * 100, 1)

# ── Silhouette sweep k=2..6 ───────────────────────────────────
sil = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(X_sc)
    sil.append(silhouette_score(X_sc, lab))

best_k = np.argmax(sil) + 2
print(f"Silhouette scores k=2..6: {[round(s, 3) for s in sil]}")
print(f"Optimal k = {best_k}\n")

# ── Fit final KMeans ──────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean["cluster"] = km_final.fit_predict(X_sc)

# ── Print cluster summaries ───────────────────────────────────
print("Cluster means:")
print(df_clean.groupby("cluster")[BB_VARS].mean().round(3))

print("\nTop 5 players per cluster (by gb_rate):")
palette = {0: "#888780", 1: "#3266ad", 2: "#1D9E75", 3: "#BA7517"}
for c in range(best_k):
    sub = df_clean[df_clean["cluster"] == c].sort_values("gb_rate", ascending=False)
    print(f"\n  Cluster {c} (n={len(sub)}):")
    print(sub[["name", "year", "gb_rate", "fb_rate", "pull_rate"]].head(5).to_string(index=False))

# ── Plot 1: silhouette scores ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(2, 7), sil, "o-", color="#3266ad", linewidth=2, markersize=7)
ax.axvline(best_k, color="#E24B4A", linestyle="--", linewidth=1.2, label=f"Best k={best_k}")
for k, s in zip(range(2, 7), sil):
    ax.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=8)
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("Batted Ball Profile Clustering — Silhouette Score vs k", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("bb_clust_1_silhouette.png", dpi=150)
plt.close()
print("\nSaved: bb_clust_1_silhouette.png")

# ── Plot 2: PCA scatter coloured by cluster ───────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for c, grp in df_clean.groupby("cluster"):
    ax.scatter(grp["PC1"], grp["PC2"],
               color=palette.get(c, "#E24B4A"),
               label=f"Cluster {c} (n={len(grp)})",
               alpha=0.75, s=60, edgecolors="white", linewidths=0.4)
    top5 = grp.nlargest(5, "gb_rate")
    for _, row in top5.iterrows():
        ax.annotate(row["name"].split(",")[0],
                    (row["PC1"], row["PC2"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=palette.get(c, "#E24B4A"))

ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
ax.set_xlabel(f"PC1 ({pct1}% variance)", fontsize=10)
ax.set_ylabel(f"PC2 ({pct2}% variance)", fontsize=10)
ax.set_title("Batted Ball Profile Clustering — PCA Scatter", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, linewidth=0.5)
plt.tight_layout()
plt.savefig("bb_clust_2_scatter.png", dpi=150)
plt.close()
print("Saved: bb_clust_2_scatter.png")

# ── Plot 3: cluster mean comparison (z-scored) ───────────────
means   = df_clean.groupby("cluster")[BB_VARS].mean()
means_z = (means - X.mean()) / X.std()

x_pos = np.arange(len(BB_VARS))
width = 0.8 / best_k
fig, ax = plt.subplots(figsize=(11, 5))
for i, c in enumerate(range(best_k)):
    offset = (i - best_k / 2 + 0.5) * width
    ax.bar(x_pos + offset, means_z.loc[c], width,
           label=f"Cluster {c}", color=palette.get(c, "#E24B4A"), alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(BB_VARS, rotation=20, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Z-score vs overall mean")
ax.set_title("Batted Ball Profile Clustering — Cluster Mean Comparison (standardised)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("bb_clust_3_means.png", dpi=150)
plt.close()
print("Saved: bb_clust_3_means.png")
