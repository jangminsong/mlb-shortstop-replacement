"""
Running Ability / Speed — Clustering
Variables: hp_to_1b, sprint_speed
File: combined-SS-all_years_UPDATED.xlsx (same directory)
Output plots: spd_clust_1_silhouette.png, spd_clust_2_scatter.png, spd_clust_3_means.png
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

SPD_VARS = ["hp_to_1b", "sprint_speed"]

for v in SPD_VARS:
    df[v] = pd.to_numeric(df[v], errors="coerce")

df_clean = df[["name", "year"] + SPD_VARS].dropna().reset_index(drop=True)
X = df_clean[SPD_VARS]
print(f"Dataset: {len(df_clean)} complete observations\n")

# ── Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# ── Silhouette sweep k=2..5 ───────────────────────────────────
sil = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(X_sc)
    sil.append(silhouette_score(X_sc, lab))

best_k = np.argmax(sil) + 2
print(f"Silhouette scores k=2..5: {[round(s, 3) for s in sil]}")
print(f"Optimal k = {best_k}\n")

# ── Fit final KMeans — forced k=3 (slow / average / fast) ────
best_k = 3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean["cluster"] = km_final.fit_predict(X_sc)

# Re-label clusters by sprint_speed mean so 0=slow, 1=average, 2=fast
order = df_clean.groupby("cluster")["sprint_speed"].mean().sort_values().index.tolist()
remap = {old: new for new, old in enumerate(order)}
df_clean["cluster"] = df_clean["cluster"].map(remap)

# ── Print summaries ───────────────────────────────────────────
print("Cluster means:")
print(df_clean.groupby("cluster")[SPD_VARS].mean().round(3))

print("\nTop 5 fastest per cluster (by sprint_speed):")
palette = {0: "#3266ad", 1: "#1D9E75", 2: "#E24B4A", 3: "#BA7517"}
for c in range(best_k):
    sub = df_clean[df_clean["cluster"] == c].sort_values("sprint_speed", ascending=False)
    print(f"\n  Cluster {c} (n={len(sub)}):")
    print(sub[["name", "year", "sprint_speed", "hp_to_1b"]].head(5).to_string(index=False))

# ── Plot 1: silhouette ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(2, 6), sil, "o-", color="#1D9E75", linewidth=2, markersize=7)
ax.axvline(best_k, color="#E24B4A", linestyle="--", linewidth=1.2, label=f"Chosen k={best_k}")
for k, s in zip(range(2, 6), sil):
    ax.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=8)
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("Speed Clustering — Silhouette Score vs k", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("spd_clust_1_silhouette.png", dpi=150)
plt.close()
print("\nSaved: spd_clust_1_silhouette.png")

# ── Plot 2: scatter coloured by cluster ───────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for c, grp in df_clean.groupby("cluster"):
    ax.scatter(grp["sprint_speed"], grp["hp_to_1b"],
               color=palette.get(c, "#E24B4A"),
               label=["Slow","Average","Fast"][c] + f" (n={len(grp)})",
               alpha=0.75, s=65, edgecolors="white", linewidths=0.4)
    top3 = grp.nsmallest(3, "hp_to_1b")
    for _, row in top3.iterrows():
        ax.annotate(row["name"].split(",")[0],
                    (row["sprint_speed"], row["hp_to_1b"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7, color=palette.get(c, "#E24B4A"))

ax.set_xlabel("Sprint speed (ft/s)", fontsize=11)
ax.set_ylabel("Home-to-1B time (sec)", fontsize=11)
ax.set_title("Speed Clustering — Scatter by Cluster", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, linewidth=0.5)
plt.tight_layout()
plt.savefig("spd_clust_2_scatter.png", dpi=150)
plt.close()
print("Saved: spd_clust_2_scatter.png")

# ── Plot 3: cluster mean bar ──────────────────────────────────
means   = df_clean.groupby("cluster")[SPD_VARS].mean()
means_z = (means - X.mean()) / X.std()

x_pos = np.arange(len(SPD_VARS))
width = 0.8 / best_k
fig, ax = plt.subplots(figsize=(7, 4))
for i, c in enumerate(range(best_k)):
    offset = (i - best_k / 2 + 0.5) * width
    ax.bar(x_pos + offset, means_z.loc[c], width,
           label=["Slow","Average","Fast"][c], color=palette.get(c, "#E24B4A"), alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(SPD_VARS, fontsize=11)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Z-score vs overall mean")
ax.set_title("Speed Clustering — Cluster Mean Comparison (standardised)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("spd_clust_3_means.png", dpi=150)
plt.close()
print("Saved: spd_clust_3_means.png")
