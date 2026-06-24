"""
Running Ability / Speed — Correlations
Variables: hp_to_1b, sprint_speed
File: combined-SS-all_years_UPDATED.xlsx (same directory)
Output plots: spd_corr_1_scatter.png, spd_corr_2_distributions.png
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

SPD_VARS = ["hp_to_1b", "sprint_speed"]

for v in SPD_VARS:
    df[v] = pd.to_numeric(df[v], errors="coerce")

df_clean = df[["name", "year"] + SPD_VARS].dropna().reset_index(drop=True)
X = df_clean[SPD_VARS]
print(f"Dataset: {len(df_clean)} complete observations\n")

# ── Correlation ───────────────────────────────────────────────
r, p = stats.pearsonr(X["hp_to_1b"], X["sprint_speed"])
print(f"hp_to_1b vs sprint_speed")
print(f"  Pearson r  : {r:.4f}")
print(f"  p-value    : {p:.4f}")
print(f"  r²         : {r**2:.4f}  ({r**2*100:.1f}% shared variance)")
print(f"  Note: negative r is expected — lower hp_to_1b time = faster runner\n")

print("Descriptive statistics:")
print(X.describe().round(3))

print("\nTop 10 fastest (by sprint_speed):")
print(df_clean.nlargest(10, "sprint_speed")[["name", "year", "sprint_speed", "hp_to_1b"]].to_string(index=False))

# ── Plot 1: scatter with regression ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X["sprint_speed"], X["hp_to_1b"],
           alpha=0.65, color="#1D9E75", edgecolors="white", linewidths=0.4, s=60)

m, b = np.polyfit(X["sprint_speed"], X["hp_to_1b"], 1)
xs = np.linspace(X["sprint_speed"].min(), X["sprint_speed"].max(), 100)
ax.plot(xs, m * xs + b, color="#0F6E56", linewidth=1.8, linestyle="--",
        label=f"r = {r:.3f}")

# label top 5 fastest
top5 = df_clean.nsmallest(5, "hp_to_1b")
for _, row in top5.iterrows():
    ax.annotate(row["name"].split(",")[0],
                (row["sprint_speed"], row["hp_to_1b"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, color="#0F6E56")

ax.set_xlabel("Sprint speed (ft/s)", fontsize=11)
ax.set_ylabel("Home-to-1B time (sec)", fontsize=11)
ax.set_title(f"sprint_speed vs hp_to_1b  (r = {r:.3f},  p = {p:.4f})", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.2, linewidth=0.5)
plt.tight_layout()
plt.savefig("1_running_correlation_scatter.png", dpi=150)
plt.close()
print("\nSaved: 1_running_correlation_scatter.png")

# ── Plot 2: distributions ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, var, color, label in zip(
    axes,
    ["sprint_speed", "hp_to_1b"],
    ["#1D9E75", "#3266ad"],
    ["Sprint speed (ft/s)", "Home-to-1B time (sec)"]
):
    ax.hist(X[var], bins=20, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(X[var].mean(),   color="black",   linewidth=1.2, linestyle="--",
               label=f"mean={X[var].mean():.2f}")
    ax.axvline(X[var].median(), color="#888780", linewidth=1.2, linestyle=":",
               label=f"median={X[var].median():.2f}")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title(var, fontsize=11)
    ax.legend(fontsize=8)

fig.suptitle("Speed Variables — Distributions", fontsize=13)
plt.tight_layout()
plt.savefig("2_running_correlation_istributions.png", dpi=150)
plt.close()
print("Saved: 2_running_correlation_distributions.png")
