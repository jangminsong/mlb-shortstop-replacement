"""
Running Ability / Speed — Recommend Variables
Variables: hp_to_1b, sprint_speed
File: combined-SS-all_years_UPDATED.xlsx (same directory)
Output plots: spd_rec_1_shared_variance.png, spd_rec_2_percentiles.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── Analysis ──────────────────────────────────────────────────
r, p = stats.pearsonr(X["hp_to_1b"], X["sprint_speed"])
shared = r ** 2

print(f"hp_to_1b vs sprint_speed")
print(f"  r          : {r:.4f}")
print(f"  r²         : {shared:.4f}  ({shared*100:.1f}% shared variance)")
print(f"  p-value    : {p:.4f}")

# Cases where the two disagree most (largest residual)
m, b = np.polyfit(X["sprint_speed"], X["hp_to_1b"], 1)
df_clean["expected_hp"] = m * X["sprint_speed"] + b
df_clean["residual"]    = X["hp_to_1b"] - df_clean["expected_hp"]

print("\nPlayers where hp_to_1b is SLOWER than expected for their sprint_speed (top 5):")
slow = df_clean.nlargest(5, "residual")[["name", "year", "sprint_speed", "hp_to_1b", "residual"]]
print(slow.round(3).to_string(index=False))

print("\nPlayers where hp_to_1b is FASTER than expected for their sprint_speed (top 5):")
fast = df_clean.nsmallest(5, "residual")[["name", "year", "sprint_speed", "hp_to_1b", "residual"]]
print(fast.round(3).to_string(index=False))

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KEEP
  ✓  sprint_speed   primary metric
                    standardised (ft/s), league-wide benchmark,
                    higher = unambiguously faster

  CONSIDER DROPPING
  ⚠  hp_to_1b       r = {r:.3f} with sprint_speed
                    {shared*100:.1f}% shared variance → near-redundant
                    only adds value if batter-box-to-1B timing
                    is specifically needed (e.g. bunt/infield hit analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ── Plot 1: shared variance bar ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(["sprint_speed", "hp_to_1b"],
        [1 - shared, 1 - shared],
        color=["#1D9E75", "#3266ad"], alpha=0.75, label="Unique variance")
ax.barh(["sprint_speed", "hp_to_1b"],
        [shared, shared],
        left=[1 - shared, 1 - shared],
        color=["#888780", "#888780"], alpha=0.5,
        label=f"Shared variance ({shared*100:.1f}%)")
ax.set_xlim(0, 1.4)
ax.set_xlabel("Proportion of total variance")
ax.set_title(f"sprint_speed vs hp_to_1b — Shared Variance  (r = {r:.3f})", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("1_running_recommendation_shared_variance.png", dpi=150)
plt.close()
print("Saved: 1_running_recommendation_shared_variance.png")

# ── Plot 2: top 15 by sprint_speed — both metrics side by side
df_clean["sprint_pct"] = df_clean["sprint_speed"].rank(pct=True) * 100
df_clean["hp_pct"]     = df_clean["hp_to_1b"].rank(pct=True, ascending=False) * 100

top15 = df_clean.nlargest(15, "sprint_speed")
y     = np.arange(len(top15))

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(y - 0.2, top15["sprint_pct"].values, 0.35,
        color="#1D9E75", alpha=0.85, label="sprint_speed percentile")
ax.barh(y + 0.2, top15["hp_pct"].values,     0.35,
        color="#3266ad", alpha=0.85, label="hp_to_1b percentile (inverted)")
ax.set_yticks(y)
ax.set_yticklabels(top15["name"].str.split(",").str[0] + " (" + top15["year"].astype(str) + ")",
                   fontsize=8)
ax.set_xlabel("Percentile rank")
ax.set_title("Top 15 by Sprint Speed — Percentile Comparison", fontsize=12)
ax.axvline(50, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
ax.legend(fontsize=9)
ax.grid(alpha=0.2, axis="x")
plt.tight_layout()
plt.savefig("2_running_recommendation_percentiles.png", dpi=150)
plt.close()
print("Saved: 2_running_recommendation_percentiles.png")

# ── Plot 3: residual scatter (where the two vars disagree) ────
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(X["sprint_speed"], X["hp_to_1b"],
                c=df_clean["residual"], cmap="RdBu_r",
                alpha=0.75, s=60, edgecolors="white", linewidths=0.3)
xs = np.linspace(X["sprint_speed"].min(), X["sprint_speed"].max(), 100)
ax.plot(xs, m * xs + b, color="black", linewidth=1.2, linestyle="--", alpha=0.5,
        label="regression line")

# label biggest disagreements
for _, row in pd.concat([fast, slow]).iterrows():
    ax.annotate(row["name"].split(",")[0],
                (row["sprint_speed"], row["hp_to_1b"]),
                textcoords="offset points", xytext=(6, 3), fontsize=7)

plt.colorbar(sc, ax=ax, label="Residual (positive = slower hp_to_1b than expected)")
ax.set_xlabel("Sprint speed (ft/s)", fontsize=11)
ax.set_ylabel("Home-to-1B time (sec)", fontsize=11)
ax.set_title("Where sprint_speed and hp_to_1b disagree", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.2, linewidth=0.5)
plt.tight_layout()
plt.savefig("3_running_recommendation_residuals.png", dpi=150)
plt.close()
print("Saved: 3_running_recommendation_residuals.png")
