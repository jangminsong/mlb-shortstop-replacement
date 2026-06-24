import pandas as pd
import numpy as np

# --------------------
# Load files
# --------------------
season = pd.read_excel("combined-SS-all_years_Contract_Injuries.xlsx")
inj = pd.read_excel("Injury.xlsx")

# --------------------
# Clean + standardize Injury.xlsx
# (columns shown in the file include: Name, Injury / Surgery Date, Missed Days,
#  Average days missed if getting that injury, etc.)
# --------------------
inj = inj.rename(columns=lambda c: str(c).strip())

# Try to locate the key columns robustly
name_col = "Name"
date_col = "Injury / Surgery Date"
avg_col  = "Average days missed if getting that injury"
missed_col = "Missed Days"

for c in [name_col, date_col, avg_col]:
    if c not in inj.columns:
        raise ValueError(f"Missing column in Injury.xlsx: {c}")

inj[name_col] = inj[name_col].astype(str).str.strip()
inj[date_col] = pd.to_datetime(inj[date_col], errors="coerce")
inj["year"] = inj[date_col].dt.year

inj[avg_col] = pd.to_numeric(inj[avg_col], errors="coerce")
if missed_col in inj.columns:
    inj[missed_col] = pd.to_numeric(inj[missed_col], errors="coerce")

# Expected severity proxy for that injury event:
# prefer your average-days column; if missing, fall back to actual missed days; else 0
inj["expected_days_event"] = inj[avg_col]
if missed_col in inj.columns:
    inj["expected_days_event"] = inj["expected_days_event"].fillna(inj[missed_col])
inj["expected_days_event"] = inj["expected_days_event"].fillna(0)

# Aggregate to player-season
inj_agg = (inj.groupby([name_col, "year"], as_index=False)
             .agg(expected_days=("expected_days_event", "sum"),
                  injury_events=("expected_days_event", "size"),
                  max_expected_days=("expected_days_event", "max")))

# --------------------
# Prep season-level file
# --------------------
season = season.rename(columns=lambda c: str(c).strip())
season["full name"] = season["full name"].astype(str).str.strip()

need = ["full name","year","age","WAR","Contract AAV","Contract Yrs Left","Contract Type",
        "Total Injuries","Total Missed Days"]
season = season[[c for c in need if c in season.columns]].copy()

for c in ["year","age","WAR","Contract AAV","Contract Yrs Left","Total Injuries","Total Missed Days"]:
    if c in season.columns:
        season[c] = pd.to_numeric(season[c], errors="coerce")

# Merge expected injury days into season rows
df = season.merge(inj_agg, how="left",
                  left_on=["full name","year"],
                  right_on=[name_col,"year"])

df["expected_days"] = df["expected_days"].fillna(0)
df["injury_events"] = df["injury_events"].fillna(0)

# --------------------
# Scoring helpers
# --------------------
def pct_scale(s):
    # percentile rank scaling to 0..1 (robust to outliers)
    return s.rank(pct=True)

def abs_scale(s):
    # absolute risk based on actual values, scaled to 0..1
    return (s - s.min())/(s.max()-s.min())

def log_scale(s):
    return (np.log1p(s) - np.log1p(s.min()))/(np.log1p(s.max())-np.log1p(s.min()))

def age_risk(age):
    if pd.isna(age): return np.nan
    if age < 22:          return 0.05
    if 22 <= age <= 26:   return 0.10
    if 27 <= age <= 29:   return 0.25
    if 30 <= age <= 32:   return 0.45
    if 33 <= age <= 35:   return 0.70
    return 0.90

# --------------------
# 1) Age risk
# --------------------
df["age_risk"] = df["age"].apply(age_risk)

# --------------------
# 2) Injury risk (uses expected days from Injury.xlsx)
# --------------------
# Actual availability loss (from combined file)
df["actual_days_score"] = pct_scale(df["Total Missed Days"].fillna(0))

# Expected severity loss (from Injury.xlsx averages)
df["expected_days_score"] = pct_scale(df["expected_days"])

# Frequency: you can use Total Injuries from combined file or injury_events from Injury.xlsx.
# If they differ, blend them.
df["inj_freq_raw"] = df["Total Injuries"].fillna(0)
df["injury_freq_score"] = pct_scale(df["inj_freq_raw"])
df["severity_score"] = pct_scale(df["max_expected_days"].fillna(0))


# Combine: give expected days a meaningful weight to fix late-season bias
df["injury_risk"] = (
    0.35 * df["actual_days_score"] +
    0.25 * df["expected_days_score"] +
    0.25 * df["injury_freq_score"] +
    0.15 * df["severity_score"]
).clip(0, 1)

# --------------------
# 3) Contract risk (affordability, discounted by WAR)
# --------------------
df["aav_score"] = pct_scale(df["Contract AAV"].fillna(df["Contract AAV"].median()))
df["yrs_left_score"] = pct_scale(df["Contract Yrs Left"].fillna(0))
df["war_score"] = pct_scale(df["WAR"].fillna(0))

value_discount = (df["war_score"] - 0.5) * 0.30  # tune strength

df["contract_risk"] = (
    0.75*df["aav_score"] +
    0.25*df["yrs_left_score"] -
    value_discount
).clip(0, 1)

type_bump = {"Rookie": -0.05, "Arb": -0.03, "Guaranteed": 0.08}
df["contract_risk"] = (df["contract_risk"] + df["Contract Type"].map(type_bump).fillna(0)).clip(0,1)

# --------------------
# 4) Overall risk
# --------------------
df["overall_risk"] = (
    0.30*df["age_risk"] +
    0.40*df["injury_risk"] +
    0.30*df["contract_risk"]
).clip(0, 1)

# Inspect
show_cols = ["full name","year","age","WAR","Contract AAV","Total Missed Days",
             "expected_days","max_expected_days","inj_freq_raw",
             "age_risk","injury_risk","contract_risk","overall_risk"]
print(df.sort_values(["year","overall_risk"], ascending=[True, False])[show_cols].head(30))

df.to_csv("risk_scores_by_season_with_expected_injury_days.csv", index=False)