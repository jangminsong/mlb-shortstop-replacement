import pandas as pd
import os

'''
files = [
    "bat tracking/bat-tracking-all.csv",
    "bat tracking swing path/bat-tracking-swing-path-all.csv",
    "batted ball/batted-ball-all.csv",
    "arm strength/arm_strength-all.csv",
    "arm value/base_running-all.csv",
    "baserunning run value/baserunning_run_value-all.csv",
    "basestealing run value/basestealing_running_game-all.csv",
    "exit velocity/exit_velocity-all.csv",
    "expected stats/expected_stats-all.csv",
    "fielding run value/fielding-run-value-all.csv",
    "outs above average/outs_above_average-all.csv",
    "run value/swing-take-all.csv",
    "sprint speed/sprint_speed-all.csv"
]

df = pd.read_csv(files[0])

for file in files[1:]:
    temp = pd.read_csv(file)

    # keep only NEW columns (plus keys for alignment)
    new_cols = ["id", "year"] + [c for c in temp.columns if c not in df.columns]
    temp = temp[new_cols]

    df = df.merge(temp, on=["id", "year"], how="left")

df.to_csv("dataset_converge/combined-all.csv", index=False)

'''

df = pd.read_csv("dataset_converge/combined-all.csv")

print(df.shape)  # rows, columns
print("Duplicate (id,year) rows:", df.duplicated(subset=["id","year"]).sum())
print(df["year"].value_counts().sort_index())