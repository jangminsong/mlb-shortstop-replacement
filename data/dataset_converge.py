import pandas as pd

folders = [
    "bat tracking",
    "bat tracking swing path",
    "batted ball",
    "arm strength",
    "arm value",
    "baserunning run value",
    "basestealing run value",
    "exit velocity",
    "expected stats",
    "fielding run value",
    "outs above average",
    "run value",
    "sprint speed"
]

csv = [
    "bat-tracking",
    "bat-tracking-swing-path",
    "batted-ball",
    "arm_strength",
    "base_running",
    "baserunning_run_value",
    "basestealing_running_game",
    "exit_velocity",
    "expected_stats",
    "fielding-run-value",
    "outs_above_average",
    "swing-take",
    "sprint_speed"
]

years = ["2023","2024","2025"]

for folder, name in zip(folders, csv):

    dfs = []

    for year in years:
        file = f"{folder}/{name}-{year}.csv"
        df = pd.read_csv(file)
        dfs.append(df)

    merged = pd.concat(dfs)

    merged.to_csv(f"{folder}/{name}-all.csv", index=False)

print("All datasets merged.")