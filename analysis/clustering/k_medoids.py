import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances

# =========================
# K-Medoids (no sklearn_extra needed)
# =========================
class KMedoids:
    def __init__(self, n_clusters=5, random_state=42, max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def fit_predict(self, X):
        rng = np.random.default_rng(self.random_state)
        D = pairwise_distances(X)
        n = len(X)

        # Initialize medoids randomly
        medoid_ids = rng.choice(n, self.n_clusters, replace=False)

        for _ in range(self.max_iter):
            # Assign each point to nearest medoid
            labels = np.argmin(D[:, medoid_ids], axis=1)

            new_medoids = medoid_ids.copy()
            for k in range(self.n_clusters):
                cluster_pts = np.where(labels == k)[0]
                if len(cluster_pts) == 0:
                    continue
                sub_D = D[np.ix_(cluster_pts, cluster_pts)]
                new_medoids[k] = cluster_pts[np.argmin(sub_D.sum(axis=1))]

            if np.all(new_medoids == medoid_ids):
                break
            medoid_ids = new_medoids

        self.medoid_indices_ = medoid_ids
        self.labels_ = np.argmin(D[:, medoid_ids], axis=1)
        return self.labels_

# =========================
# Load data
# =========================
file_path = "combined-SS-all_years FINAL.xlsx"   # <-- update path if needed
df = pd.read_excel(file_path, sheet_name="in", header=1)

cols = [
    "est_woba", "avg_hit_speed", "brl_percent",
    "whiff_per_swing", "avg_bat_speed", "squared_up_per_swing",
    "fb_rate", "pull_rate",
    "runner_runs_XB", "runner_runs_SBX", "rate_sbx",
    "sprint_speed", "range_runs", "dp_runs", "arm_ss"
]

data = df[cols].dropna().copy()

player_col = "name" if "name" in df.columns else None
players = df.loc[data.index, player_col] if player_col else pd.Series(data.index, index=data.index)

# =========================
# Groups
# =========================
offense_cols = [
    "est_woba", "avg_hit_speed", "brl_percent",
    "whiff_per_swing", "avg_bat_speed", "squared_up_per_swing",
    "fb_rate", "pull_rate"
]
defense_cols = ["range_runs", "dp_runs", "arm_ss"]
running_cols = ["runner_runs_XB", "runner_runs_SBX", "rate_sbx", "sprint_speed"]

def standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Feature Sets
X_A = standardize(data[cols])

remove_vars = ["pull_rate"]
trimmed_cols = [c for c in cols if c not in remove_vars]
X_B = standardize(data[trimmed_cols])

X_off = standardize(data[offense_cols])
pca = PCA(n_components=3)
X_off_pca = pca.fit_transform(X_off)
X_C = np.concatenate([
    X_off_pca,
    standardize(data[defense_cols]),
    standardize(data[running_cols])
], axis=1)

# =========================
# Clustering function
# =========================
def run_kmedoids(X, name):
    print("\n" + "="*80)
    print(f"K-MEDOIDS RESULTS: {name}")
    print("="*80)

    model = KMedoids(n_clusters=5, random_state=42)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)
    print(f"\nSilhouette Score: {score:.4f}")

    df_temp = data.copy()
    df_temp["cluster"] = labels
    df_temp["player"] = players

    cluster_means = df_temp.groupby("cluster")[cols].mean()
    print("\nCluster Means (Original Variables):")
    print(cluster_means.round(3))

    print("\nPlayers per Cluster:")
    for c in sorted(df_temp["cluster"].unique()):
        print(f"\nCluster {c}:")
        print(df_temp[df_temp["cluster"] == c]["player"].head(30).tolist())

    return labels, score, cluster_means

# =========================
# RUN ALL THREE
# =========================
labels_A, score_A, means_A = run_kmedoids(X_A, "Feature Set A (Full)")
labels_B, score_B, means_B = run_kmedoids(X_B, "Feature Set B (Trimmed)")
labels_C, score_C, means_C = run_kmedoids(X_C, "Feature Set C (Offense PCA)")

print("\n" + "="*80)
print("SILHOUETTE SCORE SUMMARY")
print("="*80)
print(f"Feature Set A (Full):         {score_A:.4f}")
print(f"Feature Set B (Trimmed):      {score_B:.4f}")
print(f"Feature Set C (Offense PCA):  {score_C:.4f}")
