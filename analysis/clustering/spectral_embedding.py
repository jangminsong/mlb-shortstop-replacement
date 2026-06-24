import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel


components = 2



df = pd.read_excel(r"C:\Users\14799\Desktop\602\combined-SS-all_years FINAL.xlsx", sheet_name="in", header=1)

cols = [
    "est_woba", "avg_hit_speed", "brl_percent",
    "whiff_per_swing", "avg_bat_speed", "squared_up_per_swing",
    "fb_rate", "pull_rate",
    "runner_runs_XB", "runner_runs_SBX", "rate_sbx",
    "sprint_speed", "range_runs", "dp_runs", "arm_ss"
]

cols_name = ["name", "est_woba", "avg_hit_speed", "brl_percent",
    "whiff_per_swing", "avg_bat_speed", "squared_up_per_swing",
    "fb_rate", "pull_rate",
    "runner_runs_XB", "runner_runs_SBX", "rate_sbx",
    "sprint_speed", "range_runs", "dp_runs", "arm_ss"]

X = df[cols].dropna()
data = X.copy()
data_name = df[cols_name].dropna()
# make X
X = X.to_numpy()

print("X shape:", X.shape)
print("NaN count in X:", np.isnan(X).sum())

# scale
X = StandardScaler().fit_transform(X)



# spectral embedding
embed = SpectralEmbedding(
    n_components=components,
    affinity="nearest_neighbors",
    n_neighbors=min(10, X.shape[0] - 1),   # avoid sample-size error
    random_state=42
)
X_embedded = embed.fit_transform(X)

k_values = [3, 4, 5, 6, 7, 8]
scores = []

for k in k_values:
    if k < X_embedded.shape[0]:   # silhouette needs number of samples > number of clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_embedded)
        score = silhouette_score(X_embedded, labels)
        scores.append(score)
    else:
        scores.append(np.nan)

plt.plot(k_values, scores, marker="o")
for i, (k, s) in enumerate(zip(k_values, scores)):
    if not np.isnan(s):
        plt.annotate(
            f"{s:.3f}",
            (k, s),
            textcoords="offset points",
            xytext=(0, 8),   # move label slightly above point
            ha='center'
        )

plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title(f"Silhouette Score vs Number of Clusters for components = {components}")
plt.show()

####
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

labels = kmeans.fit_predict(X_embedded)

print(X_embedded.shape)
# plot
colors = plt.cm.tab10.colors   # up to 10 distinct colors

for cluster in np.unique(labels):
    plt.scatter(
        X_embedded[labels == cluster, 0],
        X_embedded[labels == cluster, 1],
        color=colors[cluster],
        label=f"Cluster {cluster}"
    )
plt.legend()
plt.xlabel("Embedding 1")
plt.ylabel("Embedding 2")
plt.title(f"{k} Clustering for the smallest {components} eigenvectors")
plt.show()

##### plot inertia#####
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_embedded)   # use your spectral embedding output
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Inertia vs k")
plt.show()

df = pd.DataFrame(data)   # convert numpy array → DataFrame
df["cluster"] = labels
data_name["cluster"] = labels
cluster_summary = df.groupby("cluster").mean()
print(cluster_summary)

cluster_4_players = data_name[labels == 4]
print(cluster_4_players["name"])