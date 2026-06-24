
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

st.set_page_config(page_title="MLB Player Similarity Search", layout="wide")

FILE_PATH = r"C:\Users\14799\Desktop\602\combined-SS-all_years FINAL.xlsx"
SHEET_NAME = "in"
HEADER_ROW = 1
K = 5
COMPONENTS = 2

FEATURE_COLS = [
    "est_woba", "avg_hit_speed", "brl_percent",
    "whiff_per_swing", "avg_bat_speed", "squared_up_per_swing",
    "fb_rate", "pull_rate",
    "runner_runs_XB", "runner_runs_SBX", "rate_sbx",
    "sprint_speed", "range_runs", "dp_runs", "arm_ss"
]

NAME_COL = "name"


@st.cache_data
def load_and_prepare_data():
    df_raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW)

    use_cols = [NAME_COL] + FEATURE_COLS
    df = df_raw[use_cols].dropna().copy()
    df[NAME_COL] = df[NAME_COL].astype(str).str.strip()

    X = df[FEATURE_COLS].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    embed = SpectralEmbedding(
        n_components=COMPONENTS,
        affinity="nearest_neighbors",
        n_neighbors=min(10, X_scaled.shape[0] - 1),
        random_state=42
    )
    X_embedded = embed.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_embedded)

    df["cluster"] = labels
    df["embed_1"] = X_embedded[:, 0]
    df["embed_2"] = X_embedded[:, 1]
    return df, X_embedded


def find_top_matches(player_name, df, X_embedded, top_n=3):
    matches = df[df[NAME_COL].str.lower() == player_name.lower()]
    if matches.empty:
        return None, None, None

    idx = matches.index[0]
    target_cluster = df.loc[idx, "cluster"]

    same_cluster = df[df["cluster"] == target_cluster].copy()
    same_cluster = same_cluster[same_cluster.index != idx]

    if same_cluster.empty:
        return df.loc[idx], target_cluster, same_cluster

    target_vec = X_embedded[df.index.get_loc(idx)].reshape(1, -1)
    cluster_vecs = X_embedded[[df.index.get_loc(i) for i in same_cluster.index]]

    dists = pairwise_distances(target_vec, cluster_vecs, metric="euclidean")[0]
    same_cluster["distance"] = dists
    same_cluster = same_cluster.sort_values("distance").head(top_n)

    show_cols = [NAME_COL, "cluster", "distance"] + FEATURE_COLS
    return df.loc[idx], target_cluster, same_cluster[show_cols]


st.title("MLB Player Similarity Search")
st.write("Search any player from your file and return the top 3 closest players from the same cluster.")

df, X_embedded = load_and_prepare_data()

player_list = sorted(df[NAME_COL].unique().tolist())

col1, col2 = st.columns([2, 1])

with col1:
    selected_player = st.selectbox("Choose a player", player_list)

with col2:
    top_n = st.number_input("Number of similar players", min_value=1, max_value=10, value=3, step=1)

if st.button("Search"):
    target_row, target_cluster, result = find_top_matches(selected_player, df, X_embedded, top_n=top_n)

    if target_row is None:
        st.error("Player not found.")
    else:
        st.subheader("Selected Player")
        st.dataframe(pd.DataFrame([target_row[[NAME_COL, "cluster"] + FEATURE_COLS]]), use_container_width=True)

        st.subheader(f"Top {top_n} Closest Players in Cluster {target_cluster}")
        if len(result) == 0:
            st.warning("No other players found in the same cluster.")
        else:
            st.dataframe(result.reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.subheader("All Players by Cluster")
st.dataframe(df[[NAME_COL, "cluster"] + FEATURE_COLS], use_container_width=True)
