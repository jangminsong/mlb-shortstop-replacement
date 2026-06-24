import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# load data
df = pd.read_csv(r"C:\Users\14799\Desktop\602\player_year_xy_risk.csv")

players = df["player_name"].tolist()
coords = df[["x", "y"]].values
risk = df["overall_risk"].values

# distance → similarity
dist_matrix = cdist(coords, coords, metric="euclidean")
sim_matrix = 1 / (1 + dist_matrix / dist_matrix.mean())

# recommendation factor:
# row i recommending column j → use risk[j]
rec_matrix = 0.5 * sim_matrix + 0.5 * (1 - risk)[None, :]

# OPTIONAL: remove self-recommendation
np.fill_diagonal(rec_matrix, 0)

# make table
df_rec = pd.DataFrame(rec_matrix, index=players, columns=players)

print(df_rec.head())

# save
df_rec.to_csv(r"C:\Users\14799\Desktop\602\pairwise_recommendation_table.csv")