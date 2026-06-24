import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

df = pd.read_excel(r"C:\Users\14799\Desktop\602\mypart.xlsx")
#print(df.head())
df = df.fillna(df.mean())
scaler = StandardScaler()


data_scaled = scaler.fit_transform(df)

corr = df.corr()
var = df.var()

df_reduced = df[[ "max_hit_speed", "brl_percent"]]


pca = PCA(n_components=2)
data_scaled = pca.fit_transform(df_reduced)
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(df_reduced)
plt.scatter(data_scaled[:,0], data_scaled[:,1], c = labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
# # plot
# plt.figure(figsize=(8,6))
# sns.heatmap(
#     corr,
#     annot=True,          # show numbers
#     cmap='coolwarm',     # color style
#     fmt=".2f",           # 2 decimal places
#     square=True
# )

# plt.title("Correlation Matrix")
# plt.show()
# dist = 1 - df.corr().abs()

# linkage_matrix = linkage(dist, method='ward')

# plt.figure(figsize=(12,6))
# dendrogram(linkage_matrix, labels=df.columns, leaf_rotation=90)
# plt.show()


from sklearn.metrics import silhouette_score
var = ""
for str in df_reduced:
    var = var + " " + str
print("Silhouette_score for k-mean clusters of " + var)
for k in range(2,6):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(data_scaled)
    print(k, silhouette_score(data_scaled, labels))
