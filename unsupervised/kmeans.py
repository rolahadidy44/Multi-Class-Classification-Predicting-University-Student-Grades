import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supervisedlearning.data_loader import load_data
from supervisedlearning.preprocess import preprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.decomposition import PCA

df = load_data()

# df = preprocess_data(df)

# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# x_pca = pca.fit_transform(x_scaled)

# for k in range(2, 11):
#     model = KMeans(n_clusters=k, random_state=42)
#     model.fit(x_pca)
#     labels = model.labels_
#     score = silhouette_score(x_pca, labels)
#     print(f"k={k} → Silhouette Score (PCA): {score:.3f}")
# print(pca.explained_variance_ratio_)
df_clustering = df.select_dtypes(include=['int64', 'float64'])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_clustering)


# KMeans.fit(x_scaled)
# clusters = KMeans.labels_
# df['cluster'] = clusters

# score = silhouette_score(x_scaled, clusters)
# print(f"silhouette Score: {score:.3f}")


pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(x_pca)
    labels = model.labels_
    score = silhouette_score(x_pca, labels)
    print(f"k={k} → Silhouette Score (PCA): {score:.3f}")
print(pca.explained_variance_ratio_)