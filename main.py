print("Main started")

from src.data_prep import load_data
from src.feature_engineering import create_basket_text, build_tfidf, apply_pca, apply_svd
from src.models import run_kmeans, run_hierarchical, run_dbscan
from src.evaluation import silhouette, elbow_plot, pca_plot
import numpy as np

# Load data
df = load_data(r"D:\Groceseries\data\Groceries_dataset.csv")

# Feature engineering
baskets = create_basket_text(df)

# TF-IDF with optimized parameters
X, tfidf = build_tfidf(baskets['basket_text'])

# Apply SVD for better clustering on sparse data
X_svd, svd = apply_svd(X, n_components=50)

# PCA for visualization
X_pca, _ = apply_pca(X)

print(f"Original feature shape: {X.shape}")
print(f"SVD reduced shape: {X_svd.shape}")

# Optimal k found: 11
optimal_k = 11

# KMeans on SVD features (best results)
labels_km_svd, km_svd_model = run_kmeans(X_svd, k=optimal_k)
print(f"\n✓ KMeans (SVD, k={optimal_k}) Silhouette: {silhouette(X_svd, labels_km_svd):.6f}")

# KMeans on original features
labels_km, km_model = run_kmeans(X, k=optimal_k)
print(f"✓ KMeans (Original, k={optimal_k}) Silhouette: {silhouette(X, labels_km):.6f}")

# Hierarchical clustering
labels_h = run_hierarchical(X_svd, k=optimal_k)
print(f"✓ Hierarchical (SVD, k={optimal_k}) Silhouette: {silhouette(X_svd, labels_h):.6f}")

# DBSCAN
labels_d = run_dbscan(X_svd)
print(f"✓ DBSCAN Silhouette: {silhouette(X_svd, labels_d):.6f}")

# Visualizations
pca_plot(X_pca, labels_km, out='pca_kmeans_final.png')
pca_plot(X_pca, labels_h, out='pca_hierarchical_final.png')
elbow_plot(X, max_k=6, out='elbow_final.png')

# Print top terms for clusters
print(f"\n{'='*60}")
print(f"Top Terms per Cluster (k={optimal_k})")
print(f"{'='*60}")

terms = tfidf.get_feature_names_out()
centroids = km_model.cluster_centers_

for i, c in enumerate(centroids):
    top_indices = np.argsort(c)[-8:]
    top_terms = terms[top_indices]
    cluster_size = np.sum(labels_km == i)
    print(f"Cluster {i:2d} ({cluster_size:4d} items): {', '.join(top_terms)}")

print(f"{'='*60}")
print("✓ Clustering complete. Visualizations saved.")
