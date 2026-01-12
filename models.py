from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def run_kmeans(X, k=6, init='k-means++', n_init=20, max_iter=500):
    """Run KMeans with optimized parameters for better convergence."""
    X_arr = X.toarray() if hasattr(X, 'toarray') else X
    model = KMeans(
        n_clusters=k,
        init=init,  # K-means++ initialization
        n_init=n_init,  # Run algorithm 20 times with different initializations
        max_iter=max_iter,  # More iterations for convergence
        random_state=42
    )
    labels = model.fit_predict(X_arr)
    return labels, model


def run_hierarchical(X, k=6, linkage='ward'):
    """Run Hierarchical clustering with Ward linkage."""
    X_arr = X.toarray() if hasattr(X, 'toarray') else X
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X_arr)
    return labels


def run_dbscan(X, eps=0.8, min_samples=3):
    """Run DBSCAN with tuned parameters for sparse data."""
    X_arr = X.toarray() if hasattr(X, 'toarray') else X
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = model.fit_predict(X_arr)
    return labels
