from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def silhouette(X, labels):
    X_arr = X.toarray() if hasattr(X, 'toarray') else X
    unique = set(labels)
    if len(unique) <= 1:
        return float('nan')
    try:
        return silhouette_score(X_arr, labels)
    except Exception:
        return float('nan')


def elbow_plot(X, max_k=6, out='elbow.png'):
    from sklearn.cluster import KMeans
    X_arr = X.toarray() if hasattr(X, 'toarray') else X
    inertias = []
    ks = range(1, max_k + 1)
    for k in ks:
        inertias.append(KMeans(n_clusters=k, random_state=42).fit(X_arr).inertia_)
    plt.figure()
    plt.plot(list(ks), inertias, '-o')
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.savefig(out)
    plt.close()


def pca_plot(X_pca, labels, out='pca.png'):
    plt.figure(figsize=(6,4))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', s=10)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
