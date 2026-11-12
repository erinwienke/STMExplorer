from sklearn.cluster import KMeans


# Cluster embeddings using k-means and return WCSS
def cluster_embeddings(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15).fit(X)
    return model.labels_, float(model.inertia_)
