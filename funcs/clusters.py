from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


def k_means(arr, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=0, max_iter=100, n_init=20)
    result = model.fit(arr)
    labels = result.labels_
    print("K_means clustering done!")
    return labels


def minibatch_kmeans(arr, n_clusters):
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, max_iter=100, n_init=20)
    result = model.fit(arr)
    labels = result.labels_
    print("MiniBatchKMeans clustering done!")
    return labels


def spectral(arr, n_clusters):
    model = SpectralClustering(n_clusters=n_clusters, random_state=0)
    result = model.fit(arr)
    labels = result.labels_
    affinity_matrix = result.affinity_matrix_
    print("SpectralClustering clustering done!")
    return labels


def agglomerative(arr, n_clusters, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    result = model.fit(arr)
    labels = result.labels_
    n_leaves_ = result.n_leaves_
    n_components_ = result.n_components_
    children_ = result.children_
    print("AgglomerativeClustering clustering done!")
    return labels


def dbscan(arr, n_clusters, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    result = model.fit(arr)
    labels = result.labels_
    core_sample_indices = result.core_sample_indices_
    components = result.components_
    print("DBSCAN clustering done!")
    return labels


clusters = {'KMeans': k_means,
            'MiniBatchKMeans': minibatch_kmeans,
            'Spectral': spectral,
            'Agglomerative': agglomerative,
            'DBSCAN': dbscan,
            }
