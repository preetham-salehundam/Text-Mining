from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest, chi2
from multiprocessing import Pool
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from numpy import float32
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


SILHOUETTE = "Silhouette Coefficient"
NMI = "Normalized Mutual Index"


def fit(model, X):
    return model.fit(X)


def plot_k_vs_metrics(K, X, y, k_means_labels, hierarchical_labels, metric=SILHOUETTE):
    """
    plots no.of cluster vs silhouette score/ normalized mutual index
    :param K:  number of clusters
    :param X: Selected features using either CHI or MI
    :param k_means_labels: cluster labels of the data after k means clustering 
    :param hierarchical_labels: cluster labels of the data after hierarchical clustering 
    :param metric: Whether Sihouette Coefficient or Normalized Mutual Index
    :return: None
    """
    k_means_sc, hierarchical_cluster_sc, k_means_nmi, hierarchical_cluster_nmi = {},{},{},{}
    for i, k in enumerate(K):
        if metric == SILHOUETTE:
            k_means_sc[k] = silhouette_score(X, k_means_labels[i], metric="euclidean")
            hierarchical_cluster_sc[k] = silhouette_score(X, hierarchical_labels[i], metric="euclidean")
        elif metric == NMI:
            k_means_nmi[k] = normalized_mutual_info_score(labels_true=y, labels_pred=k_means_labels[i])
            hierarchical_cluster_nmi[k] = normalized_mutual_info_score(labels_true=y, labels_pred=hierarchical_labels[i])
    if metric == SILHOUETTE:
        # plot the n-clusters vs silhouette scores for analysis
        plt.plot(list(K), list(k_means_sc.values()), label="K means")
        plt.plot(list(K), list(hierarchical_cluster_sc.values()), label="hierarchical clustering")
        plt.xlabel("no. of clusters")
        plt.ylabel("silhouette_score")
        plt.legend(loc="best")
        plt.title("n_clusters vs silhouette_score")
        plt.show()
    elif metric == NMI:
        # plot the n-clusters vs NMI scores for analysis
        plt.plot(list(K), list(k_means_nmi.values()), label="K means")
        plt.plot(list(K), list(hierarchical_cluster_nmi.values()), label="hierarchical clustering")
        plt.xlabel("no. of clusters")
        plt.ylabel("NMI")
        plt.legend(loc="best")
        plt.title("n_clusters vs NMI")
        plt.show()


if __name__ == "__main__":

    # read the training data
    TRAINING_FILE = "training_data_file.TFIDF"
    # read the sparse matrix from training file
    features, target = load_svmlight_file(TRAINING_FILE, dtype=float32)
    # fork 4 processes for multiprocessing
    pool = Pool(processes=4)

    # choosing the best K value using CHI-square from previous experiments.
    K_BEST = 5500

    print("selecting {} best features using CHI-Square selection".format(K_BEST))
    # chi squared method has yielded better F1 score for 5200 features in the previous experiment
    X_new = SelectKBest(chi2, k=K_BEST).fit_transform(features, target)

    print("Performing K means clustering with no. of clusters varying from 2 to 25")
    params = [(KMeans(n_clusters=K), X_new) for K in range(2, 26)]

    # k means cluster models
    K_means_cluster_results = pool.starmap(fit, params)

    print("Performing Agglomerative clustering with no. of clusters varying from 2 to 25")
    params = [(AgglomerativeClustering(n_clusters=K, linkage="ward"), X_new.toarray()) for K in range(2, 26)]

    # agglomerative cluster models
    hierarchical_cluster_results = pool.starmap(fit, params)

    # cluster labels
    k_means_labels = [ km.labels_ for km in K_means_cluster_results ]
    hierarchical_labels = [hc.labels_ for hc in hierarchical_cluster_results]

    # plots
    plot_k_vs_metrics(K=range(2, 26), X=X_new, y=target,  k_means_labels=k_means_labels, hierarchical_labels=hierarchical_labels,
                      metric=SILHOUETTE)

    plot_k_vs_metrics(K=range(2, 26), X=X_new, y=target, k_means_labels=k_means_labels, hierarchical_labels=hierarchical_labels,
                      metric=NMI)










