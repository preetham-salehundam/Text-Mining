from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.datasets import load_svmlight_file
from classification import Classification, MULTINOMIAL_NB, BERNOULLI_NB, KNN, SUPPORT_VEC, F1_MACRO
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from util import argmax
import warnings
warnings.filterwarnings("ignore")

# constants
CHI = "chi2"
MI = "mutual info"
color_schema = {MULTINOMIAL_NB: "red", BERNOULLI_NB: "green", KNN: "blue", SUPPORT_VEC: "yellow"}
N_PROCESSES = cpu_count()


def fit_transform(selection_instance, X, y, K):
    """
    fit_transform for feature selection
    :param selection_instance: CHI or MI
    :param X: selected features
    :param y: targets of selected features
    :param K: no of selected features
    :return:
    """
    return selection_instance(X, y), K


def compute(classifier_instance, metric):
    """
    compute metrics
    :param classifier_instance: Multinomial NB, Bernouli NB, K-Nearest Neighbor, Support vec instance
    :param metric: f1_macro, precision_macro, recall_macro
    :return: metric
    """
    instance = classifier_instance.eval()
    K = instance.features.shape[-1]
    return instance.compute_metric(metric), K


def plot_k_vs_f1(X, y, k=(100, 20000), selection_method=CHI):
    log_file = None
    if selection_method == CHI:
        method = chi2
        log_file = open("CHI.values", "w")
    else:
        method = mutual_info_classif
        log_file = open("MI.values", "w")
    pool = Pool(processes=N_PROCESSES)
    # selecting K best features using CHI or MI ofr K's ranging from 100 to 20000
    params = [(SelectKBest(method, k=K).fit_transform, X, y , K) for K in range(k[0], k[1], 300)]
    # accumulate selected feature sets
    feature_selection_results = pool.starmap(fit_transform, params)
    # pass the newly selected feature X_new and y to the classifier
    for classifier in [MULTINOMIAL_NB, BERNOULLI_NB, KNN, SUPPORT_VEC]:
        print("processing ", classifier)
        classifier_instances = [Classification(X_new, y).get_classifier(classifier=classifier) for X_new, K in
                                    feature_selection_results]

        eval_params = [(instance, F1_MACRO) for instance in classifier_instances]
        # needs classifier instance and metric to be computed
        evaluation_results = pool.starmap(compute, eval_params)

        # for reporting purposes
        k_s = []
        mean = []
        p_std, n_std = [], []

        for scores, k in evaluation_results:
            k_s.append(k)
            mean.append(scores[0])
            p_std.append(scores[0] + scores[1])
            n_std.append(scores[0] - scores[1])

        plt.fill_between(k_s, p_std, n_std, alpha=0.2, color=color_schema[classifier])
        plt.plot(k_s, mean, color=color_schema[classifier], label=classifier)

        # plot a line at max peak
        max_k_idx = argmax(mean)
        max_k_value = k_s[max_k_idx]

        # for logging purposes
        log_file.write("{}\n".format(classifier))
        log_file.write("{} {} \n".format("max_k", str(max_k_value)))
        log_file.write("{} {} \n".format("max_mean", str(max(mean))))
        log_file.write("{} {} \n".format("K_S", str(k_s)))
        log_file.write("{} {} \n".format("p_std", str(p_std)))
        log_file.write("{} {} \n".format("n_std", str(n_std)))

        print("The best K - {} and the corresponding mean is {} for classifier {}".format(max_k_value, mean[max_k_idx], classifier))

        plt.axvline(max_k_value, linestyle="--", color=color_schema[classifier], alpha=0.4)
        plt.axhline(max(mean), linestyle="--", color=color_schema[classifier], alpha=0.4)

        plt.xlabel("K - values")
        plt.ylabel("Mean F1 scores")
        plt.legend(loc='upper right')
        plt.title("No. of features vs F1 scores for "+selection_method)

    return plt


if __name__ == "__main__":
    """
    USAGE: python feature_selection.py
    
    the default training file names are selected and would fail if those files are not found
    Dependencies: training_data_file.TF, training_data_file.IDF, training_data_file.TFIDF
    """

    feature_tf, targets = load_svmlight_file("training_data_file.TF")
    feature_idf, targets = load_svmlight_file("training_data_file.IDF")
    feature_tf_idf, targets = load_svmlight_file("training_data_file.TFIDF")

    # plot chi square selected feature set vs f1 scores
    plt = plot_k_vs_f1(feature_tf_idf, targets, selection_method=CHI, k=(100, 20000))
    plt.show()

    # plot mutual information selected feature set vs f1 scores
    plt = plot_k_vs_f1(feature_tf_idf, targets, selection_method=MI, k=(100, 20000))
    plt.show()


