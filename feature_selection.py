from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.datasets import load_svmlight_file
from classification import Classification, MULTINOMIAL_NB, BERNOULLI_NB, KNN, SUPPORT_VEC, F1_MACRO
from multiprocessing import Pool
import matplotlib.pyplot as plt
from util import argmax

#constants
CHI = "chi2"
MI = "mutual info"
color_schema = {MULTINOMIAL_NB: "red", BERNOULLI_NB: "green", KNN: "blue", SUPPORT_VEC: "yellow"}

# def diff(l1, l2):
#     i = 0
#     l1_diff = []
#     l2_diff = []
#     while i < len(l1) and i < len(l2):
#         if l1[i] != l2[i]:
#             l1_diff.append(l1[i])
#             l2_diff.append(l2[i])
#         i = i + 1
#
#     if i < len(l1):
#         l1_diff.extend(l1[i:])
#     if i < len(l2):
#         l2_diff.extend(l2[i:])
#
#     return l1_diff, l2_diff

def fit_transform(selection_instance, X, y, K):
    return selection_instance(X, y), K


def compute(classifier_instance, metric):
    instance = classifier_instance.eval()
    K = instance.features.shape[-1]
    return instance.compute_metric(metric), K


def __plot__():
    pass

def plot_k_vs_f1(X, y, k=(100, 20000), selection_method=CHI):
    log_file = None
    if selection_method == CHI:
        method = chi2
        log_file = open("CHI.values", "w")
    else:
        method = mutual_info_classif
        log_file = open("MI.values", "w")
    pool = Pool(processes=4)
    # selecting K best features using CHI or MI ofr K's ranging from 100 to 20000
    params = [(SelectKBest(method, k=K).fit_transform, X, y , K) for K in range(k[0], k[1], 300)]
    # accumulate selected feature sets
    feature_selection_results = pool.starmap(fit_transform, params)
    # pass the newly selected feature X_new and y to the classifier

    for classifier in [MULTINOMIAL_NB, BERNOULLI_NB, KNN, SUPPORT_VEC]:
        print("processing ", classifier)
        classifier_instances = [Classification(X_new, y).get_classifier(classifier=classifier) for X_new, K in
                                    feature_selection_results]
        # needs classifier instance and metric to be computed
        eval_params = [(instance, F1_MACRO) for instance in classifier_instances]
        evaluation_results = pool.starmap(compute, eval_params)

    # for result in results:
    #     print(result.shape)

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
    # for scores, K in evaluation_results:
    #     print(K, scores[0], scores[1], "*")

    return plt


if __name__ == "__main__":

    feature_tf, targets = load_svmlight_file("training_data_file.TF")
    feature_idf, targets = load_svmlight_file("training_data_file.IDF")
    feature_tf_idf, targets = load_svmlight_file("training_data_file.TFIDF")

    plt = plot_k_vs_f1(feature_tf_idf, targets, selection_method=MI, k=(100, 20000))
    plt.show()

    plt = plot_k_vs_f1(feature_tf_idf, targets, selection_method=CHI, k=(100, 20000))
    plt.show()
    #plt.savefig("K_vs_F1.png", format="png")
    #plt = plot_k_vs_f1(feature_tf_idf, targets, selection_method=MI, k=(100, 20000))

    # clf = Classification()
    # pool = Pool(processes=4)

    # X_new1 = pool.starmap(SelectKBest(chi2, k=100).fit_transform, [(feature_tf, targets),  (feature_idf, targets),
    #                                                       (feature_tf_idf, targets)])
    # X_new2 = pool.starmap(SelectKBest(mutual_info_classif, k=100).fit_transform, [(feature_tf, targets),  (feature_idf, targets),
    #                                                       (feature_tf_idf, targets)])
    # # print(X_new1)
    # # print("---------")
    # # print(X_new2)
    # feature_type = ["TF", "IDF", "TFIDF"]
    # for classifier in [MULTINOMIAL_NB, BERNOULLI_NB, KNN, SUPPORT_VEC]:
    #     i = 0
    #     for feature_1, feature_2 in zip(X_new1, X_new2):
    #         print("=============" * 5)
    #         print("Classifier - {},  feature - {} , Metrics:".format(classifier, feature_type[i]))
    #         print("=============" * 5)
    #         print("Using Chi square test")
    #         print("=============" * 5)
    #         clf.get_classifier(classifier).eval(feature_1, targets).report_metrics()
    #         print("=============" * 5)
    #         print("Using Mutual Information")
    #         print("=============" * 5)
    #         clf.get_classifier(classifier).eval(feature_2, targets).report_metrics()
    #         i = i + 1
    #         print("\n\n")

