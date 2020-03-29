from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score

MULTINOMIAL_NB = "MultinomialNB"
BERNOULLI_NB = "BernoulliNB"
KNN = "KNeighborsClassifier"
SUPPORT_VEC = "SVC"

#METRICS
F1_MACRO = "f1_macro"
PRECISION_MACRO = "precision_macro"
RECALL_MACRO = "recall_macro"


class Classification:
    def __init__(self, X, y, classifier = MultinomialNB()):
        self.clf = classifier
        self.features = X
        self.targets = y
        self.f1_macro = 0
        self.precision_macro = 0
        self.recall_macro = 0

    def get_classifier(self, classifier):
        """
        classifier factory
        :param classifier: MULTINOMIAL_NB or BERNOULLI_NB or KNN or SVC
        :return: self
        """
        if classifier == MULTINOMIAL_NB:
            self.clf = MultinomialNB()
        elif classifier == BERNOULLI_NB:
            self.clf = BernoulliNB()
        elif classifier == KNN:
            self.clf = KNeighborsClassifier(n_neighbors=6)
        else:
            self.clf = SVC(class_weight="balanced")
        return self

    def eval(self, features=None, targets=None):
        if features is not None:
            self.features = features
        if targets is not None:
            self.targets = targets
        self.f1_macro = cross_val_score(self.clf, self.features, self.targets, cv=5, scoring='f1_macro')
        self.precision_macro = cross_val_score(self.clf, self.features, self.targets, cv=5, scoring="precision_macro")
        self.recall_macro = cross_val_score(self.clf, self.features, self.targets, cv=5, scoring="recall_macro")
        return self

    def report_metrics(self, metrics=[F1_MACRO, PRECISION_MACRO, RECALL_MACRO]):
        for metric in metrics:
            print("{}: {:0.2f} (+/- {:0.2f})".format(metric, *self.compute_metric(metric)))

    def compute_metric(self, metric=F1_MACRO):
        """
        computes specified metric
        :param metric: f1_macro, precision_macro, recall_macro
        :return: return mean and std
        """
        if metric == F1_MACRO:
            return self.f1_macro.mean(), self.f1_macro.std() * 2
        elif metric == PRECISION_MACRO:
            return self.precision_macro.mean(), self.precision_macro.std() * 2
        elif metric == RECALL_MACRO:
            return self.recall_macro.mean(), self.recall_macro.std() * 2


if __name__ == "__main__":
    """
    REPORT CLASSIFICATION PERFORMANCE METRICS
    USAGE: python classfication.py
    Dependencies: training_data_file.TF, training_data_file.IDF, training_data_file.TFIDF
    """
    # load training data
    features_tf, targets = load_svmlight_file("training_data_file.TF")
    features_idf, targets = load_svmlight_file("training_data_file.IDF")
    features_tf_idf, targets = load_svmlight_file("training_data_file.TFIDF")

    for name in ["MultinomialNB", "BernoulliNB", "KNeighborsClassifier", "SVC"]:
        if name == "MultinomialNB":
            clf = MultinomialNB()
        elif name == "BernoulliNB":
            clf = BernoulliNB()
        elif name == "KNeighborsClassifier":
            clf = KNeighborsClassifier(n_neighbors=6)
        else:
            clf = SVC(class_weight="balanced")

        print("=============" * 5)
        print("Feature - Term Frequency, Classifier - {}, Metrics:".format(name))
        print("=============" * 5)
        classification = Classification(classifier=clf)
        classification.eval(features_tf, targets=targets).report_metrics()
        print("=============" * 5)
        print("Feature - Inverse Document Frequency, Classifier - {}, Metrics:".format(name))
        print("=============" * 5)
        classification.eval(features_idf, targets=targets).report_metrics()
        print("=============" * 5)
        print("Feature - TFIDF, Classifier - {}, Metrics:".format(name))
        print("=============" * 5)
        classification.eval(features_tf_idf, targets=targets).report_metrics()
