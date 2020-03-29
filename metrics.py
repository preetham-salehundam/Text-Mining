
'''
by Mathieu Blondel
'''

import numpy as np
from math import log10

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])


    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best



def logarithmic_tf(term_frequency):
    return (1 + log10(term_frequency)) if term_frequency > 0 else 0

def augmented_tf(term_frequency, max_word_tf):
    return 0.5 + (0.5 * term_frequency)/ float(max_word_tf) # type conversion for backward compatibility

def logarithmic_avg(term_frequency, avg_word_tf ):
    return (1 + log10(term_frequency)) / (1 + log10(avg_word_tf))

def probabilistic_idf(N, df):
    return max([0, log10((N-df)/df)])

def idf(N, df):
    return log10(N/df)