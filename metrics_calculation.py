import pandas as pd
import numpy as np
import time
import math


def calculate_sse(X, y):
    """
    X : data points for clustering

    y : array, shape = [n_samples]
    The corresponding predicted label to each data point
    Only calculate cluster labels >= 0
    """
    df = pd.DataFrame(X)
    df['y'] = y
    sse = 0
    for c in np.unique(y):
        if c < 0:
            continue
        cluster = df.loc[df['y'] == c]
        centroid = cluster.mean()[:-1].values
        for x in cluster.iloc[:, :-1].values:
            sse += np.sum((x - centroid) ** 2)
    return sse


def time_counting(X, algorithm):
    t0 = time.time()
    algorithm.fit(X)
    t1 = time.time()
    time_cost = t1 - t0
    return time_cost


def clustering_score(time_cost, sse, csm, n_samples, noise_ratio):
    score = (csm*n_samples)*(1-noise_ratio)/(math.exp(time_cost)*sse)
    return score