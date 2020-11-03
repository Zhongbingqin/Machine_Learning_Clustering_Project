import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def pca_var_plotbar(X, title=''):
    """
    Plot bar for PCA analysis on X
    """
    pca = PCA()
    principal_components = pca.fit_transform(X)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='blue')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.title(title)
    plt.xticks(features)
    plt.show()


def pca_ratio_plotcurve(X, percentage=0.9, title=''):
    """
    Plot curve for PCA analysis on X
    """
    pca = PCA().fit(X)
    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    xi = np.arange(1, len(pca.explained_variance_ratio_) + 1, step=1)
    # return the cumulative sum of the variance ratio along the y-axis
    y = np.cumsum(pca.explained_variance_ratio_)
    # set the y-limits of the y-axis
    plt.ylim(0.0, 1.1)
    # set the tick locations and labels of the x-axis
    plt.xticks(np.arange(0, 11, step=1))
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative variance (%)')
    plt.title(title)
    # Add a horizontal line (threshold line) across the axis
    plt.axhline(y=percentage, color='r', linestyle='-')
    # the position to place the text
    plt.text(0.5, percentage - 0.1, f"{percentage * 100}% cut-off threshold", color='red', fontsize=16)
    ax.grid(axis='both')
    plt.show()