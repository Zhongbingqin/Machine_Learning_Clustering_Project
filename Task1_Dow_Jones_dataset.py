# import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import time

'''--------------------------------------------load data----------------------------------------'''
dow_data = pd.read_csv('dow_jones_index.data')

'''--------------------------------------------feature engineering------------------------------'''
# 1. remove dollar symbol
dollar_feas = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
for name in dollar_feas:
    rem_sym = [x.strip('$') for x in dow_data[name].values]
    dow_data[name] = rem_sym
# 2. date feature processing
dow_data['date'] = pd.to_datetime(dow_data['date'])
dow_data['date_month'] = dow_data['date'].dt.month
dow_data['date_day'] = dow_data['date'].dt.day
# 3. drop nominal and date features
df = dow_data.drop(['stock', 'date', 'percent_change_price', 'percent_change_volume_over_last_wk',
                    'days_to_next_dividend', 'percent_return_next_dividend'], axis=1)
# 4. missing data processing
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)

'''--------------------------------------------normalize data------------------------------'''
mms = MinMaxScaler()
x_std = mms.fit_transform(imputed_data)

'''--------------------------------------------reduce dimensions using PCA------------------------------'''
# feature selection using PCA
# determine the number of components
pca = PCA(n_components=3)
fit = pca.fit(x_std)
features = fit.transform(x_std)
ratio = fit.explained_variance_ratio_
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid()
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

'''--------------------------------general functions: compute SSE, CSM, display CSM plot--------------------------'''
# compute SSE for DBSCAN and Agglomerative algorithm
def get_sse(X, labels, model_name):
    df = pd.DataFrame(X)
    df['labels'] = labels
    sse = 0
    if model_name == "DBSCAN":
        n = len(set(labels))-1
    else:
        n = len(set(labels))
    for c in np.unique(labels):
        if c <0:
            continue
        cluster = df.loc[df['labels'] == c]
        centroid = cluster.mean()[:-1].values
        for x in cluster.iloc[:, :-1].values:
            sse += np.sum((x-centroid)**2)
    print("For", model_name, "when n_clusters =", n, "The SSE is :", round(sse, 2))

# compute CSM for DBSCAN and Agglomerative
def get_csm(X, labels, model_name):
    silhouette_avg = silhouette_score(X, labels)
    print("For", model_name, "n_clusters =", c, "The average silhouette_score is :", round(silhouette_avg, 2))

# display the CSM plot
def csm_plot(n_clusters, X, labels, model_name):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters.
    silhouette_avg = silhouette_score(X, labels)

    # Compute the silhouette scores for each sample.
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", y=-0.01)
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    unique_labels = set(labels)
    cmap = cm.get_cmap("Spectral")
    colors = [cmap(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters, y=-0.01)
    plt.suptitle(("Silhouette analysis for", model_name, "clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')



'''--------------------------------------------Kmeans-------------------------------------------'''
'''-----------compute SSE of Kmeans--------------'''
sse_km = []
for c in range(2, 10):
    model = KMeans(n_clusters=c, random_state=0)
    model.fit(features)
    sse = model.inertia_
    sse_km.append(sse)
    print("For Kmeans, when n_clusters =", c, "The SSE is :", round(sse, 2))
plt.plot(range(2, 10), sse_km, '-o', color='black')
plt.xlabel('number of clusters, c')
plt.ylabel('inertia')
plt.xticks(range(2, 10))
plt.show()

'''-----------compute and display CSM of Kmeans--------------'''
# determine the optimal n_clusters by comparing the values of CSM
for c in range(2, 10):
    km_labels = KMeans(n_clusters=c, random_state=10).fit_predict(features)
    csm_km = get_csm(features, km_labels, "Kmeans")

def csm_km_plot(n_clusters, X, labels, model):

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters.
    silhouette_avg = silhouette_score(X, labels)

    # Compute the silhouette scores for each sample.
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", y=-0.01)
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    cmap = cm.get_cmap("Spectral")
    colors = cmap(labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = model.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title('The visualization of the clustered data.', y=-0.01)
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for Kmeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

for c in range(2, 10):
    km_model = KMeans(n_clusters=c, random_state=10)
    km_labels = km_model.fit_predict(features)
    csm_km_plot(c, features, km_labels, km_model)
    plt.show()

'''--------------------------------------------DBSCAN-------------------------------------------'''
'''-----------compute and display CSM of DBSCAN--------------'''
# determine the rough range of eps
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(features)
distances, indices = nbrs.kneighbors(features)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()

for i in np.arange(0.15, 0.3, 0.01):
    dbscan = DBSCAN(eps=i, min_samples=6).fit(features)
    dbscan_labels = dbscan.labels_
    c = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    csm_km = get_csm(features, dbscan_labels, "DBSCAN")
    print('eps = ', round(i, 2))

# Plot result when n_clusters = 2
db = DBSCAN(eps=0.2, min_samples=6).fit(features)
db_labels = db.labels_
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

core_samples_mask = np.zeros_like(db_labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

csm_plot(n_clusters, features, db_labels, "DBSCAN")
plt.show()

'''-----------compute SSE of DBSCAN--------------'''
sse = get_sse(features, db_labels, "DBSCAN")
print(sse)

'''--------------------------------------------Agglomerative clustering-------------------------------------------'''
'''-----------compute and display CSM of Agglomerative clustering--------------'''
# compute the CSM
for c in range(2, 10):
    agg = AgglomerativeClustering(n_clusters=c, linkage="complete").fit(features)
    agg_labels = agg.labels_
    csm_km = get_csm(features, agg_labels, "Agglomerative")

# Plot result when n_clusters = 2
agg = AgglomerativeClustering(n_clusters=2, linkage="average")
agg_labels = agg.fit_predict(features)
agg_fit = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(features)

core_samples_mask = np.zeros_like(agg_labels, dtype=bool)
csm_plot(2, features, agg_labels, "Agglomerative")
plt.show()

'''-----------compute SSE of Agglomerative clustering--------------'''
sse = get_sse(features, agg_labels, "Agglomerative")
print(sse)


start_time = time.time()
KMeans(n_clusters=2, random_state=10).fit(features)
print("--- %s seconds ---" % round(time.time() - start_time, 2))

start_time = time.time()
DBSCAN(eps=0.2, min_samples=6).fit(features)
print("--- %s seconds ---" % round(time.time() - start_time, 2))

start_time = time.time()
AgglomerativeClustering(n_clusters=2, linkage="average").fit(features)
print("--- %s seconds ---" % round(time.time() - start_time, 2))