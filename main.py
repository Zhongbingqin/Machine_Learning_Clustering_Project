from itertools import cycle, islice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler

from diagrams import pca_var_plotbar, pca_ratio_plotcurve
from metrics_calculation import calculate_sse, clustering_score, time_counting


from sklearn.manifold import TSNE
# read data from files
dow_jones_filepath = "datasets/dow_jones_index.data"
live_filepath = "datasets/Live.csv"
sales_filepath = "datasets/Sales_Transactions_Dataset_Weekly.csv"
water_filepath = "datasets/water-treatment.data"

print("Reading data from file paths ...")

dow_jones_data = pd.read_csv(dow_jones_filepath)
live_data = pd.read_csv(live_filepath)
sales_data = pd.read_csv(sales_filepath)
water_data = pd.read_csv(water_filepath,
                         names=["Date", "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E",
                                "PH-P", "DBO-P", "SS-P", "SSV-P", "SED-P", "COND-P", "PH-D", "DBO-D", "DQO-D", "SS-D",
                                "SSV-D", "SED-D", "COND-D", "PH-S", "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S",
                                "COND-S",
                                "RD-DBO-P", "RD-SS-P", "RD-SED-P", "RD-DBO-S", "RD-DQO-S", "RD-DBO-G", "RD-DQO-G",
                                "RD-SS-G", "RD-SED-G"])

# ######################################################################################################################
# pre-processing the four datasets
print("\nPre-processing dow_jones_index dataset ...")
# 1. remove dollar symbol and convert it to ndarray format
dollar_cols = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
for col in dollar_cols:
    dow_jones_data[col] = [x.strip('$') for x in dow_jones_data[col].values]
# 2. date feature processing
dow_jones_data['date'] = pd.to_datetime(dow_jones_data['date'])
# extract month and day information from the date column
dow_jones_data['month'] = dow_jones_data['date'].dt.month
dow_jones_data['day'] = dow_jones_data['date'].dt.day
# 3. drop redundant features
dow_jones_data = dow_jones_data.drop(['stock', 'date', 'percent_change_price', 'percent_change_volume_over_last_wk',
                                      'days_to_next_dividend', 'percent_return_next_dividend'], axis=1)
# 4. missing data processing
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(dow_jones_data.values)
imputed_dow_jones_data = imr.transform(dow_jones_data.values)
# normalize data
X_dow_jones = MinMaxScaler().fit_transform(imputed_dow_jones_data)

print("\nPre-processing facebook live sellers dataset ...")
# 1. date feature processing
live_data['status_published'] = pd.to_datetime(live_data['status_published'])
live_data['status_published'] = live_data.status_published.values.astype(np.int64) // 10**9
# 2. drop redundant columns
live_data = live_data.drop(['status_id', 'status_type', 'Column1', 'Column2', 'Column3', 'Column4'], axis=1)
# normalize data
X_live = MinMaxScaler().fit_transform(live_data)

print("\nPre-processing sales transactions dataset ...")
normalised_cols = ['MIN', 'MAX', 'Product_Code']
for col in sales_data.columns:
    if str(col).startswith('Normalize'):
        normalised_cols.append(str(col))
sales_data = sales_data.drop(normalised_cols, axis=1)
# normalize data
X_sales = MinMaxScaler().fit_transform(sales_data)

print("\nPre-processing water treatment plant dataset ...")
# 1. drop the date column
water_data = water_data.drop('Date', axis=1)
# 2. missing data processing
water_data = water_data.replace('?', np.nan)
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(water_data.values)
imputed_water_data = imr.transform(water_data.values)
# normalize data
X_water = MinMaxScaler().fit_transform(imputed_water_data)

# ######################################################################################################################
# PCA plots
pca_plots = [('Dow Jones Index', X_dow_jones),
             ('Facebook Live', X_live),
             ('Sales Transactions', X_sales),
             ('Water Treatment', X_water)]

plot_pca = False
if plot_pca:
    for dataset_name, dataset in pca_plots:
        pca_var_plotbar(dataset, title=dataset_name + ' PCA bar plot')
        pca_ratio_plotcurve(dataset, title=dataset_name + ' PCA curve plot')


# ######################################################################################################################
km_df = pd.DataFrame(columns=('Dataset', 'Time', 'SSE', 'CSM'))
db_df = pd.DataFrame(columns=('Dataset', 'Time', 'SSE', 'CSM'))
ac_df = pd.DataFrame(columns=('Dataset', 'Time', 'SSE', 'CSM'))

# Base params for all datasets
default_base = {'pca_components': 5,
                'n_cluster': 2,
                'noise_rate': 0,
                'eps': 0.3,
                'min_samples': 8,
                'linkage': 'ward',
                'random_state': 0}
# Optimal params for datasets
datasets = [(X_dow_jones, {'dataset_name': 'Dow Jones Index',
                           'pca_components': 3,
                           'km_cluster': 2,
                           'ac_cluster': 2,
                           'eps': .2,
                           'min_samples': 6,
                           'linkage': 'ward'}),
            (X_live, {'dataset_name': 'Facebook Live',
                      'pca_components': 3,
                      'km_cluster': 3,
                      'eps': .07,
                      'min_samples': 9,
                      'ac_cluster': 3,
                      'linkage': 'ward'}),
            (X_sales, {'dataset_name': 'Sales Transactions',
                       'pca_components': 5,
                       'km_cluster': 2,
                       'eps': .4,
                       'min_samples': 20,
                       'ac_cluster': 2,
                       'linkage': 'ward'}),
            (X_water, {'dataset_name': 'Water Treatment',
                       'pca_components': 2,
                       'km_cluster': 2,
                       'eps': .17,
                       'min_samples': 4,
                       'ac_cluster': 2,
                       'linkage': 'complete'})]

# ######################################################################################################################
# overall plots
# plot csm and scatter
plt.figure(figsize=(12, 21))
plt.subplots_adjust(left=.05, right=.95, bottom=.001, top=.96, wspace=.3,
                    hspace=.3)
plot_num = 1

for i_dataset, (dataset, optimal_params) in enumerate(datasets):
    params = default_base.copy()
    params.update(optimal_params)
    # Feature selection
    pca = PCA(n_components=params['pca_components'])
    X = pca.fit_transform(dataset)

    # Create cluster objects
    km = KMeans(n_clusters=params['km_cluster'], random_state=params['random_state'])
    db = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    ac = AgglomerativeClustering(n_clusters=params['ac_cluster'], linkage=params['linkage'])
    clustering_algorithms = (('KMeans', km), ('DBSCAN', db), ('AgglomerativeClustering', ac))

    for name, algorithm in clustering_algorithms:
        algorithm.fit(X)
        # Predict samples clusters
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)
        # Metrics to rate clustering
        sample_silhouette_values = silhouette_samples(X, y_pred)
        running_time = time_counting(X, algorithm)
        sse = calculate_sse(X, y_pred)
        csm = silhouette_score(X, y_pred)
        n_samples = len(y_pred) - (1 if -1 in y_pred else 0)
        n_cluster = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        noise_ratio = len([noise for noise in y_pred if noise == -1]) / len(y_pred)
        rate = clustering_score(running_time, sse, csm, n_samples, noise_ratio)
        # Save results into DataFrame
        if name.startswith('K'):
            km_df = km_df.append({'Dataset': params['dataset_name'] + ' (' + str(params['km_cluster']) + ')',
                                  'Time': running_time,
                                  'SSE': sse,
                                  'CSM': csm},
                                 ignore_index=True)
        if name.startswith('DB'):
            db_df = db_df.append({'Dataset': params['dataset_name'],
                                  'Time': running_time,
                                  'SSE': sse,
                                  'CSM': csm},
                                 ignore_index=True)
        if name.startswith('Ag'):
            ac_df = ac_df.append({'Dataset': params['dataset_name'] + ' (' + params['linkage'] + ')',
                                  'Time': running_time,
                                  'SSE': sse,
                                  'CSM': csm},
                                 ignore_index=True)

        # ### Plot csm and scatter
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        # Plot CSM
        plt.subplot(8, len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=16)

        ax1 = plt.gca()
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])
        y_lower = 10
        for k in range(n_cluster):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[y_pred == k]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=colors[k], edgecolor=colors[k], alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        # The vertical line for average silhouette score of all the values
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel(params['dataset_name'] + "\nCluster label")

        ax1.axvline(x=csm, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the y axis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.text(.99, .15, ('SSE: %.2f' % sse),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .08, ('CSM: %.2f' % csm),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .01, ('Score: %.2f' % rate),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')

        plt.subplot(8, len(clustering_algorithms), plot_num + len(clustering_algorithms))

        ax2 = plt.gca()
        ax2.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        # Calculate centroids
        centers = []
        df = pd.DataFrame(X)
        df['y'] = y_pred
        for c in range(n_cluster):
            cluster = df.loc[df['y'] == c]
            centers.append(cluster.mean()[:-1].values)
        centers = np.asarray(centers)

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for center_k, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % center_k, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Clustered data %.3fs" % running_time)
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plot_num += 1
    plot_num += len(clustering_algorithms)
plt.show()

# Print results
print('Best K Means results:\n', km_df)
print('Best DBSCAN results:\n', db_df)
print('Best Agglomerative results:\n', ac_df)