import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def show_clusters(data, columns, n_centroids=2):
    extracted_data = data[columns].to_numpy()

    km = KMeans(n_clusters=n_centroids)
    y_km = km.fit_predict(extracted_data)

    colors = ['lightgreen', 'orange', 'lightblue', 'red']
    markers = ['s', 'o', 'v', '.']

    # Plot all clusters
    for i in range(n_centroids):
        plt.scatter(
            extracted_data[y_km == i, 0], extracted_data[y_km == i, 1],
            s=50, c=colors[i],
            marker=markers[i], edgecolor='black',
            label=f"cluster {i + 1}"
        )

    # Plot centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.show()


df = pd.read_csv('cars.csv')
show_clusters(df, ['cubicinches', 'weightlbs'], n_centroids=4)
