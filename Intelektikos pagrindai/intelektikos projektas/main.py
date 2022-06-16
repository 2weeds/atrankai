import copy

import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer




def convert_categorical_to_continuous_data(df):
    for column_name in df:
        if type(df[column_name][0]) == str:
            df[column_name] = pd.Categorical(df[column_name])
            df[column_name] = df[column_name].cat.codes
    return df

def read_data(filename:str):
    init_data = pd.read_excel(filename)
    df = copy.deepcopy(init_data)
    df = convert_categorical_to_continuous_data(df=df)
    y = df.pop('Company size').to_numpy().astype(int)
    X = df.to_numpy()
    return X, y, init_data


def plot_minisom_results(som, training_data, labels, init_data):
    plt.bone()
    plt.pcolor(som.distance_map().T)
    plt.colorbar()
    # markers = ['o', 's', 'D', 'p', 'w']
    # colors = ['r', 'g', 'b', 'y', 'o']
    for i, x in enumerate(training_data):
        w = som.winner(x)
        plt.plot(w[0] + 0.5,
                 w[1] + 0.5,
                 # markers[labels[i]],
                 # markeredgecolor=colors[labels[i]],
                 # markerfacecolor='None',
                 # markersize=10,
                 # markeredgewidth=2
                 )
    plt.show()

def execute_minisom(som_dimensions, training_data, labels, init_data):
    som = MiniSom(x=som_dimensions[0], y=som_dimensions[1], input_len=training_data.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(training_data)
    som.train(data=training_data, num_iteration=1000)
    plot_minisom_results(som=som, training_data=training_data, labels=labels, init_data=init_data)

def plot_KMeans_results(n_clusters, dataset, y_km, clusters):
    for i in range(0,n_clusters):
        plt.scatter(dataset[y_km == i, 0], dataset[y_km == i, 1], s=50)
        plt.scatter(clusters[i][0], clusters[i, 1], marker='*', s=100, color='black')
    plt.show()

def plot_Elbow(dataset):
    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(dataset)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def plot_Silhoutte(dataset):
    for i in [2, 3, 4, 5]:
        '''
        Create KMeans instance for different number of clusters
        '''
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
        visualizer.fit(dataset)
        plt.show()

def execute_KMeans(dataset, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    points = dataset[:, [4, 5]]
    km.fit(points)
    plt.scatter(dataset[:, 4], dataset[:, 5])
    plt.show()

    clusters = km.cluster_centers_
    y_km = km.fit_predict(points)
    plot_KMeans_results(n_clusters,points,y_km,clusters)
    plot_Elbow(points)
    score = silhouette_score(points, km.labels_, metric='euclidean')
    print('Silhouetter Score: %.3f' % score)
    plot_Silhoutte(points)

if __name__ == '__main__':
    """
    Reading data from excel file
    """
    X, y, init_data = read_data(filename=r'IT Salary Survey EU  2020.xls')
    som_dimensions = [10, 10]

    # execute_minisom(som_dimensions=som_dimensions, training_data=X, labels=y, init_data=init_data)
    execute_KMeans(X,5)