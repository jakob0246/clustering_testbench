from UnsupervisedTester import kmeans_clustering, em_clustering, spectral_clustering, dbscan_clustering, \
                               optics_clustering, meanshift_clustering, agglomerative_clustering, affinity_clustering, vbgmm_clustering

from SupervisedTester import knn_clustering, svc_clustering, nearest_centroid_clustering, radius_neighbors_clustering, nca_clustering, svc_sdg_clustering

from sklearn import metrics

from sklearn.model_selection import train_test_split

import plotly.graph_objects as go

import numpy as np

import os

import memory_profiler


def save_plot_2D(dataset, result_labels, algorithm, silhouette_score, run, timestamp, dataset_name):
    colors = ["yellow", "pink", "blue", "green", "red", "orange", "black", "lightblue", "lightskyblue", "mistyrose",
              "mediumslateblue", "mediumspringgreen", "moccasin"]

    figure = go.Figure()

    for i, cluster in enumerate(set(result_labels)):
        cluster_points = dataset.loc[result_labels == cluster]

        figure.add_trace(go.Scatter(x=np.array(cluster_points)[..., 0], y=np.array(cluster_points)[..., 1], mode="markers", marker_color=colors[i % len(colors)]))

    figure.update_xaxes(title_text=dataset.columns[0])
    figure.update_yaxes(title_text=dataset.columns[1])

    figure.update_layout(height=600, width=1000, title_text="dataset: \"" + dataset_name + "\", algorithm: \"" + algorithm + "\", run: " + str(run + 1) + " (" + str(timestamp) + ")")  # showlegend=False

    save_directory = "../../results/plots/unsupervised/" + str(timestamp).replace(" ", "_")

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    dataset_name_save = dataset_name.replace(".csv", "")
    dataset_name_save = dataset_name_save.replace(".arff", "")
    dataset_name_save = dataset_name_save.replace(".txt", "")
    dataset_name_save = dataset_name_save.replace(".", "")
    figure.write_image(save_directory + "/" + dataset_name_save + "_" + algorithm + "_" + str(run + 1) + "_2D.jpeg", scale=3.0)
    figure.write_image(save_directory + "/" + dataset_name_save + "_" + algorithm + "_" + str(run + 1) + "_2D.pdf",
                       scale=3.0)


def save_plot_3D(dataset, result_labels, algorithm, silhouette_score, run, timestamp, dataset_name):
    colors = ["yellow", "pink", "blue", "green", "red", "orange", "black", "lightblue", "lightskyblue", "mistyrose",
              "mediumslateblue", "mediumspringgreen", "moccasin"]

    traces = []
    for i, cluster in enumerate(set(result_labels)):
        cluster_points = dataset.loc[result_labels == cluster]

        traces.append(go.Scatter3d(x=np.array(cluster_points)[..., 0], y=np.array(cluster_points)[..., 1],
                                   z=np.array(cluster_points)[..., 2], mode='markers', name="cluster " + str(cluster),
                                   marker=dict(color=colors[cluster % len(colors)])))

    layout = go.Layout(title="dataset: \"" + dataset_name + "\", algorithm: \"" + algorithm + "\", run: " + str(run + 1) + " (" + str(timestamp) + ")")
    figure = go.Figure(data=traces, layout=layout)

    figure.update_layout(scene=dict(xaxis_title=dataset.columns[0], yaxis_title=dataset.columns[1],
                                    zaxis_title=dataset.columns[2]))

    save_directory = "../../results/plots/unsupervised/" + str(timestamp).replace(" ", "_")

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    dataset_name_save = dataset_name.replace(".csv", "")
    dataset_name_save = dataset_name_save.replace(".arff", "")
    dataset_name_save = dataset_name_save.replace(".txt", "")
    dataset_name_save = dataset_name_save.replace(".", "")
    figure.write_image(save_directory + "/" + dataset_name_save + "_" + algorithm + "_" + str(run + 1) + "_3D.jpeg", scale=6.0)


def standardize_silhouette_score(silhouette_score):
    silhouette_score_standardized = 0.5 * (silhouette_score + 1)
    return silhouette_score_standardized


def run_unsupervised_experiment(dataset, algorithm, parameters, show_unsupervised_clusterings, run, timestamp, dataset_name):
    print("\tRunning " + algorithm + " ...")

    n_clusters = None
    if algorithm == "kmeans":
        result_labels, time = kmeans_clustering(dataset, parameters["kmeans"])
        memory = memory_profiler.memory_usage((kmeans_clustering, (dataset, parameters["kmeans"], )))

    elif algorithm == "em":
        result_labels, time = em_clustering(dataset, parameters["em"])
        memory = memory_profiler.memory_usage((em_clustering, (dataset, parameters["em"], )))

    elif algorithm == "spectral":
        result_labels, time = spectral_clustering(dataset, parameters["spectral"])
        memory = memory_profiler.memory_usage((spectral_clustering, (dataset, parameters["spectral"], )))

    elif algorithm == "dbscan":
        result_labels, time, n_clusters = dbscan_clustering(dataset, parameters["dbscan"])
        memory = memory_profiler.memory_usage((dbscan_clustering, (dataset, parameters["dbscan"], )))

    elif algorithm == "optics":
        result_labels, time, n_clusters = optics_clustering(dataset, parameters["optics"])
        memory = memory_profiler.memory_usage((optics_clustering, (dataset, parameters["optics"], )))

    elif algorithm == "meanshift":
        result_labels, time, n_clusters = meanshift_clustering(dataset, parameters["meanshift"])
        memory = memory_profiler.memory_usage((meanshift_clustering, (dataset, parameters["meanshift"], )))

    elif algorithm == "agglomerative":
        result_labels, time = agglomerative_clustering(dataset, parameters["agglomerative"])
        memory = memory_profiler.memory_usage((agglomerative_clustering, (dataset, parameters["agglomerative"], )))

    elif algorithm == "affinity":
        result_labels, time, n_clusters = affinity_clustering(dataset, parameters["affinity"])
        memory = memory_profiler.memory_usage((affinity_clustering, (dataset, parameters["affinity"],)))

    elif algorithm == "vbgmm":
        result_labels, time, n_clusters = vbgmm_clustering(dataset, parameters["vbgmm"])
        memory = memory_profiler.memory_usage((vbgmm_clustering, (dataset, parameters["vbgmm"],)))

    else:
        raise Exception("[Data Clusterer] clustering algorithm \"" + algorithm + "\" not supported or spelled incorrectly!")

    silhouette_score = metrics.silhouette_score(dataset, result_labels, metric='euclidean')
    silhouette_score_standardized = standardize_silhouette_score(silhouette_score)

    if show_unsupervised_clusterings:
        if dataset.shape[1] == 2:
            save_plot_2D(dataset, result_labels, algorithm, silhouette_score, run, timestamp, dataset_name)
        elif dataset.shape[1] == 3:
            save_plot_3D(dataset, result_labels, algorithm, silhouette_score, run, timestamp, dataset_name)

    result = {"algorithm": algorithm, "cputime": time, "n_clusters": n_clusters, "silhouette_score_standardized": silhouette_score_standardized, "accuracy": None, "memory": max(memory)}

    return result


def run_supervised_experiment(X_train, X_test, y_train, y_test, algorithm, parameters):
    print("\tRunning " + algorithm + " ...")

    if algorithm == "knn":
        accuracy, time_train, time_test = knn_clustering(X_train, X_test, y_train, y_test, parameters["knn"])
        memory = memory_profiler.memory_usage((knn_clustering, (X_train, X_test, y_train, y_test, parameters["knn"], )))

    elif algorithm == "svc":
        accuracy, time_train, time_test = svc_clustering(X_train, X_test, y_train, y_test, parameters["svc"])
        memory = memory_profiler.memory_usage((svc_clustering, (X_train, X_test, y_train, y_test, parameters["svc"],)))

    elif algorithm == "svc_sgd":
        accuracy, time_train, time_test = svc_sdg_clustering(X_train, X_test, y_train, y_test, parameters["svc_sgd"])
        memory = memory_profiler.memory_usage((svc_sdg_clustering, (X_train, X_test, y_train, y_test, parameters["svc_sgd"],)))

    elif algorithm == "nearest_centroid":
        accuracy, time_train, time_test = nearest_centroid_clustering(X_train, X_test, y_train, y_test, parameters["nearest_centroid"])
        memory = memory_profiler.memory_usage((nearest_centroid_clustering, (X_train, X_test, y_train, y_test, parameters["nearest_centroid"],)))

    elif algorithm == "radius_neighbors":
        accuracy, time_train, time_test = radius_neighbors_clustering(X_train, X_test, y_train, y_test, parameters["radius_neighbors"])
        memory = memory_profiler.memory_usage((radius_neighbors_clustering, (X_train, X_test, y_train, y_test, parameters["radius_neighbors"],)))

    elif algorithm == "nca":
        accuracy, time_train, time_test = nca_clustering(X_train, X_test, y_train, y_test, parameters["nca"])
        memory = memory_profiler.memory_usage((nca_clustering, (X_train, X_test, y_train, y_test, parameters["nca"],)))

    else:
        raise Exception("[Data Clusterer] clustering algorithm \"" + algorithm + "\" not supported or spelled incorrectly!")

    result = {"algorithm": algorithm, "cputime_train": time_train, "cputime_test": time_test, "cputime_total": time_train + time_test, "accuracy": accuracy, "TODO_score": None, "memory": max(memory)}

    return result


def run_experiments(dataset_unsupervised, dataset_supervised, n_runs, unsupervised_algorithms, supervised_algorithms,
                    unsupervised_input_parameters, supervised_input_parameters, class_column, test_size, show_unsupervised_clusterings,
                    timestamp, dataset_name):
    results_unsupervised = []
    for algorithm in unsupervised_algorithms:
        for run in range(n_runs):
            try:
                result = run_unsupervised_experiment(dataset_unsupervised, algorithm, unsupervised_input_parameters,
                                                     show_unsupervised_clusterings, run, timestamp, dataset_name)
                result["run"] = run

                results_unsupervised.append(result)
            except ValueError as error:
                print("\t-Error- Exception happend while testing with algorithm \"" + algorithm + "\": \"" + str(error) + "\"")

    results_supervised = []
    if class_column != "" and class_column in dataset_supervised.columns:
        print()

        X = dataset_supervised.drop(columns=[class_column], axis=1)
        y = dataset_supervised[class_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        for algorithm in supervised_algorithms:
            for run in range(n_runs):
                result = run_supervised_experiment(X_train, X_test, y_train, y_test, algorithm, supervised_input_parameters)
                result["run"] = run

                results_supervised.append(result)

    return results_unsupervised, results_supervised
