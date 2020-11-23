import datetime

from sources.Configurator import get_configuration, get_dataset_configs, get_input_parameters_for_algorithms
from sources.DataIntegrator import read_in_data, feature_selection, preprocess_dataset, prepare_for_unsupervised_learning, prepare_for_supervised_learning
from sources.DataPreprocessor import further_preprocessing, clean_dataset, scale_and_normalize_features
from sources.HardwareReader import get_hardware_specs
from sources.ExperimentRunner import run_experiments
from sources.ResultsManager import export_results


supported_unsupervised_algorithms = ["kmeans", "spectral", "dbscan", "optics", "meanshift", "agglomerative", "affinity", "em", "vbgmm"]
supported_supervised_algorithms = ["knn", "svc", "nearest_centroid", "radius_neighbors", "nca", "svc_sdg"]

config = get_configuration(supported_unsupervised_algorithms, supported_supervised_algorithms)
dataset_configs = get_dataset_configs(config["parameters"]["supervised_algorithms"])
unsupervised_input_parameters, supervised_input_parameters = get_input_parameters_for_algorithms()

hardware_specs = get_hardware_specs()
timestamp = datetime.datetime.now()

for i, dataset_config in dataset_configs.iterrows():
    print("Testing dataset " + str(i + 1) + "/" + str(dataset_configs.shape[0]) + ": \"" + dataset_config["dataset"] + "\"")

    dataset_initial = read_in_data(config["datasets"]["directory_path"] + dataset_config["dataset"], dataset_config["csv_delimiter"])

    dataset_initial = feature_selection(dataset_initial, dataset_config["features"], dataset_config["feature_selection_type"])
    dataset = preprocess_dataset(dataset_initial, dataset_config["numeric_categorials"])

    dataset = further_preprocessing(dataset_initial, config["datasets"]["missing_values"])
    dataset = clean_dataset(dataset)

    dataset_unsupervised = prepare_for_unsupervised_learning(dataset, dataset_config["numeric_categorials"], dataset_config["class"])
    dataset_supervised = prepare_for_supervised_learning(dataset, dataset_config["numeric_categorials"], dataset_config["class"])

    dataset_supervised = scale_and_normalize_features(dataset_supervised, config["datasets"]["feature_scaling_and_normalization"], dataset_config["class"], True)  # TODO: "True"

    results_unsupervised, results_supervised = run_experiments(dataset_unsupervised, dataset_supervised, config["parameters"]["n_runs"],
                                                               config["parameters"]["unsupervised_algorithms"], config["parameters"]["supervised_algorithms"],
                                                               unsupervised_input_parameters, supervised_input_parameters,
                                                               dataset_config["class"], config["parameters"]["test_size"],
                                                               config["parameters"]["show_unsupervised_clusterings"], timestamp,
                                                               dataset_config["dataset"])

    export_results(dataset_config, results_unsupervised, results_supervised, hardware_specs, config["results"]["directory_path"],
                   timestamp, unsupervised_input_parameters, supervised_input_parameters, dataset.shape)

    print()
