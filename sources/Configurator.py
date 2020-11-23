import configparser

import pandas as pd

import math

import json


def get_input_parameters_for_algorithms(path="../configs/algorithm_parameters.json"):
    file = open(path)
    json_data = json.load(file)

    return json_data["unsupervised"], json_data["supervised"]


def get_dataset_configs(supervised_algorithms):
    dataset_configs = pd.read_csv("../configs/dataset_configs.csv", delimiter=";", escapechar="\\")

    dataset_configs = dataset_configs.replace({math.nan: ""})

    # TODO: integrity checks + conversions
    for i, row in dataset_configs.iterrows():
        assert row["feature_selection_type"] in ["include", "exclude"]

        row["class"] = row["class"].lower()

        # if supervised_algorithms != "":
        #     assert row["class"] != "", "-Error- If supervised algorithms should be tested for dataset \"" + row["dataset"] + "\", then \"class\" must contain a value in dataset_configs.csv"

    return dataset_configs


def parse_config(path: str = "../configs/config.txt") -> dict:
    parser_config = configparser.ConfigParser()
    parser_config.read(path)

    # DONE: verify config
    # TODO: exception handling
    assumed_metastructure = {
        "datasets": ["directory_path", "feature_scaling_and_normalization", "missing_values"],
        "parameters": ["unsupervised_algorithms", "supervised_algorithms", "test_size", "n_runs", "show_unsupervised_clusterings"],
        "results": ["directory_path"]
    }

    assert list(assumed_metastructure.keys()).sort() == list(parser_config.sections()).sort()

    config = {}
    for key_outer in assumed_metastructure.keys():
        assert list(assumed_metastructure[key_outer]).sort() == list(parser_config[key_outer]).sort()

        config[key_outer] = {}
        for key_inner in assumed_metastructure[key_outer]:
            config[key_outer][key_inner] = str(parser_config[key_outer][key_inner])

    return config


def get_configuration(supported_unsupervised_algorithms, supported_supervised_algorithms) -> dict:
    # TODO: check if algorithms valid

    raw_config_dict = parse_config()

    config_dict = raw_config_dict.copy()

    # TODO: preprocess raw config parameters:
    for key_outer in config_dict.keys():
        for key_inner in config_dict[key_outer].keys():
            config_dict[key_outer][key_inner] = config_dict[key_outer][key_inner].lower().strip()

    # TODO parse raw config: feature-extraction, type-conversions etc.:
    config_dict["parameters"]["show_unsupervised_clusterings"] = config_dict["parameters"]["show_unsupervised_clusterings"] == "true"

    config_dict["parameters"]["n_runs"] = int(config_dict["parameters"]["n_runs"])
    config_dict["parameters"]["test_size"] = float(config_dict["parameters"]["test_size"])

    config_dict["parameters"]["unsupervised_algorithms"] = list(map(lambda ele: ele.strip(), config_dict["parameters"]["unsupervised_algorithms"].split(",")))
    config_dict["parameters"]["supervised_algorithms"] = list(map(lambda ele: ele.strip(), config_dict["parameters"]["supervised_algorithms"].split(",")))

    if config_dict["parameters"]["unsupervised_algorithms"] == [""]:
        config_dict["parameters"]["unsupervised_algorithms"] = []
    if config_dict["parameters"]["supervised_algorithms"] == [""]:
        config_dict["parameters"]["supervised_algorithms"] = []

    # TODO: check integrity of user config attributes
    assert config_dict["datasets"]["feature_scaling_and_normalization"] in ["standard", "quantile"]

    config_dict = raw_config_dict

    return config_dict
