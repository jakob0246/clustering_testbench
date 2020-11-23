from scipy.io import arff

import numpy as np
import pandas as pd

import re

from sklearn.preprocessing import OneHotEncoder


def check_for_ohe(columns):
    # determine that <ohe>_ is not at the beginning of input columns
    for column in columns:
        pattern = re.compile("<ohe>_*")
        assert not pattern.match(column), "column names of the dataset shouldn't start with <ohe>_"


def read_in_data_csv(path, delimiter):
    dataframe = pd.read_csv(path, delimiter=delimiter)

    check_for_ohe(dataframe.columns)

    for column in dataframe.columns:
        if dataframe[column].dtype != "int64" and dataframe[column].dtype != "float64":
            dataframe[column] = dataframe[column].replace({"?": np.nan})

    for column in dataframe.columns:
        column_type = column.split(":", 1)[1].lower().strip()
        if column_type == "categorial" or column_type == "categorical":  # :D
            dataframe[column] = dataframe[column].astype("category")
        elif column_type == "number" or column_type == "numerical":  # :D
            dataframe[column] = dataframe[column].astype("float64")
        else:
            raise Exception("Column type should be \"number\" or \"categorial\"! -> see config")

        new_column_name = column.split(":", 1)[0].lower().strip()
        dataframe.rename(columns={column: new_column_name}, inplace=True)

    return dataframe


def read_in_data_arff(path):
    dataset, meta = arff.loadarff(open(path))

    dataframe = pd.DataFrame(dataset)

    check_for_ohe(dataframe.columns)

    for column in dataframe.columns:
        if dataframe[column].dtype != "int64" and dataframe[column].dtype != "float64":
            dataframe[column] = dataframe[column].replace({b"?": np.nan, "?": np.nan})
            dataframe[column] = dataframe[column].str.decode("utf-8")

    return dataframe


def read_in_data(path, csv_delimiter):
    if re.search("\.arff", path):
        dataframe = read_in_data_arff(path)
    elif re.search("\.csv", path):
        dataframe = read_in_data_csv(path, csv_delimiter)
    else:
        raise Exception("File should be .csv or .arff")

    return dataframe


def feature_selection(dataset, config_features, config_type):
    features = list(map(lambda ele: ele.strip(), config_features.split("-")))

    if config_type == "exclude":
        return dataset[set(dataset.columns) - set(features)]

    return dataset[features]


def preprocess_dataset(dataframe, config_numeric_categorials):
    numeric_categorials = list(map(lambda ele: ele.strip(), config_numeric_categorials.split("-")))

    # replace missing values (if columns aren't all numeric dtypes -> would return error)
    if len(dataframe.select_dtypes([np.object]).columns) != 0:
        dataframe = dataframe.replace({b"?": np.nan})

    # convert binary-strings to strings:
    for column in dataframe.select_dtypes([np.object]).columns:
        dataframe[column] = dataframe[column].str.decode("utf-8")

    # convert objects to categorials:
    for column in dataframe.select_dtypes([np.object]).columns:
        dataframe[column] = dataframe[column].astype("category")

    # convert numeric categorials to floats:
    for column in numeric_categorials:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("float64")

    # handle missing values (CCA or imputation):
    dataframe = dataframe.dropna()  # CCA

    return dataframe


def prepare_for_unsupervised_learning(dataframe, config_numeric_categorials, class_column):
    numeric_categorials = list(map(lambda ele: ele.strip(), config_numeric_categorials.split("-")))

    # remove class-column:
    if class_column != "" and class_column in dataframe.columns:
        dataframe = dataframe.drop(columns=[class_column])

    # do one-hot-encoding:
    columns_to_be_encoded = list(set(dataframe.select_dtypes(include=['category', 'object']).columns) - set(numeric_categorials))
    selection_to_be_encoded = dataframe[columns_to_be_encoded]

    encoder = OneHotEncoder()
    selection_transformed = encoder.fit_transform(selection_to_be_encoded).toarray()
    dataframe_selection_transformed = pd.DataFrame(selection_transformed, columns=list(map(lambda ele: "<ohe>_" + ele, encoder.get_feature_names(columns_to_be_encoded))))

    dataframe = dataframe.reset_index(drop=True)
    dataframe_selection_transformed = dataframe_selection_transformed.reset_index(drop=True)
    dataframe = dataframe.join(dataframe_selection_transformed)

    dataframe = dataframe.drop(columns=columns_to_be_encoded)

    return dataframe


def prepare_for_supervised_learning(dataframe, config_numeric_categorials, class_column):
    numeric_categorials = list(map(lambda ele: ele.strip(), config_numeric_categorials.split("-")))

    # do one-hot-encoding:
    columns_to_be_encoded = list(set(dataframe.select_dtypes(include=['category', 'object']).columns) - set(numeric_categorials) - {class_column})
    selection_to_be_encoded = dataframe[columns_to_be_encoded]

    encoder = OneHotEncoder()
    selection_transformed = encoder.fit_transform(selection_to_be_encoded).toarray()
    dataframe_selection_transformed = pd.DataFrame(selection_transformed, columns=list(map(lambda ele: "<ohe>_" + ele, encoder.get_feature_names(columns_to_be_encoded))))

    dataframe = dataframe.reset_index(drop=True)
    dataframe_selection_transformed = dataframe_selection_transformed.reset_index(drop=True)
    dataframe = dataframe.join(dataframe_selection_transformed)

    dataframe = dataframe.drop(columns=columns_to_be_encoded)

    return dataframe