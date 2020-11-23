import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

import pandas as pd


def standard_scale_and_normalize(dataset, feature):
    dataset_transformed = dataset.copy()

    scaler = StandardScaler()

    feature_array = np.array(dataset_transformed[feature])
    feature_array_reshaped = np.reshape(feature_array, (len(feature_array), 1))

    feature_array_reshaped = scaler.fit_transform(feature_array_reshaped)
    feature_array_original_shape = np.reshape(feature_array_reshaped, (1, len(feature_array_reshaped)))[0]
    dataset_transformed[feature] = pd.Series(feature_array_original_shape)

    return dataset_transformed


def quantile_scale_and_normalize(dataset, feature):
    dataset_transformed = dataset.copy()

    n_quantiles = dataset.shape[0] // 10

    scaler = QuantileTransformer(n_quantiles=n_quantiles)

    feature_array = np.array(dataset_transformed[feature])
    feature_array_reshaped = np.reshape(feature_array, (len(feature_array), 1))

    feature_array_reshaped = scaler.fit_transform(feature_array_reshaped)
    feature_array_original_shape = np.reshape(feature_array_reshaped, (1, len(feature_array_reshaped)))[0]
    dataset_transformed[feature] = pd.Series(feature_array_original_shape)

    return dataset_transformed


def scale_and_normalize_features(dataset, parameter, class_column, supervised):
    dataset_transformed = dataset.copy()

    features_to_normalize = (set(dataset.columns) - set([class_column])) if supervised else set(dataset.columns)
    for feature in features_to_normalize:
        if parameter == "standard":
            dataset_transformed = standard_scale_and_normalize(dataset_transformed, feature)
        else:
            dataset_transformed = quantile_scale_and_normalize(dataset_transformed, feature)

    return dataset_transformed


def clean_dataset(dataset):
    dataset_cleaned = dataset.copy()

    # remove duplicates:
    dataset_cleaned = dataset_cleaned.drop_duplicates()
    print("[Dataset Cleaning] Removed " + str(dataset.shape[0] - dataset_cleaned.shape[0]) + " row(s) being not distinct.")

    # TODO: more?

    return dataset_cleaned


def handle_missing_values(dataset, config_parameter):
    dataset_modified = dataset.copy()

    if config_parameter == "cca":
        dataset_modified = dataset_modified.dropna()
    elif config_parameter == "aca":
        dataset_modified = dataset_modified.dropna(axis='columns')
    elif config_parameter == "impute":
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataset_modified = imputer.fit_transform(dataset_modified)
        dataset_modified = pd.DataFrame(dataset_modified, columns=dataset.columns)

    assert not dataset_modified.empty, "missing value handler dropped all values of the dataset! maybe try a different " \
                                       "handling method regarding missing values"

    return dataset_modified


def further_preprocessing(dataframe, missing_value_parameter):
    # handle missing values (CCA or imputation):
    dataframe = handle_missing_values(dataframe, missing_value_parameter)

    return dataframe