import pandas as pd


def export_results(dataset_config, results_unsupervised, results_supervised, hw_specs, dir_path, timestamp, parameters_unsupervised, parameters_supervised, shape):
    timestamp_string = "{:02d}".format(timestamp.hour) + ":" + "{:02d}".format(timestamp.minute) + ":" + "{:02d}".format(timestamp.second) + \
                       ", " + "{:02d}".format(timestamp.day) + "." + "{:02d}".format(timestamp.month) + "." + "{:04d}".format(timestamp.year)

    results_dataframe_unsupervised = pd.DataFrame(columns=["timestamp of execution", "dataset",	"algorithm", "run", "parameters",
                                                           "features", "include/exclude", "shape", "cpu time", "memory", "accuracy", "silhouette score standardized", "hardware"])
    for i, result in enumerate(results_unsupervised):
        results_dataframe_unsupervised_tmp = pd.DataFrame([[timestamp_string, dataset_config["dataset"], result["algorithm"], result["run"] + 1,
                                               str(parameters_unsupervised[result["algorithm"]]), dataset_config["features"], dataset_config["feature_selection_type"], shape,
                                               result["cputime"], result["memory"], result["accuracy"], result["silhouette_score_standardized"], str(hw_specs)]], columns=["timestamp of execution", "dataset", "algorithm", "run", "parameters",
                                               "features", "include/exclude", "shape", "cpu time", "memory", "accuracy", "silhouette score standardized", "hardware"])
        results_dataframe_unsupervised = results_dataframe_unsupervised.append(results_dataframe_unsupervised_tmp)

    results_dataframe_supervised = pd.DataFrame(columns=["timestamp of execution", "dataset", "algorithm", "run", "parameters",
                                                         "features", "include/exclude", "shape", "cpu time train", "cpu time test", "cpu time total", "memory", "accuracy", "mean_squared_error", "hardware"])
    for i, result in enumerate(results_supervised):
        results_dataframe_supervised_tmp = pd.DataFrame([[timestamp_string, dataset_config["dataset"], result["algorithm"], result["run"] + 1,
                                               str(parameters_supervised[result["algorithm"]]), dataset_config["features"], dataset_config["feature_selection_type"], shape,
                                               result["cputime_train"], result["cputime_test"], result["cputime_train"] + result["cputime_test"], result["memory"], result["accuracy"], result["TODO_score"], str(hw_specs)]], columns=["timestamp of execution", "dataset", "algorithm", "run", "parameters",
                                               "features", "include/exclude", "shape", "cpu time train", "cpu time test", "cpu time total", "memory", "accuracy", "mean_squared_error", "hardware"])
        results_dataframe_supervised = results_dataframe_supervised.append(results_dataframe_supervised_tmp)

    excel_writer_unsupervised = pd.ExcelWriter(dir_path + 'unsupervised_results.xlsx', mode='a', engine='openpyxl')
    excel_writer_unsupervised.sheets = dict((ws.title, ws) for ws in excel_writer_unsupervised.book.worksheets)
    results_dataframe_unsupervised.to_excel(excel_writer_unsupervised, sheet_name="Sheet1", startrow=excel_writer_unsupervised.sheets['Sheet1'].max_row, index=False, header=False)
    excel_writer_unsupervised.save()

    excel_writer_supervised = pd.ExcelWriter(dir_path + 'supervised_results.xlsx', mode='a', engine='openpyxl')
    excel_writer_supervised.sheets = dict((ws.title, ws) for ws in excel_writer_supervised.book.worksheets)
    results_dataframe_supervised.to_excel(excel_writer_supervised, sheet_name="Sheet1",  startrow=excel_writer_supervised.sheets['Sheet1'].max_row, index=False, header=False)
    excel_writer_supervised.save()
