## Installation
Run project with PyCharm and Python 3.6. For that, create a new PyCharm project for the _clustering\_testbench_ repository. Then install the required libraries, as well, using the _requirements.txt_.

## Usage
Execute _Main.py_ if you want to test specific clustering algorithms on different datasets. The usage of the program also is determined by the utilization of various configuration files.

#### config.txt
  -  main configuration parameters

#### dataset_configs.csv
  -  every row here is a test experiment configuration for one dataset
  -  numeric_categorials are categorials which should be treated as a number, e.g. {1, 2, 3, 4}
  -  csv_delimiter is only important if a csv file should be used

#### algorithm_parameters.json
  -  specifies the parameters for the learning algorithms to use;
  -> syntax: 
       -  {"unsupervised": { <algorithm_1> ... <algorithm_m>}, "supervised": { <algorithm_1> ... <algorithm_n> }}
       -  where <algorithm[...]> contains the parameters as an object, e.g. "kmeans": { <parameter_1> ... <parameter_o> }
  -  "distance" represents the distance function and can be "euclidean", "manhattan", "minkowski", "cosine" or "mahalanobis"
  -> knn and nca don't support "cosine"
    
## Datasets
CSV & ARFF files are supported. Use the following formatting for the header in the case of CSVs per column: _"\<column-name\> \<data-type\>"_. Where the data type can be _"numerical"_ or _"categorical"_.
  
Exemplary datasets are obtained from the UCI machine learning repository.
###### Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
