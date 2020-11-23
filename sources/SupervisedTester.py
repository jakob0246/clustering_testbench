from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

import time


def knn_clustering(X_train, X_test, y_train, y_test, parameters):
    initial_classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=parameters["k"], metric=parameters["distance"])

    cputime_start_train = time.process_time()
    classifier = initial_classifier.fit(X_train, y_train)
    cputime_end_train = time.process_time()

    cputime_start_test = time.process_time()
    y_pred = classifier.predict(X_test)
    cputime_end_test = time.process_time()

    accuracy = classifier.score(X_test, y_test)

    return accuracy, cputime_end_train - cputime_start_train, cputime_end_test - cputime_start_test


def svc_clustering(X_train, X_test, y_train, y_test, parameters):
    initial_classifier = SVC(degree=parameters["degree"])

    cputime_start_train = time.process_time()
    classifier = initial_classifier.fit(X_train, y_train)
    cputime_end_train = time.process_time()

    cputime_start_test = time.process_time()
    y_pred = classifier.predict(X_test)
    cputime_end_test = time.process_time()

    accuracy = classifier.score(X_test, y_test)

    return accuracy, cputime_end_train - cputime_start_train, cputime_end_test - cputime_start_test


def nearest_centroid_clustering(X_train, X_test, y_train, y_test, parameters):
    initial_classifier = NearestCentroid(metric=parameters["distance"])

    cputime_start_train = time.process_time()
    classifier = initial_classifier.fit(X_train, y_train)
    cputime_end_train = time.process_time()

    cputime_start_test = time.process_time()
    y_pred = classifier.predict(X_test)
    cputime_end_test = time.process_time()

    accuracy = classifier.score(X_test, y_test)

    return accuracy, cputime_end_train - cputime_start_train, cputime_end_test - cputime_start_test


def radius_neighbors_clustering(X_train, X_test, y_train, y_test, parameters):
    initial_classifier = RadiusNeighborsClassifier(n_jobs=-1, radius=parameters["radius"], metric=parameters["distance"])

    cputime_start_train = time.process_time()
    classifier = initial_classifier.fit(X_train, y_train)
    cputime_end_train = time.process_time()

    cputime_start_test = time.process_time()
    y_pred = classifier.predict(X_test)
    cputime_end_test = time.process_time()

    accuracy = classifier.score(X_test, y_test)

    return accuracy, cputime_end_train - cputime_start_train, cputime_end_test - cputime_start_test


def nca_clustering(X_train, X_test, y_train, y_test, parameters):
    nca = NeighborhoodComponentsAnalysis()
    initial_classifier_knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=parameters["k"], metric=parameters["distance"])

    cputime_start_train = time.process_time()
    nca.fit(X_train, y_train)
    classifier = initial_classifier_knn.fit(nca.transform(X_train), y_train)
    cputime_end_train = time.process_time()

    cputime_start_test = time.process_time()
    y_pred = classifier.predict(nca.transform(X_test))
    cputime_end_test = time.process_time()

    accuracy = classifier.score(nca.transform(X_test), y_test)

    return accuracy, cputime_end_train - cputime_start_train, cputime_end_test - cputime_start_test


def svc_sdg_clustering(X_train, X_test, y_train, y_test, parameters):
    initial_classifier = SGDClassifier()

    cputime_start_train = time.process_time()
    classifier = initial_classifier.fit(X_train, y_train)
    cputime_end_train = time.process_time()

    cputime_start_test = time.process_time()
    y_pred = classifier.predict(X_test)
    cputime_end_test = time.process_time()

    accuracy = classifier.score(X_test, y_test)

    return accuracy, cputime_end_train - cputime_start_train, cputime_end_test - cputime_start_test