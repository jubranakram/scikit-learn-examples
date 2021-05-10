# -*- coding: utf-8 -*-
"""
about: k-nearest neighbors

author: jubran akram
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    # load data set
    X, y = load_iris(return_X_y=True)
    num_samples_total, num_features = X.shape

    # if test_size is not specified, default is 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True,
                                                        random_state=0)

    num_samples_train = X_train.shape[0]
    num_samples_test = X_test.shape[0]

    print(f" Train set: {num_samples_train/num_samples_total*100}%")
    print(f" Test set: {num_samples_test/num_samples_total*100}%")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # evaluate the accuracy
    # accuracy = fraction of correct predictions

    accuracy = np.mean(y_pred == y_test)*100
    print(f" Test score is {accuracy:.1f}%.")

    # we can also use knn.score
    acc_knn_test = knn.score(X_test, y_test)*100
    print(f" Test score is {acc_knn_test:.1f}%.")
