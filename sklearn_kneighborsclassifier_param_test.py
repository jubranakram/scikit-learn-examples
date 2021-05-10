# -*- coding: utf-8 -*-
"""
about: k-nearest neighbors parameter evaluation

author: jubran akram
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    # load data set
    in_dat = load_breast_cancer()
    X, y = in_dat.data, in_dat.target
    num_samples_total, num_features = X.shape

    # if test_size is not specified, default is 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True,
                                                        random_state=0)

    num_samples_train = X_train.shape[0]
    num_samples_test = X_test.shape[0]

    print(f" Train set: {num_samples_train/num_samples_total*100}%")
    print(f" Test set: {num_samples_test/num_samples_total*100}%")

    num_neighbors = np.arange(1, 20, 2)
    num_tests = len(num_neighbors)

    acc_train = np.zeros((num_tests,))
    acc_test = np.zeros((num_tests, ))

    for idx in range(num_tests):

        knn = KNeighborsClassifier(n_neighbors=num_neighbors[idx])
        knn.fit(X_train, y_train)

        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)

        # evaluate the accuracy
        # accuracy = fraction of correct predictions

        acc_train[idx] = knn.score(X_train, y_train)*100
        acc_test[idx] = knn.score(X_test, y_test)*100

    plt.plot(num_neighbors, acc_train, '-ok',
             markerfacecolor='red', label='train')
    plt.plot(num_neighbors, acc_test, '-ok',
             markerfacecolor='blue', label='test')
    plt.legend()
