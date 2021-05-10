# -*- coding: utf-8 -*-
"""
about: linear regression (train-test size)

author: jubran akram
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':

    # load data set
    in_dat = load_boston()
    X, y = in_dat.data, in_dat.target
    num_samples_total, num_features = X.shape

    test_size = np.arange(0.1, 0.5, 0.1)
    num_tests = len(test_size)

    score_train = np.zeros((num_tests,))
    score_test = np.zeros((num_tests,))

    for idx in range(num_tests):

        # if test_size is not specified, default is 75% train, 25% test
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size[idx],
                                                            shuffle=True,
                                                            random_state=0)

        num_samples_train = X_train.shape[0]
        num_samples_test = X_test.shape[0]

        print(f" Train set: {num_samples_train/num_samples_total*100}%")
        print(f" Test set: {num_samples_test/num_samples_total*100}%")

        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)

        # weights (w) are stored in the coef_ attribute
        # bias (b) is stored in the intercept_ attribute

        print(f" weights: {lin_reg.coef_}")
        print(f" bias: {lin_reg.intercept_}")

        # evaluate the score (for regression, R-square)
        # R**2 = measure of goodness of a prediction for a regression model

        score_train[idx] = lin_reg.score(X_train, y_train)
        score_test[idx] = lin_reg.score(X_test, y_test)

    plt.plot(test_size, score_train, '-ok',
             markerfacecolor='red', label='train')
    plt.plot(test_size, score_test, '-ok',
             markerfacecolor='blue', label='test')
