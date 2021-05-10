# -*- coding: utf-8 -*-
"""
about: ridge regression (train-test size)

author: jubran akram
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

if __name__ == '__main__':

    # load data set
    in_dat = load_boston()
    X, y = in_dat.data, in_dat.target
    num_samples_total, num_features = X.shape

    test_size = np.arange(0.1, 0.5, 0.1)
    num_tests = len(test_size)

    alpha = np.array([0.01, 0.01, 1, 10])
    num_alpha = len(alpha)

    score_train = np.zeros((num_tests, num_alpha))
    score_test = np.zeros((num_tests, num_alpha))

    for idx in range(num_tests):

        # if test_size is not specified, default is 75% train, 25% test
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size[idx],
                                                            shuffle=True,
                                                            random_state=0)

        for idx_1 in range(num_alpha):
            r_reg = Ridge(alpha=alpha[idx_1])
            r_reg.fit(X_train, y_train)

            # weights (w) are stored in the coef_ attribute
            # bias (b) is stored in the intercept_ attribute

            print(f" weights: {r_reg.coef_}")
            print(f" bias: {r_reg.intercept_}")

            # evaluate the score (for regression, R-square)
            # R**2 = measure of goodness of a prediction for a regression model

            score_train[idx, idx_1] = r_reg.score(X_train, y_train)
            score_test[idx, idx_1] = r_reg.score(X_test, y_test)

    marker_detail = ['-ok', '-vk', '-^k', '-pk']
    for idx in range(num_alpha):
        plt.plot(test_size, score_train[:, idx], marker_detail[idx],
                 markerfacecolor='red', label=r'train ($\alpha$ = ' + '{})'.format(alpha[idx]))
        plt.plot(test_size, score_test[:, idx], marker_detail[idx],
                 markerfacecolor='blue', label='test')

    plt.legend()
