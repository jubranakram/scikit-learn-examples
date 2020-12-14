# -*- coding: utf-8 -*-
"""
about: lasso regression

author: jubran akram
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

if __name__ == '__main__':
    
    # load data set
    in_dat = load_boston()
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
    
    l_reg = Lasso()
    l_reg.fit(X_train, y_train)
    
    # weights (w) are stored in the coef_ attribute
    # bias (b) is stored in the intercept_ attribute
    
    print(f" weights: {l_reg.coef_}")
    print(f" bias: {l_reg.intercept_}")
    
    # evaluate the score (for regression, R-square)
    # R**2 = measure of goodness of a prediction for a regression model
    
    score_train = l_reg.score(X_train, y_train)
    score_test = l_reg.score(X_test, y_test)
       
    print(f"training score: {score_train}")
    print(f"testing score: {score_test}")
    print(f"number of features used: {np.sum(l_reg.coef_ !=0)}")