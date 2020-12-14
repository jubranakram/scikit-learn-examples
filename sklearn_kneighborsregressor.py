# -*- coding: utf-8 -*-
"""
about: k-nearest neighbors (regression)

author: jubran akram
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    num_neighbors = np.arange(1, 20, 2)
    num_tests = len(num_neighbors)
    
    acc_train = np.zeros((num_tests,))
    acc_test = np.zeros((num_tests, ))
    
    for idx in range(num_tests):
        
        knn = KNeighborsRegressor(n_neighbors=num_neighbors[idx])
        knn.fit(X_train_scaled, y_train)
    
        # evaluate the score (for regression, R-square)
        # R**2 = measure of goodness of a prediction for a regression model
    
        acc_train[idx] = knn.score(X_train_scaled, y_train)*100
        acc_test[idx] = knn.score(X_test_scaled, y_test)*100
       
    plt.plot(num_neighbors, acc_train, '-ok', markerfacecolor='red', label='train')
    plt.plot(num_neighbors, acc_test, '-ok', markerfacecolor='blue', label='test')
    plt.legend()