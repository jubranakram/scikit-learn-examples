# -*- coding: utf-8 -*-
"""
about: naive bayes classifier

author: jubran akram
"""
import warnings
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    # load data set
    in_dat = load_iris()
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
    
    # effect of normalization, also check without feature scaling for comparison
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    e_reg = GaussianNB()
    e_reg.fit(X_train_scaled, y_train)
    
    # evaluate the score (for classification, accuracy)
    
    score_train = e_reg.score(X_train_scaled, y_train)
    score_test = e_reg.score(X_test_scaled, y_test)
       
    print(f"training accuracy: {score_train}")
    print(f"testing accuracy: {score_test}")
    
    # use e_reg.proba(X_train) to find the probabilities for each class