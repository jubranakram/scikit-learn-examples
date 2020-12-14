# -*- coding: utf-8 -*-
"""
about: random forest classifier

author: jubran akram
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    # load data set
    in_dat = load_breast_cancer()
    X, y = in_dat.data, in_dat.target
    num_samples_total, num_features = X.shape
    
    # if test_size is not specified, default is 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        shuffle=True,
                                                        random_state=0)
    
    num_samples_train = X_train.shape[0]
    num_samples_test = X_test.shape[0]
    
    print(f" Train set: {num_samples_train/num_samples_total*100}%")
    print(f" Test set: {num_samples_test/num_samples_total*100}%")
    
    
    # create pipeline
    # n_jobs = None (single processor), -1 (all processors)
    pipe = make_pipeline(StandardScaler(), 
                         RandomForestClassifier(n_estimators=200,
                                                max_depth = 10,
                                                max_samples=0.6,
                                                max_features=0.6,
                                                random_state=0, 
                                                n_jobs=-1)) 
    pipe.fit(X_train, y_train)
    
    # evaluate the score   
    score_train = pipe.score(X_train, y_train)
    score_test = pipe.score(X_test, y_test)
       
    print(f"training accuracy: {score_train}")
    print(f"testing accuracy: {score_test}")
    
    print(pipe.steps[1][1].feature_importances_)
    
    fig, axs = plt.subplots()
    
    axs.barh(np.arange(num_features), pipe.steps[1][1].feature_importances_)
    axs.set_yticks(np.arange(num_features))
    axs.set_yticklabels(in_dat.feature_names)
    axs.invert_yaxis()  # labels read top-to-bottom
    axs.set_xlabel('Feature importance')
    axs.set_title('Random forest classification results')

    