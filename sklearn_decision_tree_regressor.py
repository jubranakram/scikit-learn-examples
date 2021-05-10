# -*- coding: utf-8 -*-
"""
about: decision tree regressor

author: jubran akram
"""
import warnings
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # load data set
    in_dat = load_boston()
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
    pipe = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=3))
    pipe.fit(X_train, y_train)

    # evaluate the score
    score_train = pipe.score(X_train, y_train)
    score_test = pipe.score(X_test, y_test)

    print(f"training accuracy: {score_train}")
    print(f"testing accuracy: {score_test}")

    print(pipe.steps[1][1].feature_importances_)
    plot_tree(pipe.steps[1][1])
