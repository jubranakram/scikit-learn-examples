# -*- coding: utf-8 -*-
"""
about: scikit-learn toy data sets (regression)
author: jubran akram
"""


from sklearn.datasets import make_blobs

def print_horizontal_line(line_string = '---', string_reps = 10):
    print(line_string*string_reps)
    
if __name__ == '__main__':
    
    # parameters
    
    params = {'n_samples': 500,
              'n_features': 3,
              'centers': None,
              'cluster_std': [1, 0.8, 0.7],
              'random_state': 0,
              'return_centers': True}  
    
    
    X, y, centers = make_blobs(**params)   
    
    print(X.shape)
    
    










# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.datasets import load_breast_cancer

# def print_horizontal_line():
#     print('---'*10)
    
# if __name__ == '__main__':
    
#     # return_X_y=False, as_frame=True loads a Bunch object containing 
#     # information as pandas dataframes
    
#     input_dat = load_breast_cancer(as_frame=True)
    
#     print(input_dat.DESCR)
    
#     num_samples, num_features = input_dat.data.shape
    
#     sns.pairplot(input_dat.data.iloc[:, :10])
    
   
    
   
    
   
    
   
    
   # print_horizontal_line()
   #  print('file contents')
   #  print_horizontal_line()
   #  print(list(input_dat.keys()))
    
   #  print_horizontal_line()
   #  print(input_dat.DESCR)
    
   #  fig, axs = plt.subplots(10, 10)
   #  axs = axs.flatten()
    
   #  for idx in range(100):
   #      axs[idx].imshow(input_dat.data[idx, :].reshape(8, 8), cmap='gray')
   #      axs[idx].get_xaxis().set_visible(False)
   #      axs[idx].get_yaxis().set_visible(False)
    # x_df = pd.DataFrame(input_dat.data, columns = input_dat.feature_names)
    # print(x_df.head())
    # print(x_df.describe())
    
    # sns.pairplot(x_df)
    # print(input_dat.target.shape)
    
    
    
    
    
    # # load data as input (X), labels (y)
    # X, y = load_boston(return_X_y=True)
    
    # # number of examples and features
    # num_samples, num_features = X.shape
    
    # print_horizontal_line()
    # print('input and label dimensions')
    # print_horizontal_line()
    # print(f" Input data have {num_samples} examples and {num_features} features")
    # print(f" Label data have dimensions of {y.shape}.")
    
    # x_df = pd.DataFrame(X)
    # sns.pairplot(x_df)

    