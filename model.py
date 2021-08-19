import pandas as pd
import numpy as np

# visualization imports
import matplotlib.pyplot as plt

# import modeling tools
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, explained_variance_score

def train_model(X, y, model, model_name):
    '''
    This function takes in the X_train and y_train, model object and model name, fits the
    model and returns predictions and a dictionary containg the model RMSE
    and R^2 scores on train split.
    '''
    # fit model to X_train
    model.fit(X, y)
    # predict X_train
    predictions = model.predict(X)
    # get rmse and r^2 for model predictions on X
    rmse, r2 = get_metrics(y, predictions)
    # add metrics to dictionary
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict

def model_testing(X, y, model, model_name):
    '''
    This function takes in the X and y for validate or test sets, model object and model name and
    returns predictions and a dictionary containg the model RMSE and R^2 scores
    on validate or test
    '''
    # get predictions on X
    predictions = model.predict(X)
    # get for performance and assign them to dictionary
    rmse, r2 = get_metrics(y, predictions)
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict

def get_metrics(true, predicted, display=False):
    '''
    This function takes in the true and predicted values and returns the rmse and R^2 for the
    model performance.
    '''
    rmse = mean_squared_error(true, predicted, squared=False)
    r2 = explained_variance_score(true, predicted)
    if display == True:
        print(f'Model RMSE: {rmse:.2g}')
        print(f'       R^2: {r2:.2g}')
    return rmse, r2

def plot_residuals