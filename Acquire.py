# packages for data analysis & mapping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
import seaborn as sns
from datetime import date 

# modeling methods
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE 
import sklearn.preprocessing

# address warnings
import warnings
warnings.filterwarnings("ignore")

import os.path

def get_texas_data(cached=False):
    '''
    This function reads in college data from a URL and writes data to
    a csv file if cached == False or if cached == True reads in college df from
    a csv file, returns df.
    '''
    url = 'https://s3.amazonaws.com/raw.texastribune.org/state_of_texas/salaries/03_non_duplicated_employees/2021-07-01.csv'
    
    if cached == False or os.path.isfile('texas.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = pd.read_csv(url)
        
        # Write DataFrame to a csv file.
        df.to_csv('texas.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('texas.csv', index_col=0)
        
    return df

def get_data_summary(df):
    '''
    This function takes in a dataframe and prints out the shape of the df, number of missing values, 
    columns and their data types, summary statistics of the df, as well as the value counts for categorical variables.
    '''
    # Print out the "shape" of our dataframe - the rows and columns we have to work with
    print(f'The telco dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('')
    print('-------------------')

    # print the number of missing values in our dataframe
    print(f'There are total of {df.isna().sum().sum()} missing values in the entire dataframe.')
    print('')
    print('-------------------')

    # print some information regarding our dataframe
    print(df.info())
    print('')
    print('-------------------')
    
    # print out summary stats for our dataset
    print('Here are the summary statistics of our dataset')
    print(df.describe())
    print('-------------------')

    print('Here are the categories and their relative proportions')
    # check different categories and proportions of each category for object type cols
    ignore_vars = ['customer_id','tenure', 'monthly_charges','total_charges']
    for col in df.columns:
        if col not in ignore_vars:
            print(df[col].value_counts())
            print('')
            print(f'proportions of {col}')
            print('')
            print(df[col].value_counts(normalize=True,dropna=False))
            print('-------------------')