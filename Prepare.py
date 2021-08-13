import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE


########################### Prepare Texas Salaries Script ############################

def prepare_tex(df):
    '''
    This function prepares our salary dataframe with a focus on ease of workflow.
    It lower cases the capital columns and renames all the abbreviated column to a more
    human readable format.
    
    It also drops duplicate records and changes the hire date to a date time formatted object
    '''
    
    ## first let's rename all the columns to lowercase for easeier workflow
    df.rename(str.lower, axis='columns', inplace=True)
    
    ## now let's remove any potential leading whitesapce
    df.columns = df.columns.str.strip()
    
    ## Let's drop unneccasary columns that won't be any help with predicting our
    ## target variable because they are either incomplete or insignificant information
    df = df.drop(columns = ['jobclass','mi', 'rate', 'statenum', 'duplicated',
                            'multiple_full_time_jobs',
                            'combined_multiple_jobs', 'summed_annual_salary', 
                            'hide_from_search'])
    
    ## renaming columns for ease of workflow
    df = df.rename(columns = {'jc title': 'title', 
                              'hiredt': 'hire_date', 
                              'hrswkd': 'hours_worked','name': 
                              'agency','agy': 'agency_id',
                              'monthly': 'monthly_salary',
                              'annual': 'annual_salary'})
    
    
    ## changing hire date to date time format
    df.hire_date = pd.to_datetime(df.hire_date)
    
    ## dropping the duplciates rows
    df = df.drop_duplicates()
    
    return df
    
    