import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    
    The function also trims leading and trailing white space on all the string values for the
    object columns.
    
    It also drops duplicate records and changes the hire date to a date time formatted object
    '''
    
    ## first let's rename all the columns to lowercase for easeier workflow
    df.rename(str.lower, axis='columns', inplace=True)
    
    ## now let's remove any potential leading whitesapce from column names
    df.columns = df.columns.str.strip()
    
    ## removing any potential whitespace from the columns with string values
    df.race = df.race.str.strip()
    df.name = df.name.str.strip()
    df.lastname = df.lastname.str.strip()
    df.firstname = df.firstname.str.strip()
    df['jc title'] = df['jc title'].str.strip()
    df.sex = df.sex.str.strip()
    df.emptype = df.emptype.str.strip()
    
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
    
    ## getting rid of clerical errors
    ## these three employees had a hire date in the future
    df = df.drop(index=[794, 118710, 144495])
    
    return df


def create_features(df):
    '''
    This function is designed to create multipal features form the original columns of 
    the dataframe.
    
    It uses one hot encoding to create an is_female column with the values of 1 for female
    and 0 for male
    
    It also uses one hot encoding to create categorical columns for the three top races: white,
    hispanic, and black. While also using a label encoder to create a machine formatted 
    race_encoded column where the values are different integers instead of strings.
    
    Knowing the last updated date of the employee information is July 1st, 2021 we were able 
    to use date time format and a timedelta to create a tenure in months column and a tenure in 
    years column by subtracting the hire_date column from the domain knowledge date July 1st, 
    2021
    '''
    ## creating a label encoder to encode certain categorical columns
    label_encoder = LabelEncoder()
    
    ## one hot encoding column for gender
    df['is_female'] = np.where(df.sex == 'FEMALE', 1, 0)
    
    ## one hot encoding the top three races in the dataframe
    df['is_white'] = np.where(df.race == 'WHITE', 1, 0)
    df['is_hispanic'] = np.where(df.race == 'HISPANIC', 1, 0)
    df['is_black'] = np.where(df.race == 'BLACK', 1, 0)
    
    ## one hot encoding a BIPOC: Black Indigenous People of Color column
    df['is_BIPOC'] = np.where(df.race != 'WHITE', 1, 0)
    
    ## creating a race column that is incoded for machine readable formate
    df['race_encoded'] = label_encoder.fit_transform(df['race'])
    
    ## creating a tenure in months column by subrtracting the hire date from the last updated 
    ## date of the dataframe (7/1/21) and dividing it by a time delta 
    df['tenure_months'] = np.round((pd.to_datetime('2021-07-01') -
                                    df['hire_date'])/np.timedelta64(1,'M'))
    
    ## casting tenure in months as an int
    df['tenure_months'] = df['tenure_months'].astype(int)
    
    ## creating a tenure in years column and rounding it to one decimal place
    df['tenure_years'] = np.round(df['tenure_months'] / 12, 1)
    
    ## creating a categorical column of whether someone is an elected official or not
    df['is_elected'] = np.where((df.title == 'ELECTED OFFICIAL') | 
                            (df.title == 'JUSTICE') |
                            (df.title == 'ATTORNEY GENERAL') |
                            (df.title == 'GOVERNOR') | 
                            (df.title == 'LIEUTENANT GOVERNOR') |
                            (df.title == 'COMPTROLLER OF PUBLIC ACCOUNTS') |
                            (df.title.str.startswith('COMMISSIONER'))
                            , 1, 0)
    
    return df


def split_data(df):
    '''
    This function is designed to split out data for modeling into train, validate, and test 
    dataframes.
    
    It will also perform quality assurance checks on each dataframe to make sure the target 
    variable was correctcly stratified into each dataframe.
    '''
    
    ## splitting the data stratifying for out target variable is_fraud
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123)
    
    print('Making Sure Our Shapes Look Good')
    print(f'Train: {train.shape}, Validate: {validate.shape}, Test: {test.shape}')
    
    return train, validate, test
    
    