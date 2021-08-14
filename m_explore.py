import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
import acquire as aq
import prepare as pr
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor


def explore_univariate(df, cat_vars, quant_vars):
    '''
    This function takes in categorical and quantitative (continuous) variables from a dataframe.
    It returns a bar plot for each categorical variable
    and a histogram and boxplot for each continuous variable.
    '''
    # plot frequencies for each categorical variable
    for var in cat_vars: 
        print('Bar Plot of ' + var)
        bp = df[var].hist()
        plt.xlabel(var)
        plt.ylabel('count')
        bp.grid(False)
        plt.show()
        
    # print histogram for each continuous variable
    for var in quant_vars:
        generate_hist(df, var)
        # creating boxplot for each variable
        plt.figure(figsize=(10,5))
        sns.boxplot(x=var, data=df,  palette="twilight_shifted")
        plt.title('Distribution of ' + var)
        plt.show()


def generate_hist(df, var):
    '''
    Helper function. Given a dataframe DF and a variable to plot, this function will 
    generate and display a histogram for that variable.
    '''
    print ('Distribution of ' + var)
    df[var].hist()
    plt.grid(False)
    plt.xlabel(var)
    plt.ylabel('Number of Properties')
    plt.show()

def generate_barplot(df, target, var):
    '''
    Helper function to generate barplots. Given a dataframe df, a target column and a 
    variable, this will generate and draw a barplot for that data set.
    '''
    overall_mean = df[target].mean()
    sns.barplot(var, target, data=df, palette="twilight_shifted")
    plt.xlabel('')
    plt.ylabel('Tax Value')
    plt.title('Bar plot of ' + var + ' vs ' + target)
    plt.axhline(overall_mean, ls = '--', color = 'grey')
    plt.show()

def generate_boxplot(df,target, var):
    '''
    Given a dataframe df, a target column and a variable to plot, this helper function
    will generate and display a boxplot comparing the given data elements.
    '''
    plt.figure(figsize=(10,5))
    sns.boxplot(y=var, x=target, data=df,  palette="twilight_shifted")
    plt.title('Boxplot of ' + var)
    plt.show()

def generate_countplot(df, target, var):
    '''
    Another helper function used to display a plot. Given a dataframe df, a target
    column and a variable, this function will create and display a countplot.
    '''
    sns.countplot(data=df, x=var, hue=target,  palette="twilight_shifted")
    plt.tight_layout()
    plt.show()

def generate_scatterplot(df, target, var):
    sns.lmplot(x=var, y=target, data=df, scatter=True,  line_kws={'color': 'red'})

def explore_bivariate(df, target, cat_vars, quant_vars):
    '''
    This function takes in takes in a dataframe, the name of the binary target variable, a list of 
    the names of the categorical variables and a list of the names of the quantitative variables. It returns
    bar plots for categorical variables and scatterplots for quantitative(continuous) variables.
    For each categorical variable, the bar plot shows the tax value for each class in each category
    with a dotted line for the average overall tax value. 
    The scatterplots show the relationship between quantitative variables and the target variable.
    '''
    for var in cat_vars:
        # bar plot with overall horizontal line
        generate_barplot(df, target, var)
    for var in quant_vars:
        # creates scatterplot with regression line
        generate_scatterplot(df, target, var)
        
     
def explore_multivariate(train, target, cat_vars, quant_vars):
    '''
    This function takes in takes in a dataframe, the name of the binary target variable, a list of 
    the names of the categorical variables and a list of the names of the quantitative variables.
    It generates boxplots showing the target variable for each class of the categorical variables 
    against the quantitative variables.
    '''
    for cat in cat_vars:
        for quant in quant_vars:
            sns.lmplot(x=quant, y=target, data=train, scatter=True, hue=cat, palette ='colorblind')
            plt.xlabel(quant)
            plt.ylabel(target)
            plt.title(quant + ' vs ' + target + ' by ' + cat)
            plt.show()

def plot_variable_pairs(train, cols, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line.
    '''
    plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.7}}
    sns.pairplot(train[cols], hue=hue, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()


def select_rfe(X, y, k, return_rankings=False, model=LinearRegression()):
    # Use the passed model, LinearRegression by default
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    features = X.columns[rfe.support_].tolist()
    if return_rankings:
        rankings = pd.Series(dict(zip(X.columns, rfe.ranking_)))
        return features, rankings
    else:
        return features