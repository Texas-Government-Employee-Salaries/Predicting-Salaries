import numpy as np
import pandas as pd

from scipy import stats

def two_tail_ttest(sample1, sample2, alpha=0.05, variance=False):
    '''
    This function takes in two samples, the alpha, and variance of the samples
    and conducts a two tail ttest and returns whether we can accept or reject the null hypothesis
    '''
    
    # assign t and p value
    t, p = stats.ttest_ind(sample1, sample2, equal_var=variance)
    
    
    if (p < alpha) & (t > 0):
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")
        
        
def one_tail_ttest(sample1, sample2, alpha=0.05, variance=False):
    '''
    This function takes in two samples, the alpha, and variance of the samples
    and conducts a two tail ttest and returns whether we can accept or reject the null hypothesis
    '''
    
    # assign t and p value
    t, p = stats.ttest_ind(sample1, sample2, equal_var=variance)
    
    
    if (p/2 < alpha) & (t > 0):
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")