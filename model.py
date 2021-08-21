import pandas as pd
import numpy as np

# visualization imports
import matplotlib.pyplot as plt

# import modeling tools
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, explained_variance_score



def get_metrics(df, model_name,rmse_validate,r2_validate):
    df = df.append({
        'model': model_name,
        'rmse_outofsample':rmse_validate, 
        'r^2_outofsample':r2_validate}, ignore_index=True)
    return df
