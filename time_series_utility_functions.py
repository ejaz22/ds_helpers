# Time series lags

import numpy as np
import pandas as pd

#1 create lagged values
def ts_lags(df, lag=1):

    cols = [df.shift(i) for i in range(1, lag+1)]
    cols.append(df)      # append the original as the output y onto the last column
    df = pd.concat(cols,axis=1)
    df.fillna(0, inplace=True)
    return df
    

#2 ceated datasets for time series lstm based models
def create_dataset(data, look_back=1):
    
    """
    this function takes numpy array as input and 
    returns two numpy array output
    """
    
    X,Y = [], []
    i_range = len(data) - look_back - 1

    for i in range(0, i_range):
        X.append(data[i:(i+look_back)])
        Y.append(data[i + look_back])   
    return np.array(X), np.array(Y)
