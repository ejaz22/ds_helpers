# Time series lags

import numpy as np
import pandas as pd

# create lagged values
def ts_lags(df, lag=1):

    cols = [df.shift(i) for i in range(1, lag+1)]
    cols.append(df)      # append the original as the output y onto the last column
    df = pd.concat(cols, axis=1)
    df.fillna(0, inplace=True)   # turn NaN to 0
    return df
    

# ceated datasets for time series lstm based models
def create_dataset(df, look_back=1):
    
    """
    this function numpy array as input and returns two
    numpy array output
    """
    
    X,Y = [], []
    i_range = len(data) - look_back - 1

    for i in range(0, i_range):
        X.append(data[i:(i+look_back)])    # index can move down to len(dataset)-1
        Y.append(data[i + look_back])      # Y is the item that skips look_back number of items
    
    return np.array(X), np.array(Y)
