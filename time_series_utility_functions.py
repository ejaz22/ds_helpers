import numpy as np
import pandas as pd

#1 create lagged values
def ts_lags(df, lag=1):

    cols = [df.shift(i) for i in range(1, lag+1)]
    cols.append(df)      # append the original as the output y onto the last column
    df = pd.concat(cols,axis=1)
    df.fillna(0, inplace=True)
    return df


# Lagging value by one row
df['previous_days_stock_prize']=df["stock_prize"].shift(1)


#2 ceated datasets for time series lstm based models
def create_dataset(data, look_back=1):
    
    """
    this function takes numpy array as input and 
    returns two numpy array output
    Usage: # trainX is input, trainY is expected output
    trainX, trainY = create_dataset(train, look_back)      
    testX, testY = create_dataset(test, look_back)
    """
    
    X,Y = [], []
    i_range = len(data) - look_back - 1

    for i in range(0, i_range):
        X.append(data[i:(i+look_back)])
        Y.append(data[i + look_back])   
    return np.array(X), np.array(Y)

# describe/analyse data
def describe_data(df):
    """
    this function takes pandas dataframe as an argument
    and returns its analysis 
    """
    # dataset dimensions
    print("1. The dataset has {} rows and {} columns".format(df.shape[0],df.shape[1]),end='\n\n\n')
    
    # describe missing, unique and data_types
    print("2. GENERAL DESCRIPTION")
    summ = pd.DataFrame(df.dtypes,columns=['Data_Types'])
    summ = summ.reset_index()
    summ['Columns'] = summ['index']
    summ = summ[['Columns','Data_Types']]
    summ['Missing'] = df.isnull().sum().values    
    summ['Uniques'] = df.nunique().values
    print(summ, end='\n\n\n')
    
    
    # nulls analysis
    print("3. NULLS VALUES ANALYSIS")
    nulls = df.isnull().sum()
    nulls = nulls[df.isnull().sum()>0].sort_values(ascending=False)
    nulls_report = pd.concat([nulls, nulls / df.shape[0]], axis=1, keys=['Missing_Values','Missing_Ratio'])
    print(nulls_report,end='\n\n\n')
    
    # describe stats
    print("4. STATISTICS")
    print(df.describe().T,end='\n\n\n')
