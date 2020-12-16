import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# get varinace inflation factor
def get_vif(X):
    
    """
    Takes a pd.DataFrame or 2D np.array
    and prints Variance Inflation Factor 
    for every variable.
    """
    
    if isinstance(data, pd.DataFrame) == False:
        X = pd.DataFrame(X)
    
    X['__INTERCEPT'] = np.ones(X.shape[0])
    
    for i in range(X.shape[1]-1):
        the_vif = vif(X.values, i)
        print("VIF for column {:03}: {:.02f}".format(i, the_vif))


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
    # set pandas
    pd.set_option('precision', 2)
    
    # dataset dimensions
    print("1. The dataset has {} records and {} columns".format(df.shape[0],df.shape[1]),end='\n\n\n')
    
    # describe missing, unique and data_types
    print("2. Attributes details are as follows")
    summ = pd.DataFrame(df.dtypes,columns=['Data_Types'])
    summ = summ.reset_index()
    summ['Columns'] = summ['index']
    summ = summ[['Columns','Data_Types']]
    summ['Missing'] = df.isnull().sum().values    
    summ['Uniques'] = df.nunique().values
    print(summ, end='\n\n\n')
    
    
    # nulls analysis
    print("3. Null analysis:")
    nulls = df.isnull().sum()
    nulls = nulls[df.isnull().sum()>0].sort_values(ascending=False)
    nulls_report = pd.concat([nulls, nulls / df.shape[0]], axis=1, keys=['Missing_Values','Missing_Ratio'])
    if nulls_report.empty:
        print('The dataset has no null values',end='\n\n\n')
    else:
        print(nulls_report,end='\n\n\n')
    
    # describe stats
    print("4. Statistics")
    print(df.describe().T,end='\n\n\n')
