import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def find_null_columns(df):
    """
    Return a list of columns with null values
    Args:
    df - dataframe - Dataframe to check columns of
    Returns:
    list of null columns
    """
    return df.columns[df.isnull().any()].tolist()


def null_column_report(df, total=True, percent=True, ):
    """
    Print each null column column in a dataframe that is null as well as percent null
    Args:
    df - pandas dataframe
    total - boolean - Flag to indicate whether to print total null records per column
    percent - boolean - Flag to indicate whether to print percent of column that is null
    Returns:
    None
    """
    num_null_columns = df.shape[1] - df.dropna(axis=1).shape[1]
    print('Number of columns with null values:\n{}\n'.format(num_null_columns))
    null_columns = find_null_columns(df)
    for col in null_columns:
        total_null_records = df[col].isnull().sum()
        print('Column:')
        print(col)
        if total:
            print('Total Nulls:')
            print(total_null_records)
        if percent:
            print('Percent Null:')
            print(round(total_null_records/df.shape[0], 2))
            print()

def null_column_report_df(df):
    """
    Searches a dataframe for null columns and returns a dataframe of the format
    Column | Total Nulls | Percent Nulls
    """
    num_null_columns = df.shape[1] - df.dropna(axis=1).shape[1]
    print('Number of columns with null values:\n{}\n'.format(num_null_columns))
    null_columns = df.columns[df.isnull().any()].tolist()
    null_info_records = []
    for col in null_columns:
        total_null_records = df[col].isnull().sum()
        percent_null_records = round(total_null_records/df.shape[0], 2)
        null_info_records.append({
            'Column':col,
            'Total_Null_Records':total_null_records,
            'Percent_Null_Records':percent_null_records
        })
    return pd.DataFrame(null_info_records)

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass a MultiIndex column dataframe and return the dataframe
    with a flattened columns. I.e. a single level of columns
    Parameters
    ----------
    df : pd.DataFrame
        The MultiIndex dataframe
    Returns
    -------
    flattened_df: pd.DataFrame
        A dataframe with a single level of column names
    Example
    -------
    multiIndex_df = pd.DataFrame(np.random.random((4,4)))
    multiIndex_df.columns = pd.MultiIndex.from_product([[1,2],['A','B']])
    flatten_multiindex_columns(multiIndex_df).columns
    > ['1_A', '1_B', '2_A', '2_B']
    """
    flat_df = df.copy()
    flat_columns = ['_'.join([
        # Cast to string in case of numeric column
        # names
        str(item) for item in multiIndex_col
    ])
        for multiIndex_col
        in flat_df.columns]
    flat_df.columns = flat_columns
    return flat_df

def column_cardinality(df, columns=None):
    """Given a dataframe and optionally subset of columns, return a table
    with the number of unique values associated with each feature and percent of
    total values that are unique"""
    if not columns:
        columns = df.columns.values.tolist()
    # Get number unique
    n_unique = [df[col].nunique() for col in columns]
    pct_unique = [cardinality/df.shape[0] for cardinality in n_unique]
    cardinality_df = pd.DataFrame.from_dict({
        'column':columns,
        'n_unique':n_unique,
        'pct_of_all_values_unique': pct_unique,
    },
        orient='columns').sort_values(by='pct_of_all_values_unique', ascending=False).reset_index(drop=True)
    return cardinality_df


# get varinace inflation factor
def get_vif(X):
    
    """
    Takes a pd.DataFrame or 2D np.array
    and prints Variance Inflation Factor 
    for every variable.
    """
    
    if isinstance(X, pd.DataFrame) == False:
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


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    
    Source: https://www.machinelearningplus.com/statistics/mahalanobis-distance/
    
    E.g.
    df_x = df[['carat', 'depth', 'price']].head(500)
    df_x['mahala'] = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
    df_x.head()
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()
    nulls = nulls[df.isnull().sum()>0].sort_values(ascending=False)
    nulls_report = pd.concat([nulls, nulls / df.shape[0]], axis=1, keys=['Missing_Values','Missing_Ratio'])
    if nulls_report.empty:
        print('The dataset has no null values',end='\n\n\n')
    else:
        print(nulls_report,end='\n\n\n')
    
    # describe stats
    print("4. Statistics")
    print(df.describe().T,end='\n\n\n')
