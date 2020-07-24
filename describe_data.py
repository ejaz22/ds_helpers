"""
This function performs several analysis on passed dataframe.
dependency : tabulate module
"""

def describe_data(df,target_var=None):
    
    """
    this function takes pandas dataframe as an argument
    and returns several analysis on dataset 
    """
    
    if not isinstance(df, pd.DataFrame):
        raise Exception ("Not a pandas dataframe")
    
    # set display of numbers
    pd.set_option('precision', 2)
    pd.options.display.float_format = "{:,.2f}".format
    
    # dataset dimensions
    print("1. The dataset has {} records and {} columns".format(df.shape[0],df.shape[1]),end='\n\n\n')
    
    # describe missing, unique and data_types
    print("2. Columns details: ",end='\n\n')
    summ = pd.DataFrame(df.dtypes,columns=['Data_Types'])
    summ = summ.reset_index()
    summ['Columns'] = summ['index']
    summ = summ[['Columns','Data_Types']]
    summ['Missing'] = df.isnull().sum().values    
    summ['Uniques'] = df.nunique().values
    print(summ.to_markdown(showindex=False,floatfmt=('.2f')), end='\n\n\n')
    
    
    # null values analysis
    print("3. Null values analysis:",end='\n\n')
    nulls = df.isnull().sum()
    nulls = nulls[df.isnull().sum()>0].sort_values(ascending=False)
    nulls_report = pd.concat([nulls, nulls / df.shape[0]], axis=1, keys=['Missing_Values','Missing_Ratio'])
    if nulls_report.empty:
        print('The dataset has no null values',end='\n\n\n')
    else:
        print(nulls_report,end='\n\n\n')
    
    # describe stats
    print("4. Statistics:",end='\n\n')
    stat = df.describe().T
    print(stat.to_markdown(floatfmt=('.2f')),end='\n\n\n')
    
    # Correlation analysis(pearson)
    print("5. Correlation Analysis (pearson)",end='\n\n')
    corr = df.corr(method ='pearson')
    print(corr.to_markdown(floatfmt=('.2f')),end='\n\n\n')
