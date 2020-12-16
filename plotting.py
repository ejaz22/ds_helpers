def plot_class_hist(data, target, feature, kde=False):
    """
    In a binary classification setting this function plots 
    two histograms of a given variable grouped by a class label.
    
    It is a wrapper around Seaborn's .distplot()
    
    Parameters:
    data    : name of your pd.DataFrame
    target  : name of a target column in data (string)
    feature : name of a feature column you want to plot (string)
    kde     : if you want to plot density estimation (boolean)
    (C) Aleksander Molak, 2018 MIT License || https://github.com/AlxndrMlk/
    """
    
    sns.distplot(data[data[target]==1][feature],\
                 label='1', color='#b71633', norm_hist=True, kde=kde)
    sns.distplot(data[data[target]==0][feature],\
                 label='0', color='#417adb', norm_hist=True, kde=kde)
    plt.ylabel('Frequency')
    plt.title(feature)
    plt.legend()
    plt.show()
