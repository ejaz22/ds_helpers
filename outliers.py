def remove_outliers(df, outlier_column_name, drop_anomalies=False, threshold=3):
    """
    Given a dataframe, remove outliers from a given column, according to some threshold.
    Return a dataframe.
    """
    from scipy.stats import zscore
    z_name = outlier_column_name + '_z'
    df[z_name] = df[[outlier_column_name]].apply(zscore)
    initial = df.shape[0]
    if drop_anomalies:
        df = df[(abs(df[z_name]) < threshold)]
        df = df.drop(z_name, axis=1)
        after = initial - df.shape[0]
        print(f"{after} outliers for {outlier_column_name} have been removed")
    return df


def remove_entries_outside_iq_range(df, col):
    q1 = df.diff_entry.quantile(0.25)
    q3 = df.diff_entry.quantile(0.75)
    iqr = q3 - q1
    iq_rem = df[~((df[col] < (q1 - iqr)) | (df[col] > (q3 + iqr)))]
    return iq_rem
