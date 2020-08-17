
# get feature importance
def get_rf_feat_importances(X,y):
    rf = RandomForestClassifier(n_estimators=20, random_state = 42)
    rf.fit(X, y)
    df = pd.DataFrame(
        {'feature': X.columns, 'importance':rf.feature_importances_})
    df = df.sort_values(by=['importance'], ascending=False)
    return df
