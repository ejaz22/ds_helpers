import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from metrics_helper import get_confusion_rates
from scipy import interp


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

    def plot_roc(y, y_pred_prob):
    '''
    for binary classification
    '''
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % ( roc_auc))    

def plot_roc_cv(classifier, X, y, cv):
    '''
    cv = KFold(len(y),n_folds=5)
    '''
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    


def plot_feature_importance_gbc(clf, feature_names, topk = 25, figsize = (50,70) ):
    #topk = 25
    fig = plt.figure(figsize = figsize)
    importances = clf.feature_importances_ 
    sorted_idx = np.argsort(importances)[-topk:]
    #sorted_idx = sorted_idx[::-1]
    padding = np.arange(len(sorted_idx)) + 0.5
    #plt.barh(padding, importances[sorted_idx], align='center')
    plt.barh(padding, importances[sorted_idx],\
       color="b", alpha = 0.5, align="center")    
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.yticks(padding, feature_names[sorted_idx])
    #plt.show()
    return fig

def plot_feature_importance(rf, feature_names, topk = 25, errorbar=False, figsize = (50,70) ):
    #topk = 25
    fig = plt.figure(figsize = figsize)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)    
    sorted_idx = np.argsort(importances)[-topk:]
    padding = np.arange(len(sorted_idx)) + 0.5
    #plt.barh(padding, importances[sorted_idx], align='center')
    if errorbar: 
        plt.barh(padding, importances[sorted_idx],\
            color="b", alpha = 0.5, xerr=std[sorted_idx], align="center")   
    else:
        plt.barh(padding, importances[sorted_idx],\
        color="b", alpha = 0.5, align="center")  
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.yticks(padding, feature_names[sorted_idx])
    plt.show()
    #plt.plot()
    return fig
