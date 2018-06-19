# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:51:48 2018

@author: wael
"""

from sklearn.metrics import accuracy_score
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def perfor(y_test, y_pred , rfecv , X_train):

    ac = accuracy_score(y_test,y_pred)
    print('Accuracy is: ',ac*100)
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,fmt="d")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', X_train.columns[rfecv.support_])
    
