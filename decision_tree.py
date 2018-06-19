# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:38:28 2018

@author: wael
"""

from data import import_data
from splitdataset import splitdataset
from model_performance import perfor
from prediction import predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

data = import_data()
X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

# Decision tree with entropy
clf_object = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 6, min_samples_leaf = 5)
clf_object.fit(X_train, y_train)
    # Performing training
    # Predicton on test 
y_pred,rfecv=predict(X_test , clf_object , X_train , y_train)
perfor(y_test , y_pred , rfecv , X_train)
joblib.dump(clf_object, 'pfa.pkl')
