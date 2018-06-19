# -*- coding: utf-8 -*-
"""
Created on Sat May  5 01:19:02 2018

@author: wael
"""

from data import import_data
from splitdataset import splitdataset
from model_performance import perfor
from prediction import predict  
from sklearn.linear_model import LogisticRegression


data = import_data()
X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

# Decision tree with entropy
clf_object = LogisticRegression()
clf_object.fit(X_train, y_train)
    # Performing training
    # Predicton on test 
y_pred,rfecv=predict(X_test , clf_object , X_train , y_train)
perfor(y_test , y_pred , rfecv , X_train)

