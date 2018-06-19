# -*- coding: utf-8 -*-
"""
Created on Sat May  5 01:53:34 2018

@author: wael
"""

from data import import_data
from splitdataset import splitdataset
from model_performance import perfor
from prediction import predict  
from sklearn.svm import SVR    


data = import_data()
X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

# Decision tree with entropy
clf_object = SVR(kernel='linear') 

clf_object.fit(X_train, y_train)
    # Performing training
    # Predicton on test 
y_pred,rfecv=predict(X_test , clf_object , X_train , y_train)
perfor(y_test , y_pred , rfecv , X_train)
