# -*- coding: utf-8 -*-
"""
Created on Fri May  4 23:20:04 2018

@author: wael
"""

from sklearn.feature_selection import RFECV






def predict(X_test, clf_object , x_train , y_train):
    
    
     
    rfecv = RFECV(estimator=clf_object, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
    rfecv = rfecv.fit(x_train, y_train)
    y_pred = rfecv.predict(X_test)
    return y_pred , rfecv
    



