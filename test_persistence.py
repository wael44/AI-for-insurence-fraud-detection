# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:04:50 2018

@author: wael
"""

from sklearn.externals import joblib
from data import import_data



clf = joblib.load('pfa.pkl')
X=import_data()
Y=clf.predict(X.iloc[4:5,1:])
print(Y)