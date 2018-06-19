# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:46:20 2018

@author: wael
"""

from sklearn.cross_validation import train_test_split

# Function to split the dataset
def splitdataset(data):
 
    # Seperating the target variable
    
    
    X = data.loc[: ,data.columns != "FraudFound_P"]
    Y = data.loc[: ,data.columns == "FraudFound_P"]
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
     
    return X, Y, X_train, X_test, y_train, y_test

