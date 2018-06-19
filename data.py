# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:40:58 2018

@author: wael
"""

from sklearn import preprocessing
from pymongo import MongoClient
from pandas.io.json import json_normalize

 #function to import the dataset
def import_data():
    client=MongoClient('wael-PC',27017)
    db = client.fraud_detection
    claims=db.pfa
    data_projection ={"$project": {"_id":0}}
    cursor = claims.aggregate([data_projection] )
    data1=list(cursor)
    data=json_normalize(data1)
    data.head()
    le = preprocessing.LabelEncoder()
    
    for i in range(33):
        if data.iloc[:,i].dtypes == object:
            le.fit(data.iloc[:,i].astype(str))
            data.iloc[:,i] = le.transform((data.iloc[:,i]).astype(str))
        else:
            pass
    
    return(data)  
