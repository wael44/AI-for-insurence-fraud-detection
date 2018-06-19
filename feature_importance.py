# -*- coding: utf-8 -*-
"""
Created on Sat May  5 01:31:22 2018

@author: wael
"""

import matplotlib.pyplot as plt
import numpy as np



def imp(clf_object , x_train ):
    importances = clf_object.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_object.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(x_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    
    plt.figure(1, figsize=(14, 13))
    plt.title("Feature importances")
    plt.bar(range(x_train.shape[1]), importances[indices],
    color="g", yerr=std[indices], align="center")
    plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
    plt.xlim([-1, x_train.shape[1]])
    plt.show()