# -*- coding: utf-8 -*-
"""
Created on Sat May  5 02:25:11 2018

@author: wael
"""

    


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from data import import_data
from splitdataset import splitdataset 
import operator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.metrics import confusion_matrix



data = import_data()
X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
print(list(X))
# sorting the futures in a list by their impact in the classification
def sort():
    f_list=dict()
    fc_list=dict()
    a=list(X)
    clf_object = MLPClassifier(hidden_layer_sizes=(13,13,13,13),max_iter=50)
    n_feats = X.shape[1]
    for i in range(n_feats):
        xi = X.iloc[:, i].reshape(-1,1)
        scores = cross_val_score(clf_object, xi, Y)
        f_list[i]= scores.mean()
        fc_list[i]= a[i]
    f_list = sorted(f_list.items() , key=operator.itemgetter(1) )   #sorted by occuracy 
    return f_list,fc_list

#function to generate lists of features 
def lists_generator(v,c):
   
    L=list()
    for i in range (32) :
        li=list()
        for j in range (i+1) :
            li.append(c[v[j][0]])
        L.append(li)
        L.reverse()
    return L


def main() :
    v,c=sort()
    Ls=lists_generator(v,c)
    ocuur_list=list()
    cm_list=list()
    
    model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=50)  
    for i in range(31) :
        model.fit(X_train.drop(Ls[i+1] , axis=1) , y_train)
        y_pred= model.predict(X_test.drop(Ls[i+1] , axis=1))
        ac=accuracy_score(y_test,y_pred)*100
        ocuur_list.append(ac)
        cm = confusion_matrix(y_test,y_pred)
        cm_list.append(cm)
    n_of_feat=list(range(1,32,1))
    plt.scatter(n_of_feat,ocuur_list,marker="o",color="blue")
    plt.title("performance ")
    plt.xlabel("number of features  ")
    plt.ylabel("occuracy")
    plt.plot(n_of_feat,ocuur_list,color="green",ls="-")
    plt.grid()
    print("best performance : " , max(ocuur_list)   )
    print("number of features  : " , ocuur_list.index(max(ocuur_list))+1   )  
    cmm = cm_list[ocuur_list.index(max(ocuur_list))]
    sns.heatmap(cmm,annot=True,fmt="d")
    plt.figure()


    
    
    
main()