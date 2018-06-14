# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:04:13 2018

@author: 杨澎钢
"""

import numpy as np
import random
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import operator
def wk_knn(inX, dataSet, labels, k):  
    dataSetSize = dataSet.shape[0]  
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet    
    sqDiffMat = diffMat**2  
    sqDistances = sqDiffMat.sum(axis=1)    
    distances = sqDistances**0.5   
    sortedDistIndicies = distances.argsort()           
    classCount={}     
    w=[]        
    for i in range(k):  
        w.append(1/distances[sortedDistIndicies[i]])  
        voteIlabel = labels[sortedDistIndicies[i]]  
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + w[i]  
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  
    return sortedClassCount[0][0]

def knn(test, dataSet, labels, k):
    result = []
    for i in test:
        result.append(wk_knn(i,dataSet, labels, k))
    return result
    
if __name__ == "__main__":
    dim_list=[50,100,150,200,250]
    Max=0
    dim=0
    k=0
    plt.figure()
    train_label = np.fromfile("train/mnist_train_label",dtype=np.uint8)
    test_label = np.fromfile("test/mnist_test_label",dtype=np.uint8)
    train_index = random.sample(range(60000),3000)
    test_index = random.sample(range(10000),500)
    for i in dim_list:
        train_data = np.load("train_pca/"+str(i)+"_train.npy")
        test_data = np.load("test_pca/"+str(i)+"_test.npy")
        
        pre_train_data = train_data[train_index]
        pre_train_label = train_label[train_index]
        pre_test_data = test_data[test_index]
        pre_test_label = test_label[test_index]
        score_list = []
        for j in range(1,30,2):    
            pre = knn(pre_test_data,pre_train_data,pre_train_label,j)
            score = precision_score(pre_test_label, pre, average='micro')
            score_list.append(score)
            if score>Max:
                Max=score
                dim=i
                k=j
        print score_list
        plt.plot(range(1,30,2),score_list,label=str(i))
    print "dim: ",dim," k: ",k," score: ",Max
    plt.legend()
    plt.savefig("pic/knn.png")
    train_data = np.load("train_pca/"+str(dim)+"_train.npy")
    test_data = np.load("test_pca/"+str(dim)+"_test.npy")
    pre = knn(test_data,train_data,train_label,k)
    print classification_report(test_label,pre)
    
        