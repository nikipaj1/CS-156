
# coding: utf-8

# In[9]:

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[47]:

class Regularization:
    def __init__(self, X_train,y_train,X_test = None,y_test = None,w_vect = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # in the non-linear case, will define Z_mat
        self.Z_mat = None
        
        if w_vect is None:
            w_vect = np.array([0.,0.,0.]).T
        else: 
            w_vect = w_vect
    
    def transform(self,data_set):
        arr = []
        for point in data_set:
            x1,x2 = point[1],point[2]
            arr.append([1,x1,x2,x1**2,x2**2,x1*x2,np.abs(x1-x2),np.abs(x1+x2)])
        return arr
    
    def lr_train(self):
        y_vect = np.array([self.y_train]).T

        self.Z_train = self.transform(self.X_train)
        self.w_vect = np.dot(np.linalg.pinv(self.Z_train),y_vect)
            
    def test(self):
        error = 0
        self.Z_test = self.transform(self.X_test)
        y_pred = np.sign(np.dot(self.Z_test, self.w_vect))

        # count all different members
        for i in range(len(y_pred)):
            if y_pred[i] != self.y_test[i]:
                error += 1
        return error
        
def run(e_in=False):
    
    # load training and testing data
    df1 = pd.read_table("in.dta.txt",delim_whitespace=True,header=None)
    df2 = pd.read_table("out.dta.txt",delim_whitespace=True, header = None)
    X_train = [[1,df1[1][i],df1[0][i]] for i in range(len(df1)-1)]
    y_train = [df1[2][i] for i in range(len(df1)-1)]
    
    if not e_in:
        # load testing data
        X_test = [[1,df2[1][i],df2[0][i]] for i in range(len(df2)-1)]
        y_test = [df2[2][i] for i in range(len(df2)-1)]
        
    else:
        # training data is used for testing
        X_test = X_train
        y_test = y_train
        
    reg = Regularization(X_train,y_train,X_test,y_test)
    
    reg.lr_train()
    error = reg.test()
    
    return error / float(len(X_test))
        
if __name__ == "__main__":
    print(run(True)) # returns 0.0294 
    print(run(False)) # returns 0.0803 closest is 0.08


# In[ ]:



