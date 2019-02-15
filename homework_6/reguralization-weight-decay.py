
# coding: utf-8

# In[1]:

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[98]:

class Regularization:
    def __init__(self, X_train,y_train,X_test = None,y_test = None,w_vect = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        if w_vect is None:
            w_vect = np.array([0.,0.,0.])
        else: 
            w_vect = w_vect
    
    def hypothesis(self,point):
        return np.sign(np.dot(point,self.w_vect))
    
    def transform(self,point):
        x1,x2 = point[1],point[2]
        return 1,x1,x2,x1**2,x2**2,x1*x2,np.abs(x1-x2),np.abs(x1+x2)
    
    def lr_train(self,non_linear = False):
        # y same for both cases
        y_vect = np.array([y_train]).T
        print(y_vect)
        if not non_linear:
            X_mat = np.array(X_train)
            self.w_vect = np.dot(np.linalg.pinv(X_mat),y_vect)
        else:
            Z_mat = np.array([self.transform(point) for point in X_train])
            print(Z_mat)
            self.w_vect = np.dot(np.linalg.pinv(Z_mat),y_vect)
            
    def test(self,non_linear = False):
        if not non_linear:
            y_pred = [np.dot(point,self.w_vect) for point in X_test]
        else:
            # rewrite code to define Z_mat
            y_pred = [np.dot(point, self.w_vect) for point in Z_mat]

        # count all different members
        error = np.sum(y_pred != y_test)
        
        return error / float(len(y_test))
        
def run(e_in=False,non_linear = False):
    
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
        
    lr = Regularization(X_train,y_train)
    
    lr.lr_train(non_linear)
    error = lr.test(non_linear)
    
    return error
        
if __name__ == "__main__":
    print(run(True,True)) # returns 0.004 
    print(run(False,True))


# In[ ]:



