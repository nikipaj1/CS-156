import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
sys.path.append('../')
from public.algorithms import LinearRegression


def nonlinear_transform(point):
    x1 = point[1]
    x2 = point[2]
    return [1,x1,x2,x1**2,x2**2,x1*x2,np.abs(x1-x2),np.abs(x1+x2)]

def validation(validation_size = 10 ,k_values = [-3,-2,-1,0,1,2,3]):
    results_val = []
    results_test = []
    
    lra.X_train = X_train[:len(X_train)-validation_size]
    lra.X_test = X_train[len(X_train)-validation_size:]
    
    lra.y_train = y_train[:len(X_train)-validation_size]
    lra.y_test = y_train[len(X_train)-validation_size:]
    
    print(len(lra.y_train), len(lra.y_test))
    for k in k_values:
        lra.w = None
        lra.train_real_data(k = k)
        error = lra.test_real_data()
        
        results_val.append([k, error / len(lra.X_test)])
        
        
    lra.X_test = X_test
    lra.y_test = y_test
    
    for k in k_values:
        lra.w = None
        lra.train_real_data(k = k)
        err_test = lra.test_real_data()
        results_test.append([k, err_test / len(lra.X_test)])
        
    return results_val, results_test

# loading data
df1 = pd.read_csv("in.txt",delim_whitespace=True,header=None)
df2 = pd.read_csv("out.txt",delim_whitespace=True, header = None)
X_train = [[1,df1[1][i],df1[0][i]] for i in range(len(df1)-1)]
y_train = [df1[2][i] for i in range(len(df1)-1)]

X_test = [[1,df2[1][i],df2[0][i]] for i in range(len(df2)-1)]
y_test = [df2[2][i] for i in range(len(df2)-1)]

# transformation to higher dimensions
X_train = [nonlinear_transform(point) for point in X_train]
X_test = [nonlinear_transform(point) for point in X_test]

lra = LinearRegression(data_set = [])
lra.X_train = X_train
lra.y_train = y_train

results_val, results_test = validation()

print(results_val)
print(results_test)
