
# coding: utf-8

# In[14]:

import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set()
from pylab import savefig


# In[72]:

class SineTrain:
    ''' linear regression with a function h(x) = ax
    finding a'''
    def __init__(self, X, a = None):
        self.a = a
        self.X = X
    
    def hypothesis(self, point):
        return np.sign(self.a*point[0]) 
        
    def train_nointercept(self, X_train):
        numerator = [point[0]*point[1] for point in X_train]
        numerator = np.sum(numerator)
        
        denominator = [point[0]**2 for point in X_train]
        denominator = np.sum(denominator)
        
        self.a = numerator / float(denominator)
        
def target(point):
    return np.sin(np.pi*point)
        
def run(run_number, X_size):
    a_vals = []
    X = [[random.uniform(-1,1)] for i in range(X_size)]
    for i in X:
        i.append(target(i[0]))

    
    for i in range(run_number):

        model = SineTrain(X = X)
        
        # two points
        point1,point2 = random.sample(model.X, 2)

        model.train_nointercept(X_train = [point1,point2])
        a_vals.append(model.a)
        
        #     x = [point[0] for point in model.X]
        #     y = [point[1] for point in model.X]


        #     sns_plot = sns.scatterplot(x,y)   

        #     figure = sns_plot.get_figure()
        #     figure.savefig("output.png")

    print("Question 4: The average hypothesis g-bar(X) = " + str(np.round(np.average(a_vals),decimals=2)) + "*X") # returns 1.49, which gives answer [e]
    
    a_avg = np.average(a_vals)
    g_bar = []
    for i in model.X:
        g_bar.append(a_avg*i[0])
    g_bar = np.array([g_bar]).T
    
    f_x = np.array([[pt[1]] for pt in model.X])
    bias = np.mean((g_bar - f_x)**2)
    
    print("Question 5: The bias is " + str(np.round(bias, decimals=2))) # closest answer is 0.3 i.e. [b]
    
    a = np.array([a_vals]).T

    variance = np.mean((a - g_bar.T[:,len(a)])**2)
    print(variance)
    
if __name__ == "__main__":
    run(100,1000)


# In[ ]:



