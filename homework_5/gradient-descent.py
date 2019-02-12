
# coding: utf-8

# In[36]:

import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy.spatial import distance


# In[24]:

def gradient():
    u,v = point[0],point[1]
    grad_u = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v) + 2*v*np.exp(-u))
    grad_v = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
    
    return np.array([grad_u, grad_v])

def error_fun():
    # initial is [1,1]
    u,v = point[0], point[1]
    return (u*np.exp(v) - 2*v*np.exp(-u))**2


# In[28]:

steps = 0
tolerance = 10.0 **(-14)
learn_rate = 0.1

point = np.array([1,1])

print(error_fun())

plot_y = []
# gradient descent
while error_fun() > tolerance:
    plot_y.append([point[0],point[1]])
    grad = gradient()
    point = point -  grad*learn_rate
    
    steps +=1


# In[29]:

print(str(steps)) # returns 10 steps to descend


# In[40]:

plot_y[-1]


# In[48]:

# now repeat for coordinate descent
steps = 0
tolerance = 10.0 **(-14)
learn_rate = 0.1
point = np.array([1.0,1.0])

count = 0

while count < 15:
    grad = gradient()
    point[0] = point[0] - grad[0] * learn_rate
    
    grad = gradient()
    point[1] = point[1] - grad[1] * learn_rate
    
    count +=1
    
print(error_fun()) # correct answer [a]


# In[ ]:



