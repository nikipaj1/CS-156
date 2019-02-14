
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[20]:

class LogisticRegression:
    def __init__(self,training_data = None, testing_data = None):
        self.training_data = training_data
        self.testing_data = testing_data
        self.w_vect = np.array([0.,0.,0.])
        self.point1 = [random.uniform(-1,1),random.uniform(-1,1)]
        self.point2 = [random.uniform(-1,1),random.uniform(-1,1)]
        
    def target(self,point):
        x1,y1 = self.point1
        x2,y2 = self.point2
        
        m = (y2-y1)/(x2-x1)
        x = point[1]
        y = point[2]
        # equation y = m(x-x1) + y1
        
        if (m*(x-x1) + y1 < y): return 1
        else: return -1
        
    def gradient(self,point,y):
        numerator = np.multiply(-y, point)
        logist = 1 + np.exp(y * np.dot(self.w_vect.T,point))
        
        # obtain gradient for signle point
        return np.divide(numerator, logist)
        
    def train(self):
        y_vect = []
        repeats = 1
        delta_w = 1
        
        # generate target outputs
        for i in self.training_data:
            y_vect.append(self.target(i))
            
        # add parameters
        step_size = 0.01
    
        while delta_w > 0.01:
            
            w_prev = self.w_vect
            
            # randomize the arrays
            train = self.training_data
            combined = list(zip(train, y_vect))
            random.shuffle(combined)
            train[:], y_vect[:] = zip(*combined)
            
            # training algorithm
            for i in range(len(y_vect)):
                grad = self.gradient(train[i],y_vect[i])
                
                self.w_vect = self.w_vect - step_size * grad
                
            repeats += 1
            
            # change weights difference modulus
            delta_w = np.linalg.norm(self.w_vect - w_prev)
        
        return repeats
            
    def test(self):
        # write test later
        return
            
            
def run(run_number,data_size, e_in = False):
    error = 0
    repeats = 0

    for i in range(run_number):
        training_data = [[1,random.uniform(-1,1),random.uniform(-1,1)] for i in range(data_size)]
        if e_in:
            testing_data = training_data
        else:
            testing_data = [[1,random.uniform(-1,1),random.uniform(-1,1)] for i in range(data_size)]

        logr = LogisticRegression(training_data,testing_data)

        repeats += logr.train()
    
    return repeats / float(run_number)
    
            
if __name__ == "__main__":
    print(run(10,100))
    # returns 358 repeats - answer [a]
            


# In[ ]:




# In[ ]:



