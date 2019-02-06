
# coding: utf-8

# In[1]:

import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[107]:

# code for linear regression, also reused in non-linear transformation

def plot(data, plot_name, x_line = None, y_line = None, w_slope = None, w_intercept = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('$x_1$'), ax.set_ylabel('$x_2$')
    
    positive = []
    negative = []
    
    for point in data:
        if target(point) == 1:
            positive.append([point[1], point[2]])
            
        else:
            negative.append([point[1], point[2]])
    
    x_line = np.linspace(-1,1,5)
    positive = np.array(positive)
    negative = np.array(negative)
    
    # if perceptron or linear regression used, plot weights line
    if w_slope is None:
        y_line = [slope* (point - x1) + y1 for point in x_line]
        plt.plot(positive[:,0], positive[:,1],'rx', negative[:,0],negative[:,1],'bx',x_line,y_line)
    # no model applied, plot target function
    else:
        y_line = [w_slope* point + w_intercept for point in x_line]
        plt.plot(positive[:,0], positive[:,1],'rx', negative[:,0],negative[:,1],'bx',x_line,y_line)
    
    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))
    plt.savefig(plot_name + ".png", dpi=100)

def target(point):
    x, y = point[1], point[2]
    return 1 if y > slope*(x-x1) + y1 else -1

def hypothesis(point):
    # make sure w is global
    return np.sign(np.dot(point,w))

def line_train():
    x_mat = np.array(training_data)
    y_vect = np.array([[target(i)] for i in training_data])
    global w
    w = np.dot(np.linalg.pinv(x_mat),y_vect)

def perceptron_train():
    misclassified = []
    repeats = 0
    global w
    
    while True:
        for point in training_data:
            # find all misclassified points
            real = target(point)
            if hypothesis(point) != real:
                misclassified.append([point, real])

        if not misclassified: break
        else:
            # choosing a random point to train the perceptron
            repeats += 1
            point, y_n = random.choice(misclassified)

            w += y_n * np.array([point]).T

            misclassified = []
        
    return repeats
    
def test():
    miss = 0
    
    target_y = np.array([[target(i)] for i in testing_data]) 
    hyp_y = np.sign(np.dot(testing_data, w))
    
    # counting the misclassified points
    for i in range(len(target_y)):
        if target_y[i] != hyp_y[i]: 
            miss +=1
    return miss / float(len(target_y))

def run(runs, data_size, in_sample = True, perceptron = False):
    for i in range(runs):
        
        # generate training and testing data
        global training_data, testing_data
        training_data = [[1.,random.uniform(-1,1), random.uniform(-1,1)] for i in range(data_size)]
        
        if in_sample:
            testing_data = training_data.copy()
        else:
            testing_data = [[1.,random.uniform(-1,1), random.uniform(-1,1)] for i in range(data_size)]
        
        # generate target function
        point1 = (random.uniform(-1, 1), random.uniform(-1, 1))
        point2 = (random.uniform(-1, 1), random.uniform(-1, 1))

        global x1,y1,x2,y2,slope
        x1, y1 = point1
        x2, y2 = point2
        slope = (y2 - y1) / (x2 - x1)
        
        if i == 1: plot(data = training_data, plot_name = "target" + str(i))
        
        global w
        w = np.array([0.,0.,0.]).T
        
        
        if i == 1: plot(data = training_data, plot_name="before_trainig" + str(i), w_slope = 0, w_intercept = 0)
        
        # train linear regression and plot the outcome    
        line_train()
        
        w_slope = -(w[1]/w[2])
        w_intercept = -(w[0]/w[2])
        
        if i == 1: plot(training_data,"after-train" + str(i),w_slope = w_slope, w_intercept=w_intercept)
        # train the perceptron and plot the outcome    
        pla_rep = perceptron_train()
        
        if i == 1 and w[2] != 0: plot(data = training_data, plot_name="after-perceptron" + str(i), w_slope = w_slope, w_intercept = w_intercept)
    
    # return the number of iterations required to seperate data with perceptron
    print("The number of steps required to train the percpetron is " + str(pla_rep / float(runs)))
        
        


# In[108]:

run(1000,10)


# In[24]:




# In[ ]:

# print(point[:,0])
# print(point[:,1])
# print(point)

# point = np.array([[random.uniform(-1,1), random.uniform(-1,1)] for i in range(10)])
# somthg = np.array([i[0],i[1]] for i in point)

