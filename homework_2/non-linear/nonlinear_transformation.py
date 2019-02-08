
# coding: utf-8

# In[18]:

import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[34]:

# code for nonlinear transformation
# part of linear regression code is reused here

def plot(data, plot_name, non_linear=False, x_line = None, y_line = None, w_slope = None, w_intercept = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('$x_1$'), ax.set_ylabel('$x_2$')
    
    positive = []
    negative = []

    for point in data:
        if target(point,non_linear=non_linear) == 1:
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

# generate either linear target or non-linear
def target(point, non_linear = False):
    x, y = point[1], point[2]
    
    if not non_linear:
        #classify according to the target line
        return 1 if y > slope*(x-x1) + y1 else -1
    else:
        # classify according to the non-linaer function
        return np.sign(x**2 + y**2 - 0.6)

def hypothesis(point):
    # make sure w is global
    return np.sign(np.dot(point,w))

def add_noise(y, noise = 0.1):
    flip_num = len(testing_data)*noise
    # use set to avoid duplications and no need of checking
    flipped = set()
    
    while len(flipped) < flip_num:
        choice = random.randint(0,len(y)-1)
        # add the index of flipped point
        flipped.add(choice)
        y[choice][0] *= -1
    return y

def line_train(noise = False,  non_linear = False):

    x_mat = np.array(training_data)
    y_vect = np.array([[target(i, non_linear=non_linear)] for i in training_data])
    
    if noise is not None:
        y_vect = add_noise(y_vect)
    
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
    
def test(non_linear = False, noise=0.1):
    miss = 0
    
    target_y = np.array([[target(i,non_linear)] for i in testing_data]) 
    hyp_y = np.sign(np.dot(testing_data, w))
    
    if noise is not None:
        target_y = add_noise(target_y)
    
    # counting the misclassified points
    for i in range(len(target_y)):
        if target_y[i] != hyp_y[i]: 
            miss +=1
    return miss / float(len(target_y))

def run(runs, data_size, noise = 0.1, in_sample = True, perceptron = False, non_linear = False):
    test_error = 0
    for i in range(runs):  
        # generate training and testing data
        global training_data, testing_data
        training_data = [[1.,random.uniform(-1,1), random.uniform(-1,1)] for i in range(data_size)]
        
        if in_sample:
            testing_data = training_data.copy()
        else:
            testing_data = [[1.,random.uniform(-1,1), random.uniform(-1,1)] for i in range(data_size)]
        
        if not non_linear:
            # generate target function
            point1 = (random.uniform(-1, 1), random.uniform(-1, 1))
            point2 = (random.uniform(-1, 1), random.uniform(-1, 1))

            global x1,y1,x2,y2,slope
            x1, y1 = point1
            x2, y2 = point2
            slope = (y2 - y1) / (x2 - x1)
        
        # plot target function if linearly separable data
        if not non_linear and i == 1: 
            plot(data = training_data, plot_name = "target" + str(i),non_linear=non_linear)
        
        global w
        w = np.array([0.,0.,0.]).T
        
        # plot from weights vector before training
        if i == 1: plot(data = training_data, plot_name="before_trainig" + str(i),non_linear = non_linear, w_slope = 0, w_intercept = 0)
        
        # train linear regression and plot the outcome    
        line_train()
        
        w_slope = -(w[1]/w[2])
        w_intercept = -(w[0]/w[2])
        
        if i == 1: plot(training_data,"after-train" + str(i),non_linear=non_linear,w_slope = w_slope, w_intercept=w_intercept)
        
        if perceptron:
            # train the perceptron and plot the outcome    
            pla_rep = perceptron_train()

            if i == 1 and w[2] != 0: plot(data = training_data, plot_name="after-perceptron" + str(i),non_linear=non_linear, w_slope = w_slope, w_intercept = w_intercept)
        
        test_error += test(noise = noise, non_linear=non_linear)
        
    if perceptron:
        # return the number of iterations required to train perceptron
        print("The number of steps required to train the percpetron is " + str(pla_rep / float(runs)))
    
    print("The {}-sample error is ".format("in" if in_sample else "out-of") + str(test_error / float(runs)))


# In[37]:

# homework 2

# problem 8
run(runs=10,data_size=1000,noise = 0.1, non_linear=True)
# returns in-sample error of 0.514, correct answer 0.5 [d]


# In[ ]:



