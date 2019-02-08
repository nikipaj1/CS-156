
# coding: utf-8

# In[39]:

import numpy as np
import math
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[57]:

# code for nonlinear transformation
# part of linear regression code is reused here
# the code builds further to implement perceptorn, linear-regression and non-linear transformation

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

def hypothesis(mat):
    # make sure w is global
    return np.sign(np.dot(mat, w))

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
    
    if not non_linear:
        x_mat = np.array(training_data)
        y_vect = np.array([[target(i, non_linear=non_linear)] for i in training_data])
    
    else:
        z_mat = np.array([[1, pt[1], pt[2],pt[1]*pt[2], pt[1]**2,pt[2]**2] for pt in training_data])
        y_vect = np.array([[target(i,non_linear=non_linear)] for i in training_data])
    
    if noise is not None:
        y_vect = add_noise(y_vect)
    
    global w
    if not non_linear:
        w = np.dot(np.linalg.pinv(x_mat),y_vect)
    else:
        w = np.dot(np.linalg.pinv(z_mat),y_vect)

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
#     hyp_y = np.sign(np.dot(testing_data, w))
    if not non_linear:
        hyp_y = hypothesis(testing_data)
    else:
        z_mat = np.array([[1, pt[1], pt[2],pt[1]*pt[2], pt[1]**2,pt[2]**2] for pt in testing_data])
        hyp_y = hypothesis(z_mat)
    
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
        if non_linear:
            w = np.array([0.,0.,0.,0.,0.,0.]).T
        else:
            w = np.array([0.,0.,0.]).T
        
        # plot from weights vector before training
        if i == 1: 
            plot(data = training_data, plot_name="before_trainig" + str(i),non_linear = non_linear, w_slope = 0, w_intercept = 0)
        
        # train linear regression and plot the outcome    
        line_train(noise = 0.1, non_linear=non_linear)
        
        w_slope = -(w[1]/w[2])
        w_intercept = -(w[0]/w[2])
        
        if i == 1: 
            # train linear algorithm and plot the outcome
            plot(training_data,"after-train" + str(i),non_linear=non_linear,w_slope = w_slope, w_intercept=w_intercept)
            print("The equation is g ={0} + {1}x1 + {2}x2 + {3}x1x2 + {4}x1**2 + {5} x2**2".format(w[0],w[1],w[2],w[3],w[4],w[5]))
            
        if perceptron:
            # train the perceptron and plot the outcome    
            pla_rep = perceptron_train()

            if i == 1 and w[2] != 0: 
                plot(data = training_data, plot_name="after-perceptron" + str(i),non_linear=non_linear, w_slope = w_slope, w_intercept = w_intercept)
                
        test_error += test(noise = noise, non_linear=non_linear)
        
    if perceptron:
        # return the number of iterations required to train perceptron
        print("The number of steps required to train the percpetron is " + str(pla_rep / float(runs)))
    
    print("The {}-sample error is ".format("in" if in_sample else "out-of") + str(test_error / float(runs)))
    


# In[61]:

# homework 2

# problem 8
run(runs=10,data_size=1000,noise = 0.1, non_linear=True)
# returns in-sample error of 0.514 if no non-linear is used, correct answer 0.5 [d]


# In[62]:

# problem 9 and problem 10
run(runs = 1, data_size = 1000, noise = 0.1, in_sample=False, non_linear=True)

# returns 
# The equation is g =[-0.97936285] + [ 0.00239167]x1 + [ 0.01255421]x2 + [-0.01206515]x1x2 + [ 1.49537485]x1**2 + [ 1.58193335] x2**2
# closest is answer [a]

# The out-of-sample error is 0.121 closest 0.1, answer [0.1]


# In[ ]:



