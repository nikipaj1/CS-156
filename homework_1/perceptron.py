# coding: utf-8

# In[76]:

import numpy as np
import random
import os
import matplotlib.pyplot as plt


# get_ipython().magic('matplotlib inline')


# In[ ]:

# code for perceptron in homework 1. Reused in later problems

def plot(data, x_line=None, y_line=None, name=None, w_slope=None, w_intercept=None):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    px, py, nx, ny = [], [], [], []

    for point in data:
        if target(point) == 1:
            px.append(point[1])
            py.append(point[2])
        else:
            nx.append(point[1])
            ny.append(point[2])

    x_line = np.linspace(-1., 1., 50)
    if w_slope is None:
        y_line = [slope * (point - x1) + y1 for point in x_line]
    else:
        y_line = [w_slope * point + w_intercept for point in x_line]

    plt.plot(px, py, 'xr', nx, ny, 'xb', x_line, y_line,'k')
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.savefig(name + ".png", dpi=100)
    plt.show()


def generate_target():
    point1 = (random.uniform(-1, 1), random.uniform(-1, 1))
    point2 = (random.uniform(-1, 1), random.uniform(-1, 1))

    x1, y1 = point1
    x2, y2 = point2

    slope = (y2 - y1) / (x2 - x1)
    return x1, y1, slope


def create_data(size, in_sample=False):
    training_data = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                     for i in range(size)]
    if in_sample:
        testing_data = training_data.copy()
    else:
        testing_data = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                        for i in range(size)]
    return training_data, testing_data


def target(point):
    x, y = point[1], point[2]
    global x1, y1, slope
    if y > slope * (x - x1) + y1:
        return 1
    else:
        return -1


def hypothesis(point):
    global w
    return np.sign(np.dot(point, w))


def train():
    misclassified = []
    repeats = 0
    global w
    while True:
        for point in training_data:
            # find all misclassified points
            real = target(point)
            if hypothesis(point) != real:
                repeats += 1
                misclassified.append([point, real])
            # modify weights to classify correctly, we apply PLA only to separable data 
            # otherwise would have to use pocket algorithm
        if not misclassified:
            break
        else:
            # choosing a random point to train the perceptron
            point, y_n = random.choice(misclassified)
            w += y_n * np.array([point]).T
            misclassified = []
    return repeats


def test():
    err = 0

    for point in testing_data:
        if hypothesis(point) != target(point):
            err += 1

    return err / float(len(testing_data))


# In[ ]:

# question 7
runs = 10
data_size = 10
repeats = 0

for i in range(runs):
    x1, y1, slope = generate_target()
    training_data, testing_data = create_data(size=data_size, in_sample=True)

    plot(data=training_data, name="target-" + str(i))
    w = np.array([[0.,0.,0.]]).T
    repeat = train()
    repeats += repeat
    slope_w = -(w[1] / w[2])
    intercept_w = -(w[0] / w[2])

    plot(data=training_data, name="trained-" + str(i), w_slope=slope_w, w_intercept=intercept_w)

print("number of repeats: " + str(repeats / float(runs)))
