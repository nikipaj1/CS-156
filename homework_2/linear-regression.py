import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../')
from public.algorithms import PLA, LinearRegression


def run56(data_size, run_number):
    
    repeats = 0
    error_count = 0
    rand_nums = []
    
    for num in range(3):
        rand_nums.append(random.randint(0, 1000))
        
    # Questions 5 and 6

    for i in range(run_number):
        data_set = [[1, random.uniform(-1,1),random.uniform(-1,1)] for i in range(data_size)]
        
        lra = LinearRegression(data_set = data_set)
        #pla = PLA(data_set = data_set, point1 = lra.point1, point2 = lra.point2)
        lra.train_test_split(train = 0.09090909)
        
        train_target = [lra.target(i) for i in lra.X_train]
        test_target = [lra.target(i) for i in lra.X_test]
        
        lra.train()
        
        error_count += (lra.test(E_in = True) / float(len(lra.X_train)))
        
        slope = - (lra.w[0,0] / lra.w[2,0]) / (lra.w[0,0] / lra.w[1,0])
        intercept = - (lra.w[0,0] / lra.w[2,0])
            
        x = [-1,1]
        y = [(slope * i + intercept) for i in x]
                    
        if i in rand_nums and slope is not None: 
            ax1 = sns.scatterplot([x[1] for x in lra.X_train],[x[2] for x in lra.X_train],marker="x",hue=train_target, palette=["b","r"])
            sns.lineplot(x,y)
            ax1.set(xlabel = "$x_1$", ylabel = "$x_2$", title="Plot for training data " + str(i))
            plt.show()
            
            ax2 = sns.scatterplot([x[1] for x in lra.X_test],[x[2] for x in lra.X_test],marker="x",hue=test_target, palette=["g","y"])
            sns.lineplot(x,y)
            ax2.set(xlabel = "$x_1$", ylabel = "$x_2$", title="Plot for testing data " + str(i))
            plt.show()
            
    print("##############################################")
    print("For the data size of " + str(data_size) + " The results are: ") 
    print("Error count for the testing set is: " + str(np.around(error_count/1000., decimals= 7)))

def run7(data_size, run_number):
    repeats = 0
    rand_nums = []
    
    for num in range(3):
        rand_nums.append(random.randint(0, 1000))
        

    for i in range(run_number):
        data_set = [[1, random.uniform(-1,1),random.uniform(-1,1)] for i in range(data_size)]
        
        lra = LinearRegression(data_set = data_set)
        #pla = PLA(data_set = data_set, point1 = lra.point1, point2 = lra.point2)
        lra.train_test_split()
        
        train_target = [lra.target(i) for i in lra.X_train]
        
        lra.train()
        
        pla = PLA(data_set = lra.data_set, w = lra.w, point1 = lra.point1, point2 = lra.point2)
        pla.X_train = lra.X_train
        
        repeats += pla.train()
        
        slope = - (pla.w[0,0] / pla.w[2,0]) / (pla.w[0,0] / pla.w[1,0])
        intercept = - (pla.w[0,0] / pla.w[2,0])
            
        x = [-1,1]
        y = [(slope * i + intercept) for i in x]
                    
        if i in rand_nums and slope is not None: 
            ax1 = sns.scatterplot([x[1] for x in pla.X_train],[x[2] for x in pla.X_train],marker="x",hue=train_target, palette=["b","r"])
            sns.lineplot(x,y)
            ax1.set(xlabel = "$x_1$", ylabel = "$x_2$", title="Plot for training data " + str(i))
            plt.show()
            
    print("##############################################")
    print("For the data size of " + str(data_size) + " The results are: ") 
    print("PLA repeats until convergence: " + str(repeats / run_number))

    
if __name__ == "__main__":
    
    # Questions 5 and 6
    run56(data_size=1100, run_number = 1000)
    
    # Question 7
    run7(data_size = 20, run_number= 1000)