import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../')
from public.algorithms import PLA


def run(data_size, run_number):
    
    repeats = 0
    error_count = 0
    rand_nums = []
    
    for num in range(5):
        rand_nums.append(random.randint(0, 1000)) 
    
    for i in range(run_number):
        data_set = [[1, random.uniform(-1,1),random.uniform(-1,1)] for i in range(2 * data_size)]
        
        pla = PLA(data_set=data_set)
        pla.train_test_split()
        
        train_target = [pla.target(i) for i in pla.X_train]
        test_target = [pla.target(i) for i in pla.X_test]
        
        repeats += pla.train()
        
        # Testing part
        error_count += (pla.test(E_in = False) / float(data_size * 0.5)) 
        
        slope = - (pla.w[0,0] / pla.w[2,0]) / (pla.w[0,0] / pla.w[1,0])
        intercept = - (pla.w[0,0] / pla.w[2,0])
            
        x = [-1,1]
        y = [(slope * i + intercept) for i in x]
                    
        if i in rand_nums and slope is not None: 
            ax1 = sns.scatterplot([x[1] for x in pla.X_train],[x[2] for x in pla.X_train],marker="x",hue=train_target, palette=["b","r"])
            sns.lineplot(x,y)
            ax1.set(xlabel = "$x_1$", ylabel = "$x_2$", title="Plot for training data " + str(i))
            plt.show()
            
            ax2 = sns.scatterplot([x[1] for x in pla.X_test],[x[2] for x in pla.X_test],marker="x",hue=test_target, palette=["g","y"])
            sns.lineplot(x,y)
            ax2.set(xlabel = "$x_1$", ylabel = "$x_2$", title="Plot for testing data " + str(i))
            plt.show()
    print("##############################################")
    print("For the data size of " + str(data_size) + " The results are: ") 
    print("The number of repeats is: " + str(repeats / 1000.))
    print("Error count for the testing set is: " + str(error_count/1000.))
    
    
if __name__ == "__main__":
    # results for question 7 and 8    
    run(data_size = 20, run_number = 1000)
    
    # results for question 9 and 10
    run(data_size = 200, run_number = 1000)