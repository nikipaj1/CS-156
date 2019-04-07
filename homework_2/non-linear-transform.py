import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../')
from public.algorithms import LinearRegression

def nonlinear_transform(point):
    return [1, point[1], point[2],point[1]*point[2], point[1]**2,point[2]**2]

def run8(data_size, run_number, noise):
    error_count = 0 
    rand_nums = set()
    
    for num in range(3):
        rand_nums.add(random.randint(0, 1000))
   
    for i in range(run_number):
        data_set = [[1, random.uniform(-1,1),random.uniform(-1,1)] for i in range(data_size)]
        
        lra = LinearRegression(data_set = data_set)
        lra.train_test_split()
        
        train_target = [lra.target_nonlinear(i) for i in lra.X_train]  
        test_target = [lra.target_nonlinear(i) for i in lra.X_test]
        
        lra.train(noise = noise)
        
        error_count = error_count + (lra.test_nonlinear(E_in = True, noise = 0.1) / float(len(lra.X_train)))
                
        slope = - (lra.w[0,0] / lra.w[2,0]) / (lra.w[0,0] / lra.w[1,0])
        intercept = - (lra.w[0,0] / lra.w[2,0])
            
        x = [-1,1]
        y = [(slope * i + intercept) for i in x]
                    
        if i in rand_nums and slope is not None: 
            ax1 = sns.scatterplot([x[1] for x in lra.X_train],[x[2] for x in lra.X_train],marker="x",hue=train_target, palette=["b","r"])
            ax1.set(xlabel = "$x_1$", ylabel = "$x_2$", title="Plot for training data " + str(i))
            sns.lineplot(x,y)
            plt.show()
            
            ax2 = sns.scatterplot([x[1] for x in lra.X_test],[x[2] for x in lra.X_test],marker="x",hue=test_target, palette=["g","y"])
            ax2.set(xlabel = "$x_1$", ylabel = "$x_2$", title="Plot for testing data " + str(i))
            plt.show()
            
    print("##############################################")
    print("For the data size of " + str(data_size) + " The results are: ") 
    print("Error count for the testing set is: " + str(np.around(error_count/1000., decimals= 7)))
        
def run910(data_size, run_number, noise):
    error_count = 0
    
    for i in range(run_number):
        data_set = [[1, random.uniform(-1,1),random.uniform(-1,1)] for i in range(data_size)]
        
        lra = LinearRegression(data_set = data_set)
        lra.train_test_split()
        
        # transform data to the requested dimensionality
        lra.X_train = [nonlinear_transform(point) for point in lra.X_train]        
        lra.X_test = [nonlinear_transform(point) for point in lra.X_test]
        
        lra.train(non_linear = True, noise = noise)
        error_count += (lra.test_nonlinear(E_in = False, noise = noise) / float(len(lra.X_train)))
    
    print("the coefficients for the g function are " + str(lra.w))    
    print("The out-of-sample error is: " + str(error_count / float(run_number)))
if __name__ == "__main__":
    run8(data_size=2000, run_number=1000, noise = 0.1)
    
    run910(data_size = 2000, run_number = 1000, noise = 0.1)