import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns 
import sys
sys.path.append('../')
from public.algorithms import PLA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def generate_data(data_size):
    data = [[1.,random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(data_size)]
    return data

def plot(pla, i):
    train_target = [pla.target(i) for i in pla.X_train]
    test_target = [pla.target(i) for i in pla.X_test]
        
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

rand_nums = []
for num in range(5):
        rand_nums.append(random.randint(0, 1000))

def question8(run_number = 1000, data_size = 20):
    error_pla = 0
    error_svm = 0
    num_SVM_better = 0
    
    for i in range(run_number):
        data = generate_data(data_size)
        
        pla = PLA(data_set = data)
        pla.train_test_split(train = 0.5)
        
        pla.train()
        
        
        # plotting
#        plot(pla, i)
        
        error_pla += pla.test(E_in = False) / float(len(pla.X_test))
    
        X_train = np.array(pla.X_train)
        X_test = np.array(pla.X_test)
        
        y_train = np.array([pla.target(point) for point in X_train])
        y_test = np.array([pla.target(point) for point in X_test])
    
        X_train = np.delete(X_train,0,1)
        X_test = np.delete(X_test,0,1)
        
        if np.array_equal(y_train,np.ones(len(pla.X_train))) or np.array_equal(y_train, -1*np.ones(len(pla.X_train))): 
            continue
        
        # C set to infinity (hard-margin) and linear kernel
        model = SVC(C = 100000000000000, kernel="linear")      
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        error_svm += len(y_test[y_test != predictions]) / float(len(pla.X_test))
        
        if error_svm < error_pla:
            num_SVM_better += 1
            
    error_pla /= float(run_number)
    error_svm /= float(run_number)
    num_SVM_better /= float(run_number)
    
    print("Question 8 PLA error is : " + str(np.around(error_pla,decimals = 3)))
    print("The error for the linear SVM, hard-margin is: " + str(np.around(error_svm, decimals=3)))
    print("Percentage of times SVM is better than pla: " + str(num_SVM_better))
    
    
    
if __name__ == "__main__":
    question8()
    print('\n')
    question8(data_size=200)