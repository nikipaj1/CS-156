import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

def one_vs_rest(choice, y_vector):
    target = []    
    for i in range(len(y_vector)):
        if int(choice) == int(y_vector[i]):
            target.append(1)
        else:
            target.append(-1)
    # use same X since no numbers were disregarded
    return target

def one_vs_one(choice_pos,choice_neg, y_vector, X_vector):
    y = []
    X = []
    for i in range(len(y_vector)):
        if int(choice_pos) == int(y_vector[i]):
            X.append(X_vector.tolist()[i])
            y.append(1)
        elif int(choice_neg) == int(y_vector[i]) :
            X.append(X_vector.tolist()[i])
            y.append(-1)
        else:
            continue
    
    return np.array(X), np.array(y)


def question2_3(checks):
    errors_in = []
    
    for i in checks:
        ovr_target = one_vs_rest(i, y_train)
        model = SVC(C = 0.01, kernel='poly',degree=2,coef0=1, gamma='auto')
        model.fit(X_train, ovr_target)
        prediction = model.predict(X_train)
        error = len(prediction[prediction != ovr_target]) / len(ovr_target)
        errors_in.append([i, np.around(error,decimals = 3), np.sum(model.n_support_.tolist())])
        
    return errors_in


def question56(C, first = 5,second = 6):
    errors_in = []
    errors_out = []
    for c in C:
        X_tr,y_tr = one_vs_one(first, second, y_train ,X_train)
        X_tst, y_tst = one_vs_one(first, second,y_test, X_test)
        model = SVC(C = c, kernel='poly', degree=2,coef0=1,gamma='auto')
        model.fit(X_tr,y_tr)
        prediction_train = model.predict(X_tr)
        prediction_test = model.predict(X_tst)
        
        error_in = len(prediction_train[prediction_train != y_tr]) / len(y_tr)
        error_out = len(prediction_test[prediction_test != y_tst]) / len(y_tst)
    
        errors_in.append([c, np.around(error_in, decimals=3)])
        errors_out.append([c, np.around(error_out, decimals = 3)])
    
    return errors_in, errors_out

def question7_8(C = [0.0001,0.001,0.01,0.1,1], first = 1, second = 5):
    errors = []
    temp = []
    for c in C:
       errors.append([c,0,0])
    
    X_tr,y_tr = one_vs_one(first, second, y_train ,X_train)
    rkf = RepeatedKFold(n_splits = 10, n_repeats=100)
    
    for train_index, test_index in rkf.split(X_tr):
        X_training, X_testing = X_tr[train_index], X_tr[test_index]
        y_training, y_testing = y_tr[train_index], y_tr[test_index]
        for c in errors:
            model = SVC(C=c[0], kernel='poly',degree=2,coef0=1,gamma='auto')
            model.fit(X_training, y_training)
            prediction_test = model.predict(X_testing)

            error_xval = len(prediction_test[prediction_test != y_testing]) / (len(y_testing)* 100)
            
            c[1] += error_xval
            temp.append(error_xval)

        min_index = temp.index(min(temp))
        temp = []
        errors[min_index][2] += 1   
            
    return errors
 
if __name__ == '__main__':
    
    # reading data
    df_train = pd.read_csv("features.train.txt", delim_whitespace = True,names=['y','x1','x2'])
    df_test = pd.read_csv('features.test.txt', delim_whitespace=True,names=['y','x1','x2'])
    
    # visualize the digits
    sns.lmplot('x1','x2',hue='y', data=df_train,fit_reg=False,markers='x')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('scatterplot for digits')
    plt.savefig('numbers.png', dpi=500)
    
    # training and testing data
    X_train = df_train[['x1','x2']].values
    y_train = df_train['y'].values
    X_test = df_test[['x1','x2']].values
    y_test = df_test['y'].values
    
    checks = [0,2,4,6,8]
    print("results for question 2: " + str(question2_3(checks)))
    print('\n')
    checks = [1,3,5,7,9]
    print("results for question 3: " + str(question2_3(checks)))

    print('\n')
    C = [0.001,0.01,0.1,1]
    print("results for question 5: " + str(question56(C=C)))
    print('\n')
    C = [0.0001,0.001,0.01,1]
    print('results for question 6: ' + str(question56(C=C,first=2,second=5)) + '\n')
    
    print(question7_8())
    # most frequent C used is 0.001 with the closest error to 
    
    
    
    
    
