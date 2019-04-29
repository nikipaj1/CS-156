import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
    c0 = None
    for i in checks:
        ovr_target = one_vs_rest(i, y_train)
        model = SVC(C = 0.01, kernel='poly',degree=2,coef0=1, gamma='auto')
        model.fit(X_train, ovr_target)
        
#        for j in range(X_train.shape[0]):
#            if ovr_target[j] == 1:
#                c0 = plt.scatter(X_train[j,0],X_train[j,1],c='r', s = 50, marker='+')
#            elif ovr_target[j] == -1:
#                c1 = plt.scatter(X_train[j,0],X_train[j,1],c='g', s = 50, marker='x')
#
#       
#        plt.legend([c0, c1], [str(i),'rest'])
#        x_min, x_max = X_train[:, 0].min() - 1, X_train[:,0].max() + 1
#        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
#        xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
#        Z = model.predict(np.c_[xx.ravel(),  yy.ravel()])
#        Z = Z.reshape(xx.shape)
#        plt.contour(xx, yy, Z)
#        plt.title('Support Vector Machine Decision Surface')
#        plt.show()
#        c0,c1=None,None
        
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
        model = SVC(C = c, kernel='poly', degree=2,coef0=1, gamma = 'auto')
        model.fit(X_tr,y_tr)
        
        prediction_train = model.predict(X_tr)
        prediction_test = model.predict(X_tst)
        
        error_in = len(prediction_train[prediction_train != y_tr]) / len(y_tr)
        error_out = len(prediction_test[prediction_test != y_tst]) / len(y_tst)
    
        errors_in.append([c, np.around(error_in, decimals=3)])
        errors_out.append([c, np.around(error_out, decimals = 3)])
    
    return errors_in, errors_out

def question7_8(C = [0.0001,0.001,0.01,0.1,1], first = 1, second = 5):
    results = []
      
    X_tr,y_tr = one_vs_one(first, second, y_train ,X_train)
    parameter_candidates = [
            {'C': C,'kernel':['poly']}
        ]    
    
    for i in range(100):
        
        clf = GridSearchCV(estimator = SVC(coef0=1,degree=2,gamma='auto'),cv=10,param_grid = parameter_candidates,n_jobs = -1)
        clf.fit(X_tr,y_tr)
        results.append(clf.best_score_)
    
    error = 1 - np.average(results)
    best_estimator = clf.best_estimator_
    return error, best_estimator

def question9(C=[0.01,1,100,1E4,1E6],first = 1, second = 5):
    results = []
    X_tr,y_tr = one_vs_one(first, second, y_train ,X_train)
    X_tst,Y_tst = one_vs_one(first, second, y_test, X_test)
    
    parameter_candidates = [
            {'C':C, 'kernel':['rbf']
            }]
    
    for i in range(100):
        clf = GridSearchCV(estimator= SVC(gamma='auto'),cv=10,param_grid = parameter_candidates, n_jobs = -1)
        clf.fit(X_tr,y_tr)
        results.append(clf.best_score_)
    error = 1 - np.average(results)
    best_estimator = clf.best_estimator_
    
    return error, best_estimator

def question10(C = [0.01,1,100,1E4,1E6], first = 1, second = 5):
    results = []
    
    X_tr,y_tr = one_vs_one(first, second, y_train ,X_train)
    X_tst,Y_tst = one_vs_one(first, second, y_test, X_test)
    
    for c in C:
        model = SVC(C=c,kernel='rbf',gamma = 'auto')
        model.fit(X_tr,y_tr)
        
        prediction = model.predict(X_tst)
        
        error = len(prediction[prediction != Y_tst]) / float(len(Y_tst))
        
        results.append([c,error])
    
    return results
    
    
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
#    
    
    # training and testing data
    X_train = df_train[['x1','x2']].values
    y_train = df_train['y'].values
    X_test = df_test[['x1','x2']].values
    y_test = df_test['y'].values
    
    # visualise numbers of each digit
    sns.countplot(y_train)
    plt.xlabel("digits")
    plt.ylabel('count')
    plt.title("graph for digit count")
    plt.savefig("training-data-count.png", dpi = 500)
#    
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
    print('results for question 6: ' + 'in errors: ' + str(question56(C=C,first=2,second=5)[0]) + 'errors out: ' + str(question56(C=C,first=2,second=5)[1]) + '\n')
    
    print(question7_8())
    # most frequent C used is 0.01 with the closest error to 0.005
    
    print(question9())
    # best c = 0.01 with E_in = 0.00384
    
    print(question10())
    # best are 100 and 10000, but 100 is chosen because smaller
    
    
    
    
