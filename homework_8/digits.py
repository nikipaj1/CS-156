import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC

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

# reading data
df_train = pd.read_csv("features.train.txt", delim_whitespace = True,names=['y','x1','x2'])
df_test = pd.read_csv('features.test.txt', delim_whitespace=True,names=['y','x1','x2'])

# visualize the digits
sns.lmplot('x1','x2',hue='y', data=df_train,fit_reg=False)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('scatterplot for digits')
plt.savefig('numbers.png', dpi=2000)


X_train = df_train[['x1','x2']].values
y_train = df_train['y'].values
X_test = df_test[['x1','x2']].values
y_test = df_test['y'].values


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


def question5(C, first = 5,second = 6):
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

if __name__ == '__main__':
    checks = [0,2,4,6,8]
    print("results for question 2: " + str(question2_3(checks)))
    print('\n')
    checks = [1,3,5,7,9]
    print("results for question 3: " + str(question2_3(checks)))

    print('\n')
    C = [0.001,0.01,0.1,1]
    print("results for question 5,6: " + str(question5(C=C)))
