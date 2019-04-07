import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
sys.path.append('../')
from public.algorithms import LinearRegression


def nonlinear_transform(point):
    x1 = point[1]
    x2 = point[2]
    return [1,x1,x2,x1**2,x2**2,x1*x2,np.abs(x1-x2),np.abs(x1+x2)]

lra = LinearRegression(data_set = [])

# uploading data
df1 = pd.read_csv("in.txt",delim_whitespace=True,header=None)
df2 = pd.read_csv("out.txt",delim_whitespace=True, header = None)
X_train = [[1,df1[1][i],df1[0][i]] for i in range(len(df1)-1)]
y_train = [df1[2][i] for i in range(len(df1)-1)]

X_test = [[1,df2[1][i],df2[0][i]] for i in range(len(df2)-1)]
y_test = [df2[2][i] for i in range(len(df2)-1)]

# transformation to higher dimensions
lra.X_train = [nonlinear_transform(point) for point in X_train]
lra.y_train = y_train

lra.X_test = [nonlinear_transform(point) for point in X_test]
lra.y_test = y_test

# training non-linear
lra.train_real_data()

sns.scatterplot(np.array(X_train)[:,1], np.array(X_train)[:,2], hue = y_train)
plt.show()
sns.scatterplot(np.array(X_test)[:,1], np.array(X_test)[:,2], hue = y_test)

error_in = lra.test_real_data(E_in = True) / float(len(lra.X_train))
error_in = np.around(error_in, decimals = 3)
print("The results for the case without regularization")
print("The in-sample error is: " + str(error_in))
error_out = lra.test_real_data(E_in = False) / float(len(lra.X_test))
error_out = np.around(error_out, decimals = 3)

print("The out-of-sample error is: " + str(error_out) + "\n")
# answer is [a]

lra.w = None

lra.train_real_data(k = -3)

error_in = lra.test_real_data(E_in = True) / float(len(lra.X_train))
error_in = np.around(error_in, decimals = 3)
print("The results for the case with regularization, k = -3")
print("The in-sample error is: " + str(error_in))
error_out = lra.test_real_data(E_in = False) / float(len(lra.X_test))
error_out = np.around(error_out, decimals = 3)
print("The out-of-sample error is: " + str(error_out) + "\n")


lra.w = None

lra.train_real_data(k = -1)

error_in = lra.test_real_data(E_in = True) / float(len(lra.X_train))
error_in = np.around(error_in, decimals = 3)
print("The results for the case with regularization, k = -1")
print("The in-sample error is: " + str(error_in))
error_out = lra.test_real_data(E_in = False) / float(len(lra.X_test))
error_out = np.around(error_out, decimals = 3)
print("The out-of-sample error is: " + str(error_out))

# this example shows that with the correct lambda selection we were able to 
# decrease the out-of-sample error