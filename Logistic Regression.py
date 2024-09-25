import numpy as np

# This program takes the Iris database from UC Irvine's Machine Learning Repository and does a logistic regression for classification of the species based on the four features below:
# sepal length, sepal width, petal length, and petal width with the expected output of species. The four features are continuous with units of centimeters (cm)
# The following program will utlize a one-vs-all classification strategy, the sigmoid function, and assumes the features are normally distributed. 
# The cost function utilized is the mean squared error. A stochastic gradient descent approach will be used.  
# Two classes are linearly inseperable and 1 class is linearly separable (as per the repository).
# Learning rate of gradient descent is experimentally found.

f = open("iris.data", "r")

# Since the data is ordered by species already, we can just append to separate the data so that feature normalization later is easier
X = [[], [], [], []] 

# import data into X arrays
for lines in f:
    data = lines
    data = data.split(",")
    for i in range(len(data)-1):
        X[i].append(data[i])


X = np.array(X, dtype=np.float64)
X_mean = np.zeros(4)
X_std = np.zeros(4)
np.mean(X, axis=1, out=X_mean, dtype=np.float64)
np.std(X, axis=1,ddof=1, out=X_std, dtype=np.float64)

# Feature normalization 
for i in range(len(X)):
    for j in range(len(X[i])):
        X[i][j] = (X[i][j] - X_mean[i]) / X_std[i]

X_class1 = [[], [], [], [], []]
X_class2 = [[], [], [], [], []]
X_class3 = [[], [], [], [], []]

#Separate the features by class (species). They are provided in batches of 50, so they can be separated by batches of 50. 
def separate_class(X):
    for k in range(50):
        X_class1[0].append(0)
        X_class2[0].append(0)
        X_class3[0].append(0)
    for i in range(len(X)):
        for j in range(len(X[i])):
            if j < 50:
                X_class1[i+1].append(X[i][j])
            elif j < 100:
                X_class2[i+1].append(X[i][j])
            else:
                X_class3[i+1].append(X[i][j])
    X_class1, X_class2, X_class3 = np.array(X_class1, dtype=np.float64), np.array(X_class2, dtype=np.float64), np.array(X_class3, dtype=np.float64)

separate_class(X)

theta = np.zeros((3, 5)) # We need 3 sets of parameters initialized to 0 for 3 binary classifications 

def hypothesis(theta, X):
    return np.matmul(theta, X)

def sigmoid(theta, X):
    ans = 1.0/(1.0 + np.exp(hypothesis(theta, X)))
    return ans

def cost_function_derivative(theta, X, Y, j):
    cost = (sigmoid(theta, X) - Y)*X[j]
    cost = cost/50
    return cost

# Smearing error appears in this function where for initial theta = 0 and X = X_class1 + X_class2, 
# the outputs are provided as 0. or - 0.
def grad_descent(theta, X, Y, alpha):
    temp = np.zeros((1, 50), dtype=np.float64), 
    for j in range(len(theta)):
        temp[j] = (theta[i] - alpha * cost_function_derivative(theta, X, Y, j))
    return temp 
