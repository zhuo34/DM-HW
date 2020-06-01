import numpy as np
import matplotlib.pyplot as plt


def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))

    X = np.vstack((np.ones(N), X))
    
    w = np.linalg.inv(X @ X.T) @ X @ y.T
    return w

def get_all_class(y):
    key = np.unique(y)
    return np.sort(key)

def get_indicator_matrix(y, y_class=None):
    N = y.shape[1]
    if y_class is None:
        y_class = get_all_class(y)
    Y = np.zeros((len(y_class), N))
    for i in range(len(y_class)):
        Y[i, y[0]==y_class[i]] = 1
    return Y

class LinearRegression:
    def __init__(self):
        self.trained = False

    def fit(self, X, y):
        self.y_class = get_all_class(y)
        self.P, self.K = X.shape[0], len(self.y_class)
        Y = get_indicator_matrix(y, self.y_class)
        self.W = np.zeros((self.P+1, self.K))
        for i in range(self.K):
            self.W[:, i] = linear_regression(X, np.expand_dims(Y[i, :], 0))[:, 0]
        self.trained = True

    def predict(self, X):
        if self.trained:
            y = self.W[1:].T @ X + self.W[0].T[:, np.newaxis]
            return self.y_class[np.argmax(y, axis=0)]