import numpy as np


def logistic_function(X, w):
    return 1. / (1 + np.exp(-w.T @ X))

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    learning_rate = 0.1
    X = np.vstack((np.ones(N), X))

    n_iter = 0
    while True:
        lf = logistic_function(X, w)
        nll = -np.average(y*np.log(lf) + (1-y)*np.log(1-lf))
        gradient = -(X @ (y - lf).T) / N
        # print(n_iter, nll, np.sum(gradient**2))
        if np.sum(gradient**2) < 0.01:
            break
        w -= learning_rate * gradient
        n_iter += 1

    return w
