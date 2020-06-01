import numpy as np


def logistic_function(X, w):
    return 1. / (1 + np.exp(-w.T @ X))


def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    learning_rate = 10
    X = np.vstack((np.ones(N), X))

    n_iter = 0
    while n_iter < 50:
        lf = logistic_function(X, w)
        nll = -np.average(y*np.log(lf) + (1-y)*np.log(1-lf)) + lmbda * np.sum(w**2)
        gradient = (-(X @ (y - lf).T) + lmbda * w) / N
        # print(n_iter, nll, np.sum(gradient**2))
        # if np.sum((gradient)**2) < .1:
            # break
        w -= learning_rate * gradient
        n_iter += 1
    return w
