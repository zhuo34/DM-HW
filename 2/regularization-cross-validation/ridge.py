import numpy as np


def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    X = np.vstack((np.ones(N), X))

    tmp = X @ X.T + lmbda * np.identity(P+1)
    if lmbda > 0:
      w = np.linalg.inv(tmp) @ X @ y.T
    else:
      w = np.linalg.pinv(tmp) @ X @ y.T
    return w
