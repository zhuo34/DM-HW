import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful
    N_test, P = x.shape
    y = np.zeros(N_test)
    for i in range(N_test):
        d = np.sum((x[i] - x_train) ** 2, axis=1)
        idx = np.argsort(d)[:k]
        y[i] = scipy.stats.mode(y_train[idx])[0][0]
    return y
