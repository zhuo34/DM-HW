import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    X = np.vstack((np.ones(N), X))
    alpha = 1.
    while True:
        y_ = np.sign(w.T @ X)
        _, mistake_idx = np.where(y != y_)
        if len(mistake_idx) == 0:
            break
        w += alpha * np.expand_dims(y[0, mistake_idx[0]] * X[:, mistake_idx[0]], -1)
        iters += 1

    return w, iters