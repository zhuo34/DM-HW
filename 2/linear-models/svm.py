import numpy as np
import scipy
import scipy.optimize


def svm(X, y, c=0):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0
    X = np.vstack((np.ones(N), X))

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # target = lambda w: np.sum(w[1:, 0]**2) / 2
    # print(np.sum(w[1:]**2) / 2)
    def target(w, P, c):
        return np.sum(w[1:(P+1)]**2) / 2 + c * np.sum(w[(P+1):])
    def constraint(w, X, y):
        return (y * (w[:(P+1)].T @ X) + w[(P+1):] - 1)[0]
    def constraint2(w):
        return w[(P+1):]
    constrains = [
        {'type': 'ineq', 'args': (X, y), 'fun': lambda w, X, y: constraint(w, X, y)},
        {'type': 'eq' if c==0 else 'ineq', 'fun': lambda w: constraint2(w)}
    ]
    w = np.ones((P + 1 + N, 1)) / (P + 1 + N)
    w = scipy.optimize.minimize(target, w, args=(P, c), constraints=constrains).x
    # print(w)
    w = w[:(P+1), np.newaxis]

    tmp = y * (w.T @ X) - 1 < 1e-9
    num = np.sum(tmp)

    return w, num
