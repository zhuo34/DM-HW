import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    n, p = x.shape
    x_unique = np.unique(x, axis=0)
    # print(x_unique.shape)
    init_idx = np.arange(x_unique.shape[0])
    np.random.shuffle(init_idx)
    init_idx = init_idx[:k]

    ctrs = x_unique[init_idx]

    iter_ctrs = []
    x_all = np.expand_dims(x, axis=1)
    x_all = np.repeat(x_all, k, axis=1)
    while True:
        new_ctrs = np.zeros((k, p))
        count = np.zeros(k)
        d_all = np.sum((x_all - ctrs) ** 2, axis=2)
        idx = np.argmin(d_all, axis=1)
        # t = np.unique(idx)
        for i in range(k):
            new_ctrs[i] = np.average(x[idx==i], axis=0)
        # d = np.average(np.sqrt(np.sum((ctrs-new_ctrs)**2, axis=1)))
        if np.alltrue(ctrs==new_ctrs):
            break
        ctrs = new_ctrs
        iter_ctrs.append(ctrs)        
    iter_ctrs = np.array(iter_ctrs)

    return idx, ctrs, iter_ctrs
