import numpy as np
import scipy
import scipy.stats

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    def gaussian_pdf(X, Mu, Sigma):
        Mu = np.expand_dims(Mu, -1)
        t = X - Mu
        Sigma_r = np.linalg.inv(Sigma)
        p = t.T @ Sigma_r @ t
        p = np.diag(p)
        p = np.exp(-p * .5)
        return p / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    p = np.array([gaussian_pdf(X, Mu=Mu[:,i], Sigma=Sigma[:,:,i]) for i in range(K)]).T
    p *= Phi
    p /= np.expand_dims(np.sum(p, axis=1), -1)
    # end answer
    
    return p

if __name__ == "__main__":
    M = 2
    N = 100
    K = 10
    X = np.zeros((M, N))
    Mu = np.zeros((M, K))
    Sigma = np.zeros((M, M, K))
    for i in range(K):
        Sigma[:, :, i] = np.identity(M)
    Phi = np.zeros((K))

    p = gaussian_pos_prob(X, Mu, Sigma, Phi)
    print(p.shape)