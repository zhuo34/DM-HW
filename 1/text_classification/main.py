# some basic imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
# %matplotlib inline

# %load_ext autoreload
# %autoreload 2


# ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = np.loadtxt('ham_train.csv', delimiter=',')
# spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = np.loadtxt('spam_train.csv', delimiter=',')
# N is the size of vocabulary.
N = ham_train.shape[0]
# There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034
num_spam_train = 3372
# Do smoothing
x = np.vstack([ham_train, spam_train]) + 1

# ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
i,j,ham_test = np.loadtxt('ham_test.txt').T
i = i.astype(np.int)
j = j.astype(np.int)
ham_test_tight = scipy.sparse.coo_matrix((ham_test, (i - 1, j - 1)))
ham_test = scipy.sparse.csr_matrix((ham_test_tight.shape[0], ham_train.shape[0]))
ham_test[:, 0:ham_test_tight.shape[1]] = ham_test_tight
# spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
i,j,spam_test = np.loadtxt('spam_test.txt').T
i = i.astype(np.int)
j = j.astype(np.int)
spam_test_tight = scipy.sparse.csr_matrix((spam_test, (i - 1, j - 1)))
spam_test = scipy.sparse.csr_matrix((spam_test_tight.shape[0], spam_train.shape[0]))
spam_test[:, 0:spam_test_tight.shape[1]] = spam_test_tight


from likelihood import likelihood
# TODO
# Implement a ham/spam email classifier, and calculate the accuracy of your classifier

# Hint: you can directly do matrix multiply between scipy.sparse.coo_matrix and numpy.array.
# Specifically, you can use sparse_matrix * np_array to do this. Note that when you use "*" operator
# between numpy array, this is typically an elementwise multiply.

# begin answer
l = likelihood(x)
print(l.shape)
# a
ratio = l[1] / l[0]
max10_idx = np.argsort(ratio)[:10]

import linecache
for i in max10_idx:
    s = linecache.getline('all_word_map.txt', i+1).strip()
    print(s)
# f = open('all_word_map.txt')


class SpamClassifier:
    def __init__(self):
        self.class_num = 2
        self.trained = False

    def train(self, x, sample_nums):
        self.likelihood = likelihood(x)
        self.prior = np.array(sample_nums) / np.sum(sample_nums)
        self.log_likelihood = np.log(self.likelihood)
        self.log_prior = np.log(self.prior)
        self.trained = True

    def __call__(self, x):
        if self.trained:
            K = x.shape[0]
            prediction = np.zeros((K,))
            for i in range(K):
                # print(x[i].shape)
                # print(self.log_likelihood.shape)
                # exit()
                print(x[1])
                exit()
                a = np.array(x[i]) * self.log_likelihood
                print(a.shape)
                a = np.sum(a, axis=1)
                print(a.shape, self.log_prior.shape)
                print(a)
                print(self.log_prior)
                log_posterior = a + self.log_prior
                prediction[i] = np.argmax(log_posterior)
                print(prediction[i])
            return prediction
        else:
            print('Please train first!')

clf = SpamClassifier()
clf.train(x, [num_ham_train, num_spam_train])
print(ham_test.shape)
prediction = clf(ham_test)
acc = (num_ham_train - np.sum(prediction)) / num_ham_train
print(acc)

# end answer