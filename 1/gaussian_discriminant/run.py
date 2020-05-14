
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from plot_ex1 import plot_ex1, figure


mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
mu0 = np.array([[-3,-3]]).T
Sigma0 = np.array([[1,0], [0,1]])
mu1 = np.array([[3,3]]).T
Sigma1 = np.array([[1,0], [0,1]])
phi = .5
# end answer
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line', 1)
figure


# %%
mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
mu0 = np.array([[0,0]]).T
Sigma0 = np.array([[5,0], [0,1]])
mu1 = np.array([[-2,0]]).T
Sigma1 = np.array([[3,0], [0,1]])
phi = .3
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line (one side)', 2)
figure


# %%
mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
mu0 = np.array([[0,0]]).T
Sigma0 = np.array([[1,0], [0,1]])
mu1 = np.array([[-1,-1]]).T
Sigma1 = np.array([[1,0.5], [0.5,1]])
phi = .5
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Parabalic', 3)
figure


# %%
mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
mu0 = np.array([[0,0]]).T
Sigma0 = np.array([[1.5,0], [0,1.5]])
mu1 = np.array([[0,0]]).T
Sigma1 = np.array([[5,4], [4,5]])
phi = .5
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Hyperbola', 4)
figure


# %%
mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
mu0 = np.array([[0,0]]).T
Sigma0 = np.array([[1,0], [0,1]])
mu1 = np.array([[0,0]]).T
Sigma1 = np.array([[5,4], [4,5]])
phi = .5
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Two parallel lines.', 5)
figure


# %%
mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
mu0 = np.array([[0,0]]).T
Sigma0 = np.array([[1,0], [0,1]])
mu1 = np.array([[0,0]]).T
Sigma1 = np.array([[5,0], [0,5]])
phi = .5
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Circle', 6)
figure


# %%
mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
mu0 = np.array([[0,0]]).T
Sigma0 = np.array([[0.9,0], [0,0.9]])
mu1 = np.array([[0,0]]).T
Sigma1 = np.array([[5,4], [4,5]])
phi = .5
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Ellipsoid', 7)
figure


# %%
mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0


# begin answer
mu0 = np.array([[0,0]]).T
Sigma0 = np.array([[1,0], [0,1]])
mu1 = np.array([[0,0]]).T
Sigma1 = np.array([[1,0], [0,1]])
phi = .5
# end answer


plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'No boundary', 8)
figure


# %%
figure.savefig("hw1_gaussian_discriminant.png")


# %%


