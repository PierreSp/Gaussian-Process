# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:34:01 2017

@author: pierre
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class GaussianProcess():
    """Gaussian Process simulaton for 2D data.

    This class calculates a gaussain process using the exponential kernel
    with length parameter l and variance parameter sigma. More informations
    about the background can be found in the github readme file

    Args:
        X_s (np_array): x-values of points to estimate
        X (np_array): x-values of training points
        l (float): length variable in the exp kernel
        sigma (float): variance in the exp kernel

    """
    def __init__(self, X_s, X, l=0.6, sigma=1):
        # Set class variables
        self.sigma = sigma
        self.l = l
        self.X = X
        self.X_s = X_s

        self.fun = lambda x: x
        self.__calculated = 0

    def add_noise(self, noise):
        self.noise_level = noise
        self.K = self.K + (noise * np.identity(self.X.size))

    def _calc_kernel(self, x1, x2):
        K = np.zeros([x1.size, x2.size])
        for i in range(x1.size):
            for j in range(i, x2.size):
                K[i, j] = ((self.sigma**2) * np.exp(-1 * (x1[i] - x2[j])**2
                    / (2 * self.l**2)))
                K[j, i] = K[i, j]
        return K

    def _calc_kernel_nonquad(self, x1, x2):
        K = np.zeros([x1.size, x2.size])
        for i in range(x1.size):
            for j in range(x2.size):
                K[i, j] = ((self.sigma**2) * np.exp(-1 * (x1[i] - x2[j])**2
                    / (2 * self.l**2)))
        return K

    def calc_samples(self, nsamples=10):
        if self.__calculated > 0:
            rs = np.random.RandomState(5)
            return rs.multivariate_normal(self.mean, self.cov, nsamples).T
        else:
            print("No values calculates so far")
            return None

    def cond_gaussian(self, fun):
        self.fun = fun
        self.K = self._calc_kernel(self.X, self.X)
        self.Ks = self._calc_kernel_nonquad(X, self.X_s)
        self.Kss = self._calc_kernel(self.X_s, self.X_s)
        self.mean = self.Ks.transpose().dot(np.linalg.inv(self.K)).dot(f)
        self.cov = self.Kss - (self.Ks.transpose().dot(
            np.linalg.inv(self.K))).dot(self.Ks)
        self.__calculated = 1

    def plot(self, nsamples):
        if (self.__calculated == 0):
            print("No values calculates so far")
            return None
        elif (self.__calculated == 1):
            yplot = self.calc_samples(nsamples)
            plt.plot(Xs, yplot, X, self.fun(X), 'ro', Xs, self.fun(Xs), '--')
            return(plt.show())
        elif (self.__calculated == 2):
            yplot = self.calc_samples(nsamples)
            plt.plot(self.X, yplot)
            return(plt.show())


    def prior_sampling(self):
        self.mean = np.array(np.repeat(0, dp.size))
        self.cov = self._calc_kernel(self.X, self.X)
        self.__calculated = 2


if __name__ == '__main__':
    X = np.random.uniform(0, 2, 10)
    fun = lambda x: np.sin(x)**2
    Xs = np.linspace(0, 2, 200)
    GP = GaussianProcess(Xs, X)
    GP.cond_gaussian(fun)