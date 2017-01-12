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
        self._calculated = 0

    def add_noise(self, noise):
        """Adds noise to the model

        Note:
            Affects only K

        Args:
            noise: Float which determins the amount of added noise

        Returns:
            None, only changes the diagonals of self.K

        """
        self.noise_level = noise
        self.K = self.K + (noise * np.identity(self.X.size))

    def _calc_kernel(self, x1, x2):
        """Calculates the kernel

        Note:
            Only works for quadratic K, as symmetry is used.
            Using the exp kernel function with the parameters
            self.sigma and self.l

        Args:
            x1: values of the x-values of the first set
            x2: values of the x-values of the second set

        Returns:
            K: Kernel matrix

        """
        K = np.zeros([x1.size, x2.size])
        for i in range(x1.size):
            for j in range(i, x2.size):
                K[i, j] = ((self.sigma**2) * np.exp(-1 * (x1[i] - x2[j])**2
                    / (2 * self.l**2)))
                K[j, i] = K[i, j]
        return K

    def _calc_kernel_nonquad(self, x1, x2):
        """Calculates the kernel

        Note:
            Using the exp kernel function with the parameters
            self.sigma and self.l

        Args:
            x1: values of the x-values of the first set
            x2: values of the x-values of the second set

        Returns:
            K: Kernel matrix

        """
        K = np.zeros([x1.size, x2.size])
        for i in range(x1.size):
            for j in range(x2.size):
                K[i, j] = ((self.sigma**2) * np.exp(-1 * (x1[i] - x2[j])**2
                    / (2 * self.l**2)))
        return K

    def calc_samples(self, nsamples=10):
        """Calculates samples from a multivariate gaussian

        Note:
            This function uses self.mean and self.cov for the sampling

        Args:
            nsamples: number of different samples to draw

        Returns:
            rs: samples from the gaussian distribution

        """
        if self._calculated > 0:
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
        self.mean = self.Ks.transpose().dot(np.linalg.inv(self.K)).dot(
            fun(self.X))
        self.cov = self.Kss - (self.Ks.transpose().dot(
            np.linalg.inv(self.K))).dot(self.Ks)
        self._calculated = 1

    def plot(self, nsamples, plot_variance=False):
        if (self._calculated == 0):
            print("No values calculates so far")
            return None
        elif (self._calculated == 1):
            yplot = self.calc_samples(nsamples)
            plt.plot(self.X_s, yplot, self.X, self.fun(self.X), 'ro')
            plt.plot(self.X_s, self.fun(self.X_s), ls='--', color="green",
                     lw=2)
            plt.plot(self.X_s, self.mean, ls='-.', color='gray', lw=2.5)
            if plot_variance:
                var = np.diagonal(self.cov)
                # Plot 96% confidence
                plt.fill_between(self.X_s, self.mean+1.96*np.sqrt(var), self.mean-1.96*np.sqrt(var),
                    alpha=0.2, edgecolor='#191919', facecolor='#737373', antialiased=True)
            return(plt.show())
        elif (self._calculated == 2):
            yplot = self.calc_samples(nsamples)
            plt.plot(self.X, yplot)
            return(plt.show())

    def prior_sampling(self):
        self.mean = np.array(np.repeat(0, self.X.size))
        self.cov = self._calc_kernel(self.X, self.X)
        self._calculated = 2


if __name__ == '__main__':
    X = np.random.uniform(0, 2, 5)
    fun = lambda x: np.sin(x*np.pi)**2
    Xs = np.linspace(0, 2, 200)
    GP = GaussianProcess(Xs, X, l=0.6)
    GP.cond_gaussian(fun)
    GP.plot(10, plot_variance=True)