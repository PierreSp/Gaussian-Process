# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:34:01 2017

@author: pierre
"""
import numpy as np
import matplotlib.pyplot as plt


class GaussianProcess:
    def __init__(self):
        self.init = 1

    def prior_sampling(mini=0, maxi=4, stepsize=0.001, sigma=2.5, l=0.2):
        dp = np.arange(mini, maxi, stepsize)
        K = np.arange(float(dp.size * dp.size)).reshape(dp.size, dp.size)
        for i in range(dp.size):
            for j in range(i, dp.size):
                K[i, j] = np.log(sigma**2) + (-1*(dp[i]-dp[j])**2 / (2*l**2))
                K[j, i] = K[i, j]
        mean = np.array(np.repeat(0, dp.size))
        rs = np.random.RandomState(5)
        x1 = rs.multivariate_normal(mean, np.exp(K), 10).T
        plt.plot(dp, x1)

    def calcK(self, dp1, dp2, sigma=2.5, l=0.2):
        K = np.arange(float(dp1.size * dp2.size)).reshape(dp1.size, dp2.size)
        for i in range(dp1.size):
            for j in range(i, dp2.size):
                K[i, j] = np.log(sigma**2) + (-1*(dp1[i]-dp2[j])**2 / (2*l**2))
                K[j, i] = K[i, j]
        return K

    def calcKstar(self, dp1, dp2, sigma=2.5, l=0.2):
        K = np.arange(float(dp1.size * dp2.size)).reshape(dp1.size, dp2.size)
        for i in range(dp1.size):
            for j in range(dp2.size):
                K[i, j] = np.log(sigma**2) + (-1*(dp1[i]-dp2[j])**2 / (2*l**2))
        return K

    def gauss_fun(self, fun, tpoints=10, mini=0, maxi=4, stepsize=0.008,
                  sigma=2.5, l=0.2):
        dpstar = np.arange(mini, maxi, stepsize)
        dp = np.linspace(mini, maxi, tpoints)
        ftrain = fun(dp)
        K = self.calcK(dp, dp)
        Kstar = self.calcKstar(dp, dpstar)
        Kstarstar = self.calcK(dpstar, dpstar)
        mean = (np.transpose(Kstar).dot(
            np.linalg.inv(K))).dot(ftrain.transpose())
        cov = Kstarstar - (np.transpose(
            Kstar).dot(np.linalg.inv(K))).dot(Kstar)
        rs = np.random.RandomState(5)
        x1 = rs.multivariate_normal(mean, np.exp(cov), 10).T
        plt.plot(dp, x1)

    def Gaussianpoints(self, X, fun, minval=0, maxval=4, nsample = 100,
                       l=0.6, sigma=1):

        f = fun(X)
        K = np.empty([X.size, X.size])
        for i in range(X.size):
            for j in range(X.size):
                K[i, j] = (sigma**2) * np.exp(-1*(X[i]-X[j])**2 / (2*l**2))


        Xs = np.linspace(minval, maxval, nsample)
        Kss = np.empty([Xs.size, Xs.size])
        for i in range(Xs.size):
            for j in range(Xs.size):
                Kss[i, j] = (sigma**2) * np.exp(-1*(Xs[i]-Xs[j])**2 / (2*l**2))

        Ks = np.empty([X.size, Xs.size])
        for i in range(X.size):
            for j in range(Xs.size):
                Ks[i, j] = (sigma**2) * np.exp(-1*(X[i]-Xs[j])**2 / (2*l**2))

        mu = 0 + Ks.transpose().dot(np.linalg.inv(K)).dot(f)
        Kov = Kss - Ks.transpose().dot(np.linalg.inv(K)).dot(Ks)

        rs = np.random.RandomState(5)
        x1 = rs.multivariate_normal(mu, Kov, 20).T

        plt.plot(Xs, x1, X, fun(X), 'ro', Xs, fun(Xs), '--')


X = np.array([0, 0.25, 0.5, 0.6, 1, 2.1, 2.2, 2.75, 2.8])
fun = lambda x: np.sin(x)**2
GP = GaussianProcess()
GP.Gaussianpoints(X, fun)
