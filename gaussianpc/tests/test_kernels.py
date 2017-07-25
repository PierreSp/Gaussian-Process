import pytest
import numpy as np
import pickle
from gaussianpc import gaussianprocess


def test_init():
    X = np.arange(0, 1, 0.1)
    fun = lambda x: np.sin(x * np.pi)**2
    Xs = np.linspace(0, 2, 200)
    GP = gaussianprocess(Xs, X, l=0.18)
    GP.cond_gaussian(fun)
    covmat = pickle.load("savecov.p")
    assert GP.cov == covmat