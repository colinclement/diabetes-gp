import numpy as np
import scipy.linalg as spl
from scipy.spatial.distance import cdist

class GPR(object):
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel

    def fit(self, y_obs, x_obs, noise):
        self.y_obs = y_obs
        self.x_obs = x_obs
        self.n = len(y_obs)
        self.K = cdist(x_obs[:,None], x_obs[:,None], self.kernel)
        self.L = spl.cho_factor(self.K + noise*np.eye(self.n))
        self.alpha = self._invKproduct(y_obs)

    def transform(self, x_test):
        K_obs_test = cdist(self.x_obs[:,None], x_test[:,None], self.kernel)
        self.testmean = K_obs_test.T.dot(self.alpha)
        K_test_test = cdist(x_test[:,None], x_test[:,None], self.kernel)
        self.testvar = (K_test_test - 
                        K_obs_test.T.dot(self._invKproduct(K_obs_test)))
        return self.testmean, self.testvar
        
    def fit_transform(self, y_obs, x_obs, noise, x_test):
        self.fit(y_obs, x_obs, noise)
        return self.transform(x_test)

    def _invKproduct(self, x):
        return spl.cho_solve(self.L, x)

