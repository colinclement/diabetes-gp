import numpy as np
import scipy.linalg as spl

class GPR(object):
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel

    def fit(self, y_obs, x_obs, noise):
        self.y_obs = y_obs
        self.x_obs = x_obs
        self.n = len(y_obs)
        self.K = self.kernel(x_obs, x_obs)
        self.noise = noise
        self._L = spl.cho_factor(self.K + noise*np.eye(self.n))
        self.alpha = self._invKproduct(y_obs)

    def transform(self, x_test):
        K_obs_test = self.kernel(self.x_obs, x_test)
        self.testmean = K_obs_test.T.dot(self.alpha)
        K_test_test = self.kernel(x_test, x_test)
        self.testvar = (K_test_test - 
                        K_obs_test.T.dot(self._invKproduct(K_obs_test)))
        return self.testmean, self.testvar
        
    def fit_transform(self, y_obs, x_obs, noise, x_test):
        self.fit(y_obs, x_obs, noise)
        return self.transform(x_test)

    def logmarg(self, pnoise):
        self.update_kernel_p(pnoise[:-1], pnoise[-1])

        lgmg = self.y_obs.dot(self.alpha) + \
                np.prod(self._L[0].diagonal())**2 +\
                self.n*np.log(2*np.pi)
        
        return -0.5*lgmg

    def grad_logmarg(self, pnoise):
        self.update_kernel_p(pnoise[:-1], pnoise[-1])

        Kder = self.kernel.gradk(self.x_obs, self.x_obs)

        Kder = np.dstack([Kder, np.eye(self.n)])

        grd = np.trace(np.outer(self.alpha, self.alpha).dot(Kder))
        grd -= np.trace(self._invKproduct(Kder))

        return 0.5*grd

    def update_kernel_p(self, p, noise=None):
        self.kernel.p = p
        noise = noise or self.noise
        self.fit(self.y_obs, self.x_obs, noise)

    def _invKproduct(self, x):
        return spl.cho_solve(self._L, x)

