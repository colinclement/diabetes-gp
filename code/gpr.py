import numpy as np
import scipy.linalg as spl


class GPR(object):
    """Perform a Gaussian Process Regression.

    The implementation follows Algorithm 2.1 in Rasmussen & Williams (2006). It
    uses Cholesky decomposition .

    input:
        kernel - a kernel object that can evaluate the covariance matrix
                between two sets of vectors and the gradients of that
                covariance matrix with respect to the hyperparameters, p, of
                the kernel, see code/kernel.py.

    methods:
        fit - perform the fit.
        transform - make predictions using that fit.
        fit_transorm - convenience to fit and predict in one go.
        logmarg - compute the log marginal likelihood of the fit.
        grad_logmarg - compute the gradient of logmarg wrt hyperparameters, p.

    """
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel

    def fit(self, y_obs, x_obs, var):
        self.y_obs = y_obs
        self.x_obs = x_obs
        self.n = len(y_obs)
        self.K = self.kernel(x_obs, x_obs)
        self.var = var
        self._L = spl.cho_factor(self.K + var*np.eye(self.n))
        self.alpha = self._invKproduct(y_obs)

    def _invKproduct(self, x):
        return spl.cho_solve(self._L, x)
    
    def transform(self, x_test):
        K_obs_test = self.kernel(self.x_obs, x_test)
        self.testmean = K_obs_test.T.dot(self.alpha)
        K_test_test = self.kernel(x_test, x_test)
        self.testvar = (K_test_test - 
                        K_obs_test.T.dot(self._invKproduct(K_obs_test)))
        return self.testmean, self.testvar
        
    def fit_transform(self, y_obs, x_obs, var, x_test):
        self.fit(y_obs, x_obs, var)
        return self.transform(x_test)

    def logmarg(self, pvar):
        """Compute the log marginal likelihood of the fit.

        See Eq. 5.8 in Rasmussen & Williams (2006).

        input:
            pvar - array hyperparameters, whose last entry is the squared
                    variance of the observations
            
        """
        self.update_kernel_p(pvar[:-1], pvar[-1])

        lgmg = self.y_obs.dot(self.alpha) + \
               2*np.sum(np.log(self._L[0].diagonal())) + \
               self.n*np.log(2*np.pi)
        
        return -0.5*lgmg

    def grad_logmarg(self, pvar):
        """Compute the gradient of the log marginal likelihood wrt
        hyperparameters and the variance.

        See Eq. 5.9 in Rasmussen & Williams (2006).

        input:
            pvar - array hyperparameters, whose last entry is the squared
                    var of the observations

        """
        self.update_kernel_p(pvar[:-1], pvar[-1])

        Kder = self.kernel.gradk(self.x_obs, self.x_obs)

        Kder = np.dstack([Kder, np.eye(self.n)])

        grd = np.trace(np.outer(self.alpha, self.alpha).dot(Kder))
        grd -= np.trace(self._invKproduct(Kder))

        return 0.5*grd

    def update_kernel_p(self, p, var=None):
        self.kernel.p[:] = p.copy()
        var = var or self.var
        self.fit(self.y_obs, self.x_obs, var)


