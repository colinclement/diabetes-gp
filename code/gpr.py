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
    The following methods take as their arguments 'log' parameters to
    enforce positivity. They are actually of the form log(1+exp(x))
        logmarg - compute the log marginal likelihood of the fit.
        grad_logmarg - compute the gradient of logmarg wrt hyperparameters, p.

    """
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel

    def _invKproduct(self, x):
        """
        Compute inv(K).dot(x) for array x.
        """
        return spl.cho_solve(self._L, x)
    
    def _update_kernel_p(self, argp, var):
        """
        Update kernel and noise parameters. Uses log(1+exp(x)) 
        parameters to impose positivity
        """
        self.kernel.updatep(argp)
        self.fit(self.y_obs, self.x_obs, self.kernel.getposp(var))
 
    def fit(self, y_obs, x_obs, var = 0.):
        """
        Use observatives to train Gaussian Process.
        input:
            y_obs : array of observed values of shape (n)
            x_obs : array of measurement points of shape (n, dimension)
            (optional)
            var : Variance of measurements
        """
        self.y_obs = y_obs
        self.x_obs = x_obs
        self.n = len(y_obs)
        self.K = self.kernel(x_obs, x_obs)
        self.var = var
        self._L = spl.cho_factor(self.K + var*np.eye(self.n))
        self.alpha = self._invKproduct(y_obs)
   
    def transform(self, x_test):
        """
        Compute predictions of fitted model using Cholesky decomp.
        input:
            x_test: array of test points of shape (n, dimension)
        returns:
            mean, covariance of test points
        """
        if not hasattr(self, '_L'):
            raise RuntimeError("Must fit model first!")
        K_obs_test = self.kernel(self.x_obs, x_test)
        self.testmean = K_obs_test.T.dot(self.alpha)
        K_test_test = self.kernel(x_test, x_test)
        self.testvar = (K_test_test - 
                        K_obs_test.T.dot(self._invKproduct(K_obs_test)))
        return self.testmean, self.testvar
        
    def fit_transform(self, y_obs, x_obs, var, x_test):
        """
        Both fit and transform.
        """
        self.fit(y_obs, x_obs, var)
        return self.transform(x_test)

    def logmarg(self, pvar):
        """Compute the log marginal likelihood of the fit.

        See Eq. 5.8 in Rasmussen & Williams (2006).

        input:
            pvar - array hyperparameters, whose last entry is the squared
                   var of the observations. These values will be modified
                   by the function log(1+exp(x)) to enforce positivity
        """
        p, var = pvar[:-1], pvar[-1]
        self._update_kernel_p(p, var)

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
                   var of the observations. These values will be modified
                   by the function log(1+exp(x)) to enforce positivity

        """
        p, var = pvar[:-1], pvar[-1]
        self._update_kernel_p(p, var)

        Kder = np.dstack([self.kernel.gradk(self.x_obs, self.x_obs),
                          np.eye(self.n)/(1.+np.exp(-var))])

        grd = np.trace(np.outer(self.alpha, self.alpha).dot(Kder))
        grd -= np.trace(self._invKproduct(Kder))

        return 0.5*grd

    def nll_dnll(self, pvar):
        """
        Helper function to hand to minimization routine.
        input:
            pvar: [kernel params, variance], will be modified by
                  log(1+exp(x)) to enforce positivity.
        output:
            negative logmarg, negative grad_logmarg
        """
        return -self.logmarg(pvar), -self.grad_logmarg(pvar)




