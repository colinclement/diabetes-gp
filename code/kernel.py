import numpy as np
from scipy.spatial.distance import cdist


class Kernel(object):
    def __init__(self, p = []):
        self.updateposp(p)
        if not len(p) == self.N_p:
            raise TypeError("len(p) should be {}".format(self.N_p))

    def __call__(self, x1, x2):
        return self.ev(x1, x2, self.argp) 

    def gradk(self, x1, x2):
        return self.evgrad(x1, x2, self.argp)

    @staticmethod
    def getposp(p):
        return np.log(1+np.exp(p))

    @staticmethod
    def getargp(posp):
        return np.log(np.exp(posp)-1)

    @property
    def p(self):
        return self.posp
   
    @property
    def argp(self):
        return Kernel.getargp(self.posp)

    def updatep(self, p):
        self.posp = Kernel.getposp(p.copy())
    
    def updateposp(self, posp):
        self.posp = posp.copy()

    @staticmethod
    def ev(x1, x2, p):
        """
        Evaluate K(x1, x2) with parameters p
        input:
            x1, x2 - numpy arrays of shape (n1, d), (n2, d)
                     for data dimension d
            p - numpy array of length N_p, with log(1+exp(parameters))
        returns:
            K(x1, x2) - (n1, n2)-shaped numpy array of covariances
        """
        raise(NotImplemented, "No Kernel evaluation defined")

    @staticmethod
    def evgrad(x1, x2, p):
        """
        Evaluate d_K(x1, x2) with parameters p
        input:
            same as self.ev
        returns:
            d_K(x1,x2)_dp - (n1, n2, N_p)-shaped numpy array
        """
        raise(NotImplemented, "No grad Kernel evaluation defined")
    
   
class SquareExponential(Kernel):
    N_p = 2 #[Variance, Decorrelation length]
    def __init__(self, p = np.array([1.,1.])):
        super(SquareExponential, self).__init__(p)

    @staticmethod
    def ev(x1, x2, p):
        v, l = Kernel.getposp(p)
        k = lambda xi, xj: v*np.exp(-0.5*(xi-xj).T.dot(xi-xj)/l**2)
        return cdist(x1[:,...,None], x2[:,...,None], k)

    @staticmethod
    def evgrad(x1, x2, p):
        v, l = Kernel.getposp(p)
        dv, dl = 1./(1+np.exp(-p))
        ev = lambda xi, xj: v*np.exp(-0.5*(xi-xj).T.dot(xi-xj)/l**2)
        evd1 = lambda x1, x2: dl * ev(x1, x2)*(x1-x2).T.dot(x1-x2)/l**3
        k = cdist(x1[:,...,None], x2[:,...,None], ev)
        dk1 = cdist(x1[:,...,None], x2[:,...,None], evd1)
        return np.dstack([dv/v * k, dk1])
                        

class LocalPeriodic(Kernel):
    N_p = 4 #[Variance, Period, Periodic Decorr Length, Decorr length]
    def __init__(self, p = np.array([1., 1., 1., 1.])):
        super(LocalPeriodic, self).__init__(p)

    @staticmethod
    def ev(x1, x2, p):
        def krn(xi, xj):
            d = xi-xj
            return p[0]*(np.exp(- 2*(np.sin(0.5*d/p[1])/p[2])**2
                                - 0.5*d.T.dot(d)/p[3]**2))
        return cdist(x1[:,...,None], x2[:,...,None], krn)

    @staticmethod
    def evgrad(x1, x2, p):
        k = LocalPeriodic.ev(x1, x2, p)
        evk = lambda d: p[0]*(np.exp(- 2*(np.sin(0.5*d/p[1])/p[2])**2
                                     - 0.5*d.T.dot(d)/p[3]**2))
        def dkrn_d1(xi, xj):
            d = xi-xj
            k = evk(d)
            return 2*k*d*np.cos(d/(2*p[1]))*np.sin(d/(2*p[1]))/(p[1]*p[2])**2
        dk1 = cdist(x1[:,...,None], x2[:,...,None], dkrn_d1) 
        def dkrn_d2(xi, xj):
            d = xi-xj
            k = evk(d)
            return 4*k*np.sin(d/(2*p[1]))**2/p[2]**3
        dk2 = cdist(x1[:,...,None], x2[:,...,None], dkrn_d2) 
        def dkrn_d3(xi, xj):
            d = xi-xj
            k = evk(d)
            return d**2*k/p[3]**3
        dk3 = cdist(x1[:,...,None], x2[:,...,None], dkrn_d3) 

        return np.dstack([k/p[0], dk1, dk2, dk3])
    


