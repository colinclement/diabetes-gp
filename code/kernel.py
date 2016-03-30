import numpy as np
from scipy.spatial.distance import cdist


class Kernel(object):
    def __init__(self, p = []):
        self.p = np.array(p)
        self.N_p = len(p)
    
    def __call__(self, x1, x2):
        return self.ev(x1, x2, self.p) 

    def gradk(self, x1, x2):
        return self.evgrad(x1, x2, self.p)

    @staticmethod
    def ev(x1, x2, p):
        """
        Evaluate K(x1, x2) with parameters p
        input:
            x1, x2 - numpy arrays of shape (n1, d), (n2, d)
                     for data dimension d
            p - numpy array of length N_p
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
    def __init__(self, p = np.array([1.,1.])):
        super(SquareExponential, self).__init__(p)

    @staticmethod
    def ev(x1, x2, p):
        k = lambda xi, xj: p[0]*np.exp(-0.5*(xi-xj).T.dot(xi-xj)/p[1]**2)
        return cdist(x1[:,...,None], x2[:,...,None], k)

    @staticmethod
    def evgrad(x1, x2, p):
        ev = lambda x1, x2: SquareExponential.ev(x1, x2, p)
        evd1 = lambda x1, x2: ev(x1, x2)*(x1-x2).T.dot(x1-x2)/p[1]**3
        k = cdist(x1[:,...,None], x2[:,...,None], ev)
        dk1 = cdist(x1[:,...,None], x2[:,...,None], evd1)
        return np.dstack([k/p[0], dk1])

                        

