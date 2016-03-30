import numpy as np

def testKernelGrad(kernel, x1 = None, x2 = None, p = None, eps = 1E-7):
    """
    Test to make sure kernel is compute derivatives correctly
    input:
        kernel: Instance of Kernel object
        (optional):
        x1, x2: numpy arrays of test points to evaluate covariance
        p : parameters of kernel to evaluate derivatives at
        eps : step size of forward difference derivative
    return:
        bool : True if gradients are correct else False
    """
    x1 = x1 if x1 is not None else 10*np.random.rand(10)
    x2 = x2 if x2 is not None else 10*np.random.rand(10)
    if p is not None:
        kernel.p[:] = p.copy()
    mygrad = kernel.gradk(x1, x2)
    p0 = kernel.p.copy()
    k0 = kernel.ev(x1, x2, p0)
    dk = []
    for i in range(kernel.N_p):
        p1 = p0.copy()
        p1[i] += eps
        dk += [(kernel.ev(x1, x2, p1)-k0)/eps]
    dk = np.dstack(dk)
    return np.allclose(dk, mygrad)

