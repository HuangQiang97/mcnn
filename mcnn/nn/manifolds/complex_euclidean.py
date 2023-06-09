import torch
import numpy as np
from .manifold import Manifold


class ComplexEuclidean(Manifold):
    """
    Complex Euclidean manifold class with internal parameters satisfying constraints: A \in \mathbb C^{m\times n}
    For knowledge of complex Euclidean manifolds see: https://www.nicolasboumal.net/book/  P154
    """

    def __init__(self, height, width):

        super(Manifold, self).__init__()
        self._n = height
        self._m = width
        self._dim = 2*self._n*self._m
        self._size = torch.Size((height, width))

    def __str__(self):
        if self._k == 1:
            return "Complex Circle manifold St({}, {})".format(self._n, self._p)
        elif self._k >= 2:
            return "Product Complex Circle manifold St({}, {})^{}".format(self._n, self._p, self._k)

    def rand(self):
        x = torch.randn(self._n, self._m)+1j*torch.randn(self._n, self._m)
        return x/np.sqrt(2)

    def proj(self, z, u):
        """
        Projection of the Euclidean gradient onto the tangent plane
        """
        return u

    def inner(self, X, G1, G2):
        return torch.real(torch.sum(torch.conj(G1) * G2))

    def retr(self, z, v):
        """
        Project the points on the tangent plane back to the manifold space
        """
        return z+v
    
    def egrad2rgrad(self, X, G):
        return G
    
    def ehess2rhess(self, x, eg, eh, d):
        # Euclidean -> Riemannian Hessian.
        return eh
    def norm(self, X, G):
        return torch.norm(G)

    def randvec(self, z):
        v = torch.randn(self._n, self._m)*(1j*z)
        v = v/torch.norm(v)
        return v

    def transp(self, x1, x2, d):
        return d

    def dist(self, X, Y):
        return torch.norm(X-Y)

    def lincomb(self, x, a1, a2):
        return a1-a2
