import torch

from .manifold import Manifold


class ComplexCircle(Manifold):
    """
    Complex Circle manifold class with internal parameters satisfying the constant modulus constraint: A \in \mathbb C^{m\times n},|A_{[i,j]}| =1
    For knowledge of complex circle manifolds see: https://www.nicolasboumal.net/book/  P157
    """

    def __init__(self, height, width):
        super(Manifold, self).__init__()
        self._n = height
        self._m = width
        self._dim = self._n*self._m
        self._size = torch.Size((height, width))

    def __str__(self):
        if self._k == 1:
            return "Complex Circle manifold St({}, {})".format(self._n, self._p)
        elif self._k >= 2:
            return "Product Complex Circle manifold St({}, {})^{}".format(self._n, self._p, self._k)

    def rand(self):
        """
        Generate random points that satisfy the constant mode constraint
        """
        x = torch.randn(self._n, self._m)+1j*torch.randn(self._n, self._m)
        return x/torch.abs(x)

    def proj(self, z, u):
        """
        Projection of the Euclidean gradient onto the tangent plane
        """
        return u-torch.real(torch.conj(u)*z)*z

    def inner(self, X, G1, G2):
        return torch.real(torch.sum(torch.conj(G1) * G2))

    def retr(self, z, v):
        """
        Project the points on the tangent plane back to the manifold space
        """
        y=z+v
        return y/torch.abs(y)

    def ehess2rhess(self, z, egrad, ehess, zdot):
        # Euclidean -> Riemannian Hessian.
        return self.proj(z, ehess - torch.real(z*torch.conj(egrad))*zdot)

    def norm(self, X, G):
        return torch.norm(G)

    def randvec(self, z):
        v = torch.randn(self._n, self._m)*(1j*z)
        v = v/torch.norm(v)
        return v

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def dist(self, X, Y):
        return torch.norm(torch.real(2*torch.asin(0.5*torch.abs(X-Y))))
    
    def lincomb(self, x,a1,a2):
        return a1-a2