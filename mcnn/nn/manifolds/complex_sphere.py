import torch

from .manifold import Manifold


class ComplexSphere(Manifold):
    """
    Complex Sphere manifold class with internal parameters satisfying the unit F-parameter constraint: A \in \mathbb C^{m\times n},\|A\|_F=1
    For knowledge of complex spherical manifolds see: https://www.nicolasboumal.net/book/  P157
    """

    def __init__(self, height, width):
        super(Manifold, self).__init__()
        self._n = height
        self._m = width
        self._dim = 2*(self._n*self._m)-1
        self._size = torch.Size((height, width))

    def __str__(self):
        if self._k == 1:
            return "Complex Circle manifold St({}, {})".format(self._n, self._p)
        elif self._k >= 2:
            return "Product Complex Circle manifold St({}, {})^{}".format(self._n, self._p, self._k)

    def rand(self):
        """
        Generate random points that satisfy the unit F-parameter constraint
        """
        x = torch.randn(self._n, self._m)+1j*torch.randn(self._n, self._m)
        return x/torch.norm(x)

    def proj(self, x, d):
        """
        Projection of the Euclidean gradient onto the tangent plane
        """
        return d-x*torch.real(torch.sum(torch.conj(x)*d))

    def inner(self, X, G1, G2=None):
        if G2 == None:
            G2 = G1
        return torch.real(torch.sum(torch.conj(G1) * G2))

    def retr(self, x, d):
        """
        Project the points on the tangent plane back to the manifold space
        """
        y = x+d
        return y/torch.norm(y)

    def ehess2rhess(self, x, egrad, ehess, u):
        # Euclidean -> Riemannian Hessian.
        return self.proj(x, ehess) - torch.real(torch.sum(torch.conj(x) * egrad))*u

    def norm(self, X, G):
        return torch.norm(G)

    def randvec(self, x):
        d = torch.randn(self._n, self._m)+1j*torch.randn(self._n, self._m)
        d = d-x*torch.real(torch.sum(torch.conj(x)*d))
        d = d/torch.norm(d)
        return d

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def dist(self, X, Y):
        return torch.real(2*torch.asin(0.5*torch.norm(X-Y)))

    def lincomb(self, x, a1, a2):
        return a1-a2
