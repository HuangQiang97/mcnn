import torch

from ..utils.manifold_multi import multihconj, multiherm, multiprod
from .manifold import Manifold


class ComplexStiefel(Manifold):
    """
    Complex Stiefel manifold class with internal parameters satisfying column orthogonality constraints: A \in \mathbb C^{m\times n},A^H*A=I
    For knowledge of complex Stiefel manifolds see: https://www.nicolasboumal.net/book/  P159
    """

    def __init__(self, height, width, k=1):
        if height < width or width < 1:
            raise ValueError(("Need height >= width >= 1. Values supplied were height = {} and width = {}.").format(height, width))
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = {}.".format(k))

        super(Manifold, self).__init__()
        self._n = height
        self._p = width
        self._k = k

        self._dim = self._k*(2*self._n*self._p-self._p**2)

        if k == 1:
            self._size = torch.Size((height, width))
        else:
            self._size = torch.Size((k, height, width))

    def __str__(self):
        if self._k == 1:
            return "Complex Stiefel manifold St({}, {})".format(self._n, self._p)
        elif self._k >= 2:
            return "Product Complex Stiefel manifold St({}, {})^{}".format(self._n, self._p, self._k)

    def rand(self):
        """
        Generate random points satisfying the flow shape constraint by QR decomposition
        """
        if self._k == 1:
            X = torch.randn(self._n, self._p)+1j*torch.randn(self._n, self._p)
            q, r = torch.linalg.qr(X)
            return q

        X = torch.zeros((self._k, self._n, self._p))

        # TODO: update with batch implementation
        for i in range(self._k):
            X[i], r = torch.linalg.qr(torch.randn(self._n, self._p)+1j*torch.randn(self._n, self._p))
        return X

    def proj(self, X, U):
        """
        Projection of the Euclidean gradient onto the tangent plane
        """
        return U - multiprod(X, multiherm(multiprod(multihconj(X), U)))

    def inner(self, X, G1, G2=None):
        if G2 == None:
            G2 = G1
        return torch.real(torch.sum(torch.conj(G1) * G2))

    def retr(self, X, G):
        """
        Project the points on the tangent plane back to the manifold space
        """
        if self._k == 1:
            # Calculate 'thin' qr decomposition of X + G
            q, r = torch.linalg.qr(X + G)
            # Unflip any flipped signs
            XNew = torch.matmul(q, torch.diag(torch.sign(torch.sign(torch.diag(torch.real(r))) + .5))+0j)
        else:
            XNew = X + G
            # TODO: update with batch implementation
            for i in range(self._k):
                q, r = torch.linalg.qr(XNew[i])
                XNew[i] = torch.matmul(q, torch.diag(torch.sign(torch.sign(torch.diag(torch.real(r))) + .5))+0j)
        return XNew

    def ehess2rhess(self, X, egrad, ehess, H):
        # Euclidean -> Riemannian Hessian.
        XtG = multiprod(multihconj(X), egrad)
        symXtG = multiherm(XtG)
        HsymXtG = multiprod(H, symXtG)
        return self.proj(X, ehess - HsymXtG)

    def norm(self, X, G):
        return torch.norm(G)

    def randvec(self, X):
        U = torch.randn(*X.size())+1j*torch.randn(*X.size())
        U = self.proj(X, U)
        U = U / self.norm(X, U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def lincomb(self, x, a1, a2):
        return a1-a2
