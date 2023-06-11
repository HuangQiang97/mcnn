"""
In this program we show how to use mcnn to compute the maximum eigenvalue
of a complex symmetric matrix via the Rayleigh quotient, restricting the optimisation
problem to the Sphere
alse see the first example in https://www.manopt.org/tutorial.html#gettingstarted
"""
import torch
from torch.linalg import eigvalsh

from torch import nn
import mcnn.nn as mnn
import mcnn.optim as mopt
import matplotlib.pyplot as plt


N = 1000  # matrix size
LR = 1.0 / N  # step-size.


class Model(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.x = mnn.ManifoldLinear(1, n, weight_manifold=mnn.ComplexSphere, bias=False).to(torch.complex64).weight

    def forward(self, A):
        x = self.x
        return x.T.conj() @ A @ x


# Generate matrix
A = torch.rand(N, N)+1j*torch.rand(N, N)  # Uniform on [0, 1)
A = 0.5 * (A + A.T.conj())


# Compare against diagonalization (eigenvalues are returend in ascending order)
max_eigenvalue = eigvalsh(A)[-1]
print("Max eigenvalue: {:10.5f}".format(max_eigenvalue))

# Instantiate model and optimiser
model = Model(N)
optim = mopt.ManifoldSGD(model.parameters(), lr=LR)

eigenvalue = float("inf")
i = 0
err_list = []
while (eigenvalue - max_eigenvalue).abs() > 1e-3:
    eigenvalue = model(A)
    optim.zero_grad()
    (-eigenvalue).backward()
    optim.step()
    print("{:2}. Best guess: {:10.5f}".format(i, eigenvalue.item()))
    i += 1
    err_list.append((eigenvalue - max_eigenvalue).abs().item())


print("Final error {:.5f}".format((eigenvalue - max_eigenvalue).abs().item()))
plt.figure()
plt.plot(err_list, marker='.')
plt.yscale('log')
plt.xlabel('Iteration numbers')
plt.ylabel('error')
plt.show()
