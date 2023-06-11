"""
In this program we show how to use mcnn to savle OPP
also see: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import mcnn.nn as mnn
import mcnn.optim as moptim


class OPPNet(nn.Module):

    def __init__(self, m):
        super(OPPNet, self).__init__()
        self.c = mnn.ManifoldLinear(m, m, weight_manifold=mnn.ComplexStiefel, bias=False).to(torch.complex128).weight

    def forward(self, a, b):
        loss = torch.linalg.norm(b-a@self.c)
        return loss


n = 8
epoch = 512
err_list = np.zeros(epoch)
err_opt = 0
repeat = 32

for _ in range(repeat):

    A = np.random.randn(n*2, n)+1j*np.random.randn(n*2, n)
    C, _ = np.linalg.qr(np.random.randn(n, n)+1j*np.random.randn(n, n))
    B = A@C+0.01*(np.random.randn(n*2, n)+1j*np.random.randn(n*2, n))

    # Optimal solution
    U, _, V_h = np.linalg.svd(B.T.conj()@A)
    O = (U@V_h).T.conj()
    err_opt += np.linalg.norm(A@O-B)

    A = torch.from_numpy(A)
    B = torch.from_numpy(B)
    C = torch.from_numpy(C)

    net = OPPNet(n)
    optimizer = moptim.ManifoldAdam(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=12, verbose=False)
    min_loss = 1e3

    for i in range(epoch):
        loss = net(A, B)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if loss < min_loss:
            min_loss = loss
        err_list[i] += loss.detach().numpy()

    # Column orthogonal verification
    w = net.c.detach()
    assert torch.linalg.norm(torch.eye(n)-w.T.conj()@w).item() < 1e-6

plt.figure()
plt.plot(err_list/repeat)
plt.plot([err_opt/repeat for _ in range(epoch)])
plt.xlabel('update')
plt.ylabel('err')
plt.legend(['mcnn', 'opt'])
plt.show()
