# MCNN

> 为复数神经网络提供约束性优化和流形优化的基于`PyTorch`的库

## Overview

`Manifold Constrained Neural Network(MCNN)`为在`PyTorch`中进行f复数约束性优化和流形优化提供了一种简单的方法。无需任何模板，提供开箱即用的优化器、网络层和网络模型，训练代码中没有。只需在构建模型时声明约束条件，即可开始使用。

全连接网络和卷积网络的流形版本：流形全连接网络（`Manifold Linear`） 、流形卷积神经网络（`Manifold Conv`），网络内部的参数均满足特定的流形约束。同时实现了流形网络优化器：流形随机梯度下降优化器（`Manifold SGD`）、流形自适应梯度优化器（`Manifold Adagrad`）、流形均方根传播优化器（`Manifold RMSprop`）。



<center><img src=".\img\arch.png" style="zoom: 45%;" /></center>

流形复数神经网络框架基于`PyTorch`开发，针对网络的参数类、网络结构类和优化器类进行修改以适应流形约束。同时定义流形类，规定了各类流形的随机初始化、投影和缩放操作。

使用流形全连接网络`ManifoldLinear`和流形卷积网络`ManifoldConv`模块，可以搭建自己的流形复数神经网络。由于流形复数神经网络中的重要组件继承自`PyTorch`，并封装了底层实现，使得框架的使用方式和普通神经网络框架基本一致。下面展示使用流形复数神经网络求解正交普鲁克问题：


$$
\begin{equation}
    \begin{aligned}& \underset{C}{\text{min}} \,\,  \left\Vert B-AC\right\Vert _F \\ &{~\rm s.t.} \,\,\,\,  C^HC=I.
    \end{aligned}
\end{equation}
$$

其中$A,B \in\mathbb C^{m \times n}$已知，$C \in\mathbb C^{n \times n }$为待求解参数，且满足正交约束。构建的流形神经网络仅包含一个流形全连接层，不使用偏置参数和激活函数，流形全连接层内部参数即表示$C$，使用`Complex Stiefel`流形对全连接层添加正交约束。使用$\| B-AC\| _F$作为损失函数，同时使用`ManifoldRMSprop`优化器优化网络

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import mcnn.nn as mnn
import mcnn.optim as moptim

# 网络模型
class OPPNet(nn.Module):

    def __init__(self, m):
        super(OPPNet, self).__init__()
        self.c = mnn.ManifoldLinear(m, m, weight_manifold=mnn.ComplexStiefel, bias=False).to(torch.complex128).weight

    def forward(self, a,b):
        loss = torch.linalg.norm(b-a@self.c)
        return loss

n=16
epoch=512
err_list=np.zeros(epoch)
err_opt=0
repeat=32

for _ in range(repeat):
    
    A = np.random.randn(n*2, n)+1j*np.random.randn(n*2, n)
    C, _ = np.linalg.qr(np.random.randn(n, n)+1j*np.random.randn(n, n))
    B = A@C+0.01*(np.random.randn(n*2, n)+1j*np.random.randn(n*2, n))
    
    # 闭式最优解
    U, _, V_h = np.linalg.svd(B.T.conj()@A)
    O = (U@V_h).T.conj()
    err_opt+=np.linalg.norm(A@O-B)

    A = torch.from_numpy(A)
    B = torch.from_numpy(B)
    C = torch.from_numpy(C)
    
    net = OPPNet(n)
    optimizer = moptim.ManifoldRMSprop(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (optimizer, factor=0.5,patience=12, verbose=False)
    min_loss = 1e3
    
    for i in range(epoch):
        loss = net(A,B)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if loss < min_loss:
            min_loss = loss
        err_list[i]+=loss.detach().numpy()
    
    # 列正交验证
    w = net.c.detach()
    assert  torch.linalg.norm(torch.eye(n)-w.T.conj()@w).item()<1e-6

plt.figure()
plt.plot(err_list/repeat)
plt.plot([err_opt/repeat for _ in range(epoch)])
plt.xlabel('update')
plt.ylabel('err')
plt.legend(['mcnn','opt'])
plt.show()
```

## Constraints

支持的流形约束：

* `Complex Sphere`，复球流形，满足约束：$ X \in \mathbb C^{m\times n},\| X\|_F=1$
* `Complex Stiefel`，复Stiefel流形，满足约束：$ X \in \mathbb C^{m\times n},{X}^H{X}={I}$
* `Complex Circle`，复单位圆流形，满足约束：$ X \in \mathbb C^{m\times n},|[{X}]_{i,j}|=1$
* `Complex Euclid`，复欧几里得流形，满足约束：$ X \in \mathbb C^{m\times n}$

## Supported Spaces

`mcnn`中的每个约束条件都是以流形的形式实现，这使用户在选择每个参数化的选项时有更大的灵活性。所有流形都支持黎曼梯度下降法，同样也支持其他`PyTorch`优化器。

`mcnn`目前支持以下空间：

* `Cn(n)`: $\mathbb C^n$空间内的无约束优化空间
* `Sphere(n)`:  $\mathbb C^n$空间内的球体， $\{x\in \mathbb C^n | \|X\|_F=1\} \subset C^n$
* `SO(n)`:  `n×n` 正交矩阵流形
* `St(n,k)`:  `n×k` 列正交矩阵流形

## Using MCNN in your Code

* 安装`mcnn`:`pip install mcnnlib==1.0.0`

