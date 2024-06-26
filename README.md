# MCNN

> 为复数神经网络提供约束性优化和流形优化的基于`PyTorch`的库

## Overview

`Manifold Constrained Neural Network(MCNN)`为在`PyTorch`中进行复数约束性优化和流形优化提供了一种简单的方法。无需任何模板，提供开箱即用的优化器、网络层和网络模型，只需在构建模型时声明约束条件，即可开始使用。

提供全连接网络和卷积网络的流形版本：流形全连接网络（`Manifold Linear`） 、流形卷积神经网络（`Manifold Conv`）、流形循环神经网络(`Manifold RNN`)，网络内部的参数均满足特定的流形约束。同时实现了流形网络优化器：流形随机梯度下降优化器（`Manifold SGD`）、流形自适应梯度优化器（`Manifold Adagrad`）、流形均方根传播优化器（`Manifold RMSprop`）、流形自适应动量估计算法优化器（`Manifold Adam`）。

<center><img src=".\img\arch.png" width = "600" alt="图片名称" align=center /></center>

流形复数神经网络框架基于`PyTorch`开发，针对网络的参数类、网络结构类和优化器类进行修改以适应流形约束。同时定义流形类，规定了各类流形的随机初始化、投影和缩放操作。

## Example

使用流形全连接网络`ManifoldLinear`和流形卷积网络`ManifoldConv`模块，可以搭建自己的流形复数神经网络。由于流形复数神经网络中的重要组件继承自`PyTorch`，并封装了底层实现，使得框架的使用方式和普通神经网络框架基本一致。

具体化算法设计与工程实现可见[算法设计](https://huangqiang97.github.io/doc/blog/mcnn_framework.html)；应用于毫米波大规模MIMO模拟波束设计可见[研究应用](https://huangqiang97.github.io/doc/blog/mcnn_beamtraining.html)。下面展示使用流形复数神经网络计算复对称矩阵 $A \in \mathbb C^{n\times n}$ 的最大特征值。

最大特征值 $\lambda$ 是下列优化问题的最优解：

$$
\max\limits_{x\in\mathbb{C}^n, x \neq 0} \frac{x^H A x}{x^H
            x}.
$$

可以重写为：

$$
\min\limits_{x\in\mathbb{C}^n, \|x\| = 1} -x^H A x.
$$

$x$的约束要求 $x$ 满足单位`2-norm`，所以 $x$ 是单位球空间上一点：

$$
\mathbb{S}^{n-1} = \{x \in \mathbb{C}^n : x^H x = 1\}.
$$

所以我们对$x$添加`Complex Sphere`流形约束。

```python
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
optim = mopt.ConjugateGradient(model.parameters(), lr=LR)

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
```

<center><img src="img\err_plot.png" width = "600" alt="图片名称" align=center /></center>

## Constraints

支持的流形约束：

* `Complex Sphere`，复球流形，满足约束： $X \in \mathbb C^{m \times n}, \| X \|_F=1$ 
* `Complex Stiefel`，复Stiefel流形，满足约束： $X \in \mathbb C^{m\times n},{X}^H{X}={I}$ 
* `Complex Circle`，复单位圆流形，满足约束： $X \in \mathbb C^{m\times n},|[{X}]_{i,j}|=1$ 
* `Complex Euclid`，复欧几里得流形，满足约束： $X \in \mathbb C^{m\times n}$ 

## Supported Spaces

`mcnn`中的每个约束条件都是以流形的形式实现，这使用户在选择每个参数化的选项时有更大的灵活性。所有流形都支持黎曼梯度下降法，同样也支持其他`PyTorch`优化器。

`mcnn`目前支持以下空间：

* `Cn(n)`: $\mathbb C^n$空间内的无约束优化空间
* `Sphere(n)`:  $\mathbb C^n$空间内的球体
* `SO(n)`:  `n×n` 正交矩阵流形
* `St(n,k)`:  `n×k` 列正交矩阵流形

## Supported Modules

`mcnn`目前支持以网络类型：

* `Linear`全连接网络层
* `Conv2d, Conv3d`二维及三维卷积层
* `RNN`循环神经网络层

## optimizers

`mcnn`目前支持以下优化器：

* `Conjugate Gradient`，共轭梯度优化器
* `Manifold Adam`，流形自适应动量估计算法优化器
* `Manifold Adagrad`，流形自适应梯度优化器
* `Manifold RMSprop`，流形均方根传播优化器
* `Manifold SGD`，流形统计梯度下降优化器
* `QManifold Adagrad`，带参数量化的流形自适应梯度优化器
* `QManifold RMSprop`，带参数量化的流形均方根传播优化器

## Using MCNN in your Code

* 安装`mcnn`:`pip install mcnnlib==1.0.2`

## Reference

* [manopt](https://github.com/NicolasBoumal/manopt) : a Matlab toolbox for optimization on manifolds
* [Pymanopt](https://github.com/pymanopt/pymanopt): A Python toolbox for optimization on Riemannian manifolds with support for automatic differentiation
* [McTorch ](https://github.com/mctorch/mctorch): a manifold optimization library for deep learning
* [An introduction to optimization on smooth manifolds](https://www.nicolasboumal.net/#book):An introduction to optimization on smooth manifolds
