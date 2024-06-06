from setuptools import find_packages, setup

setup(
    name='mcnnlib',
    version='1.0.2',
    author='huang chiang',
    author_email="huangqiang97@126.com",
    description="Constrained Optimization and Manifold Optimization in Pytorch",
    long_description=
    r'''
`Manifold Constrained Neural Network(MCNN)`为在`PyTorch`中进行复数约束性优化和流形优化提供了一种简单的方法。无需任何模板，提供开箱即用的优化器、网络层和网络模型，只需在构建模型时声明约束条件，即可开始使用。

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
    ''',
    long_description_content_type="text/markdown",
    url="https://github.com/HuangQiang97/mcnn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    keywords=["Constrained Optimization",
              "Optimization on Manifolds", "Pytorch"],
    include_package_data=True,
    zip_safe=False,
    install_requires=["torch>=1.9"],
    python_requires=">=3.5",
)
