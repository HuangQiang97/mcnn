from setuptools import find_packages, setup

setup(
    name='mcnnshit',
    version='1.0.0',
    description="Constrained Optimization and Manifold Optimization in Pytorch",
    author='huang chiang',
    license="MIT",
    keywords=["Constrained Optimization", "Optimization on Manifolds", "Pytorch"],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["torch>=1.9"],
    python_requires=">=3.5",
)