from setuptools import setup, find_packages

setup(
    name="aion_python",
    version="0.1.0",
    packages=find_packages(),
    package_data={"aion_python": ["*.so", "*.pyd"]},
    install_requires=[],
)
