
from setuptools import setup, find_packages


install_requires = [
    "torch==1.5.0",
]


setup(
    name="neuralprocess",
    version="0.1",
    description="Neural Process sample code",
    packages=find_packages(),
    install_requires=install_requires,
)
