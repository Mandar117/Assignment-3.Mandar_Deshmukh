"""
Setup script for the gym_mountaincar package.

This script configures the package for installation, specifying its dependencies
and metadata.
"""

from setuptools import find_packages, setup

setup(
    name="gym_mountaincar",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.21.0",
    ],
)
