#!/usr/bin/env python3
"""Setup script for pytorch-recsys-framework."""

from setuptools import setup, find_packages
import os


def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def read_readme():
    """Read README file."""
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


# Read version from __init__.py
version = {}
with open(os.path.join("pytorch_recsys", "__init__.py")) as fp:
    exec(fp.read(), version)


setup(
    name="pytorch-recsys-framework",
    version=version.get("__version__", "0.1.0"),
    author="Louis Wang",
    author_email="louiswang524@gmail.com",
    description="PyTorch-based modular deep learning framework for sequential recommendation systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/louiswang524/pytorch-recsys",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "serving": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pytorch-recsys=pytorch_recsys.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "pytorch",
        "recommendation-systems",
        "deep-learning",
        "transformers",
        "sequential-modeling",
        "attention-mechanisms",
        "machine-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/louiswang524/pytorch-recsys/issues",
        "Documentation": "https://pytorch-recsys.readthedocs.io/",
        "Source": "https://github.com/louiswang524/pytorch-recsys",
    },
)