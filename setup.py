#!/usr/bin/env python3
"""
Setup script for dab-evaluation package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dab-evaluation",
    version="1.0.0",
    author="DAB Team",
    author_email="dab@example.com",
    description="DAB Evaluation SDK - Web3 Agent Evaluation Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dab-team/dab-evaluation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dab-eval=dab_eval:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dab_evaluation": [
            "data/*.csv",
            "examples/*.py",
        ],
    },
    keywords="evaluation, web3, agent, llm, benchmark, assessment",
    project_urls={
        "Bug Reports": "https://github.com/dab-team/dab-evaluation/issues",
        "Source": "https://github.com/dab-team/dab-evaluation",
        "Documentation": "https://dab-evaluation.readthedocs.io/",
    },
)
