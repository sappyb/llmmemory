#!/usr/bin/env python3
"""
Setup script for Evelyn AI - Student Simulation System
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
    name="evelyn-ai",
    version="8.0.0",
    author="Evelyn AI Team",
    author_email="contact@evelyn-ai.com",
    description="AI-powered student simulation system for educational research and training",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sappyb/llmmemory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "web": [
            "streamlit>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evelyn-ai=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="ai, education, student-simulation, nlp, machine-learning, rag",
    project_urls={
        "Bug Reports": "https://github.com/sappyb/llmmemory/issues",
        "Source": "https://github.com/sappyb/llmmemory",
        "Documentation": "https://github.com/sappyb/llmmemory#readme",
    },
)