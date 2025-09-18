#!/usr/bin/env python3

"""
TruFor: Image Forgery Detection and Localization Library

A Python library for detecting and localizing manipulations in images using deep learning.
Based on the TruFor paper and research from University Federico II of Naples.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return __doc__

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="trufor",
    version="1.0.0",
    author="TruFor Team",
    author_email="trufor@unina.it",
    description="Image Forgery Detection and Localization Library",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/grip-unina/TruFor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trufor-predict=trufor.cli:predict_command",
            "trufor-info=trufor.cli:info_command",
        ],
    },
    include_package_data=True,
    package_data={
        "trufor": [
            "config/*.yaml",
            "models/cmx/*.txt",
        ],
    },
    zip_safe=False,
    keywords=[
        "image forensics",
        "deepfake detection", 
        "image manipulation",
        "computer vision",
        "deep learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/grip-unina/TruFor/issues",
        "Source": "https://github.com/grip-unina/TruFor",
        "Documentation": "https://grip-unina.github.io/TruFor/",
        "Paper": "https://arxiv.org/abs/2212.10957",
    },
)
