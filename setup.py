#!/usr/bin/env python3
"""
Ukrainian OCR Pipeline Package Setup
High-performance OCR pipeline for historical Ukrainian documents with NER
"""

import os
import sys
from setuptools import setup, find_packages

# Ensure minimum Python version
if sys.version_info < (3, 8):
    sys.exit("Python 3.8+ is required")

# Read README safely
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, UnicodeDecodeError):
        return "Ukrainian OCR Pipeline for historical Ukrainian documents"

# Read version from package
def get_version():
    try:
        # Try to read version from __init__.py
        version_file = os.path.join("ukrainian_ocr", "__init__.py")
        if os.path.exists(version_file):
            with open(version_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split('"')[1]
    except Exception:
        pass
    return "1.0.0"  # Fallback version

setup(
    name="ukrainian-ocr-pipeline",
    version=get_version(),
    description="High-performance OCR pipeline for historical Ukrainian documents with Named Entity Recognition",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mary-lev/ukrainian-ocr-pipeline",
    author="Ukrainian OCR Team",
    author_email="contact@example.com",
    license="MIT",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic", 
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    keywords="ocr, ukrainian, historical, documents, ner, genealogy, kraken, trocr",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Core dependencies - keep minimal for better compatibility
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    
    # Optional dependencies for different use cases
    extras_require={
        # GPU-accelerated inference (recommended for Colab)
        "gpu": [
            "accelerate>=0.20.0",
        ],
        
        # Kraken segmentation (optional)
        "segmentation": [
            "kraken>=4.3.0",
        ],
        
        # SpaCy NER (optional)
        "ner-spacy": [
            "spacy>=3.6.0",
            "spacy-transformers>=1.2.0",
        ],
        
        # All optional features
        "all": [
            "accelerate>=0.20.0",
            "kraken>=4.3.0", 
            "spacy>=3.6.0",
            "spacy-transformers>=1.2.0",
        ],
        
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        
        # Colab-specific dependencies
        "colab": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "ipywidgets>=8.0.0",
        ],
    },
    
    # Entry points for CLI (if needed later)
    # entry_points={
    #     "console_scripts": [
    #         "ukrainian-ocr=ukrainian_ocr.cli:main",
    #     ],
    # },
    
    # Include package data
    include_package_data=True,
    
    # Package data - specify explicitly if needed
    package_data={
        "ukrainian_ocr": ["*.yaml", "*.json"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/mary-lev/ukrainian-ocr-pipeline/issues",
        "Source": "https://github.com/mary-lev/ukrainian-ocr-pipeline",
        "Documentation": "https://github.com/mary-lev/ukrainian-ocr-pipeline#readme",
    },
    
    # Zip safe
    zip_safe=False,
)