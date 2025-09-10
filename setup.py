#!/usr/bin/env python3
"""
Ukrainian OCR Pipeline Package
High-performance OCR pipeline for historical Ukrainian documents with NER
"""

from setuptools import setup, find_packages
import pathlib

# Read README
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text() if (HERE / "README.md").exists() else "Ukrainian OCR Pipeline"

setup(
    name="ukrainian-ocr-pipeline",
    version="1.0.0",
    description="High-performance OCR pipeline for historical Ukrainian documents with Named Entity Recognition",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ukrainian-ocr-pipeline",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ocr, ukrainian, historical, documents, ner, genealogy, kraken, trocr",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.8",
    
    # Core dependencies
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "lxml>=4.6.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "click>=8.0.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
    ],
    
    # Optional dependencies for different backends
    extras_require={
        # GPU-accelerated inference (recommended for Colab)
        "gpu": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
        
        # CPU-only inference (lighter install)
        "cpu": [
            "torch>=2.0.0+cpu",
            "transformers>=4.30.0",
        ],
        
        # Kraken segmentation
        "segmentation": [
            "kraken>=4.3.0",
        ],
        
        # SpaCy NER
        "ner-spacy": [
            "spacy>=3.6.0",
            "spacy-transformers>=1.2.0",
        ],
        
        # All features (complete install)
        "all": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
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
            "mypy>=1.0.0",
        ],
        
        # Colab-specific extras
        "colab": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "ipywidgets>=8.0.0",
            "google-colab",
        ]
    },
    
    # Command-line interfaces
    entry_points={
        "console_scripts": [
            "ukrainian-ocr=ukrainian_ocr.cli:main",
            "ukr-ocr=ukrainian_ocr.cli:main",
        ],
    },
    
    # Include package data
    include_package_data=True,
    package_data={
        "ukrainian_ocr": [
            "configs/*.yaml",
            "models/*.json", 
            "data/*.txt",
        ]
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-username/ukrainian-ocr-pipeline/issues",
        "Source": "https://github.com/your-username/ukrainian-ocr-pipeline",
        "Documentation": "https://ukrainian-ocr-pipeline.readthedocs.io/",
    },
)