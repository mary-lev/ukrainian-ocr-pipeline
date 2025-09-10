"""
Ukrainian OCR Pipeline
High-performance OCR for historical Ukrainian documents with NER
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.pipeline import UkrainianOCRPipeline
from .config import OCRConfig
from .core.batch_processor import BatchProcessor

# Component imports
from .core.segmentation import KrakenSegmenter
from .core.ocr import TrOCRProcessor
from .core.ner import NERExtractor
from .core.enhancement import ALTOEnhancer

# Utility imports
from .utils.gpu import check_gpu_availability, optimize_for_device
from .utils.models import ModelManager

# Colab utilities (imported conditionally)
try:
    from .utils.colab import (
        download_results, 
        setup_colab_environment, 
        list_output_files,
        get_processing_results_summary
    )
    _colab_available = True
except ImportError:
    _colab_available = False

__all__ = [
    # Main classes
    "UkrainianOCRPipeline",
    "OCRConfig", 
    "BatchProcessor",
    
    # Components
    "KrakenSegmenter",
    "TrOCRProcessor", 
    "NERExtractor",
    "ALTOEnhancer",
    
    # Utils
    "check_gpu_availability",
    "optimize_for_device",
    "ModelManager",
]

# Add Colab utilities if available
if _colab_available:
    __all__.extend([
        "download_results",
        "setup_colab_environment", 
        "list_output_files",
        "get_processing_results_summary"
    ])

# Package metadata
__package_name__ = "ukrainian-ocr-pipeline"
__description__ = "High-performance OCR pipeline for historical Ukrainian documents"