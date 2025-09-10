# 🇺🇦 Ukrainian OCR Pipeline

[![PyPI version](https://badge.fury.io/py/ukrainian-ocr-pipeline.svg)](https://badge.fury.io/py/ukrainian-ocr-pipeline)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mary-lev/ukrainian-ocr-pipeline/blob/main/examples/Ukrainian_OCR_Colab_Demo.ipynb)

High-performance OCR pipeline for historical Ukrainian documents with Named Entity Recognition (NER). Optimized for cloud inference with GPU acceleration.

## ✨ Features

- **🚀 GPU-Accelerated**: Optimized for Google Colab and cloud inference
- **📝 TrOCR Integration**: Advanced Cyrillic handwriting recognition
- **🎯 Named Entity Recognition**: Automatic detection of persons and locations
- **📋 ALTO XML Output**: Archival-standard document format
- **🎨 Person-Dense Regions**: Automatic extraction of genealogically valuable areas
- **📊 Batch Processing**: Efficient processing of multiple documents
- **🔧 Easy Configuration**: YAML-based configuration with sensible defaults

## 🚀 Quick Start

### Google Colab (Recommended)

The fastest way to get started is with our Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ukrainian-ocr-pipeline/blob/main/examples/Ukrainian_OCR_Colab_Demo.ipynb)

### Local Installation

```bash
# Basic installation
pip install ukrainian-ocr-pipeline

# GPU support (recommended)
pip install ukrainian-ocr-pipeline[gpu]

# Complete installation with all features
pip install ukrainian-ocr-pipeline[all]
```

### Basic Usage

```python
from ukrainian_ocr import UkrainianOCRPipeline

# Initialize pipeline
pipeline = UkrainianOCRPipeline(device='auto')

# Process single image
result = pipeline.process_single_image('document.jpg')

# Process multiple images
results = pipeline.process_batch(['doc1.jpg', 'doc2.jpg'])

print(f"Processing complete! Found {result['lines_with_text']} text lines")
```

## 📊 Performance

**Google Colab (T4 GPU)**:
- ~10-15 seconds per page
- Batch processing of 4-8 images simultaneously
- Automatic GPU memory management

**Local CPU**:
- ~60-120 seconds per page
- Single image processing recommended

## 🛠️ Installation Options

### For Google Colab
```bash
pip install ukrainian-ocr-pipeline[colab]
```

### For Production
```bash
pip install ukrainian-ocr-pipeline[all]
```

### Development
```bash
git clone https://github.com/your-username/ukrainian-ocr-pipeline
cd ukrainian-ocr-pipeline
pip install -e .[dev]
```

## 🎯 Use Cases

### 📚 Historical Document Processing
- Digitize handwritten Ukrainian documents
- Extract genealogical information
- Create searchable archives

### 🔍 Named Entity Recognition
- Identify person names in historical records
- Extract location references
- Generate structured metadata

### 📋 Archival Standards
- Generate ALTO XML v4 compliant output
- Import into eScriptorium
- Preserve document structure and metadata

## 🧠 Models

The pipeline uses state-of-the-art models optimized for Ukrainian/Cyrillic text:

- **Segmentation**: Kraken BLLA model for text line detection
- **OCR**: TrOCR model fine-tuned for Cyrillic handwriting
- **NER**: spaCy ru_core_news_lg + custom filtering for Ukrainian

## ⚙️ Configuration

Create a `config.yaml` file to customize the pipeline:

```yaml
# Device configuration
device: auto  # 'cuda', 'cpu', or 'auto'
batch_size: 4  # Adjust based on GPU memory

# OCR settings
ocr:
  model_path: "cyrillic-trocr/trocr-handwritten-cyrillic"
  preprocessing: false

# NER settings  
ner:
  backend: "spacy"
  confidence_threshold: 0.7

# Post-processing
post_processing:
  extract_person_regions: true
  clustering_eps: 300
```

Then use it:

```python
from ukrainian_ocr import UkrainianOCRPipeline, OCRConfig

config = OCRConfig.from_file('config.yaml')
pipeline = UkrainianOCRPipeline(config=config)
```

## 📁 Output Structure

```
output_directory/
├── alto/
│   ├── document_basic.xml      # Basic ALTO XML
│   └── document_enhanced.xml   # Enhanced with NER tags
├── visualizations/
│   └── document_segments.jpg   # Segmentation visualization
└── person_regions/
    ├── document_person_region.jpg  # Cropped person-dense area
    └── document_visualization.jpg  # Full document with highlights
```

## 🔧 Command Line Interface

```bash
# Process single image
ukrainian-ocr process image.jpg --output ./results

# Process directory
ukrainian-ocr process ./images --output ./results --batch-size 4

# Extract person regions only
ukrainian-ocr extract-persons document_enhanced.xml image.jpg

# View help
ukrainian-ocr --help
```

## 🧪 API Reference

### UkrainianOCRPipeline

Main pipeline class for processing documents.

```python
pipeline = UkrainianOCRPipeline(
    config=None,          # Configuration object or file path
    device='auto',        # 'cuda', 'cpu', or 'auto'
    batch_size=None,      # Batch size (auto-determined if None)
    verbose=True          # Enable progress bars
)
```

**Methods:**
- `process_single_image(image_path, output_dir, save_intermediate)` 
- `process_batch(image_paths, output_dir, save_intermediate)`
- `get_stats()` - Get processing statistics
- `cleanup()` - Clean up GPU memory

### OCRConfig

Configuration management for the pipeline.

```python
# Load from file
config = OCRConfig.from_file('config.yaml')

# Create from dictionary
config = OCRConfig.from_dict({
    'device': 'cuda',
    'batch_size': 4
})

# Optimize for different environments
config.update_for_colab()      # Optimize for Google Colab
config.update_for_production() # Optimize for production deployment
```

## 🚀 Cloud Deployment

### Google Colab
- Free GPU access (T4)
- Pre-configured environment
- Interactive notebook interface
- Easy file upload/download

### Kaggle
- Similar to Colab with different GPU options
- Longer runtime limits
- Dataset integration

### Cloud Platforms
- AWS SageMaker
- Google Cloud AI Platform  
- Azure Machine Learning

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/your-username/ukrainian-ocr-pipeline
cd ukrainian-ocr-pipeline
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kraken](https://github.com/mittagessen/kraken) for segmentation
- [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) for OCR recognition
- [spaCy](https://spacy.io/) for named entity recognition
- Historical document datasets and contributors

## 📞 Support

- 📖 [Documentation](https://ukrainian-ocr-pipeline.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/your-username/ukrainian-ocr-pipeline/issues)
- 💬 [Discussions](https://github.com/your-username/ukrainian-ocr-pipeline/discussions)

## 📈 Roadmap

- [ ] Additional language support (Polish, Czech, etc.)
- [ ] Custom model training utilities
- [ ] Web interface
- [ ] Docker containerization
- [ ] Kubernetes deployment examples
- [ ] Enhanced genealogical features

---

<div align="center">
  <strong>Built with ❤️ for Ukrainian historical preservation</strong>
</div>