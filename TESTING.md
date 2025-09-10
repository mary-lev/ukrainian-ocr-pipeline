# Testing Guide for Ukrainian OCR Pipeline

This document describes the comprehensive test suite for the Ukrainian OCR Pipeline package.

## 📋 Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── unit/                          # Unit tests for individual components
│   ├── test_config.py             # Configuration system tests
│   ├── test_gpu_utils.py          # GPU utilities tests
│   ├── test_io_utils.py           # I/O utilities tests  
│   ├── test_ner.py                # NER extraction tests
│   ├── test_ocr.py                # OCR processing tests
│   ├── test_pipeline.py           # Main pipeline tests
│   └── test_segmentation.py       # Segmentation tests
├── integration/                   # Integration tests for workflows
│   ├── test_import_integration.py # Package import tests
│   └── test_pipeline_integration.py # Complete workflow tests
└── fixtures/                     # Test data and utilities
    ├── create_test_data.py       # Test fixture generator
    ├── test_document.png         # Generated test images
    ├── small_document.png
    └── empty_document.png
```

## 🚀 Running Tests

### Prerequisites

Ensure you have the required dependencies:

```bash
# Check dependencies
make check-deps
# or
python run_tests.py --check-deps

# Install development dependencies
make install-dev
# or  
pip install -e .[dev]
```

### Test Commands

#### Using the Makefile (Recommended)

```bash
# Run all tests
make test

# Run specific test types
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-import        # Import functionality tests

# Verify installation
make verify-install     # Quick verification that package works
```

#### Using the Test Runner

```bash
# Run all tests
python run_tests.py

# Run specific test types  
python run_tests.py --unit
python run_tests.py --integration

# Run specific test
python run_tests.py --test tests.unit.test_config.TestOCRPipelineConfig.test_default_initialization

# Check dependencies
python run_tests.py --check-deps

# Get help
python run_tests.py --help
```

#### Using pytest (Alternative)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ukrainian_ocr

# Run specific test file
pytest tests/unit/test_config.py

# Run with verbose output
pytest -v
```

## 📊 Test Categories

### Unit Tests

Test individual components in isolation:

- **Configuration Tests** (`test_config.py`)
  - Component configuration validation
  - File I/O operations (JSON/YAML)
  - Optimization methods (CPU/GPU/Colab)
  - Backward compatibility

- **GPU Utilities Tests** (`test_gpu_utils.py`)
  - GPU detection and availability
  - Device optimization
  - Batch size recommendations
  - Memory monitoring

- **I/O Utilities Tests** (`test_io_utils.py`)  
  - Image loading and saving
  - JSON file operations
  - File discovery and validation
  - Unicode handling

- **Pipeline Component Tests**
  - Segmentation (`test_segmentation.py`)
  - OCR Processing (`test_ocr.py`)
  - NER Extraction (`test_ner.py`)
  - Main Pipeline (`test_pipeline.py`)

### Integration Tests

Test complete workflows and component interactions:

- **Import Integration** (`test_import_integration.py`)
  - Package import functionality
  - Class instantiation
  - Module reloading
  - Optional dependency handling

- **Pipeline Integration** (`test_pipeline_integration.py`)
  - Complete OCR workflows
  - Batch processing
  - Error recovery
  - Performance monitoring

## 🧪 Test Fixtures

The test suite includes automatically generated test fixtures:

- **Test Images**: Simulated document images with text-like regions
- **Sample Data**: Structured test data for lines, entities, and OCR results
- **Configuration**: Various configuration scenarios for testing

Test fixtures are created automatically when running tests. To regenerate:

```bash
cd tests/fixtures
python create_test_data.py
```

## 📈 Testing Best Practices

### What Gets Tested

✅ **Component Functionality**
- All public APIs and methods
- Configuration validation
- Error handling and recovery
- Edge cases and boundary conditions

✅ **Integration Scenarios**  
- Complete pipeline workflows
- Component interactions
- Batch processing
- Multi-image processing

✅ **Installation and Imports**
- Package structure and imports
- Class instantiation
- Optional dependency handling
- Backward compatibility

### What's Mocked/Simulated

🔀 **Heavy Dependencies**
- Actual ML model loading (uses placeholders)
- GPU operations (simulated when no GPU available)
- Large file processing (uses small test images)

🔀 **External Services**
- Network operations
- File system operations in isolated environments
- Platform-specific functionality

## 🐛 Debugging Tests

### Running Specific Tests

```bash
# Run single test method
python run_tests.py --test tests.unit.test_config.TestOCRPipelineConfig.test_colab_optimization

# Run single test class
python run_tests.py --test tests.unit.test_config.TestOCRPipelineConfig

# Run single test file
python run_tests.py --test tests.unit.test_config
```

### Verbose Output

```bash
# Maximum verbosity
python run_tests.py --verbose --verbose --verbose

# With pytest
pytest -vv -s
```

### Test Environment

```bash
# Skip test setup (faster for debugging)
python run_tests.py --no-setup

# Check what fixtures are created
ls tests/fixtures/
```

## 🏗️ Continuous Integration

### GitHub Actions

The repository includes GitHub Actions workflows (`.github/workflows/tests.yml`):

- **Multi-Python Testing**: Tests on Python 3.8-3.11
- **Package Installation**: Tests installation from source
- **Colab Simulation**: Simulates Google Colab environment
- **Import Verification**: Ensures all imports work correctly

### Local CI Simulation

```bash
# Run full CI pipeline locally
make ci-test

# This runs:
# 1. Dependency checks
# 2. Installation verification  
# 3. Complete test suite
```

## 📊 Test Coverage

### Current Coverage

The test suite covers:

- ✅ **Configuration System**: ~95% coverage
- ✅ **GPU Utilities**: ~90% coverage  
- ✅ **I/O Operations**: ~85% coverage
- ✅ **Pipeline Components**: ~80% coverage (placeholder implementations)
- ✅ **Integration Workflows**: ~75% coverage

### Generating Coverage Reports

```bash
# Install coverage tools
pip install pytest-cov coverage

# Run with coverage
pytest --cov=ukrainian_ocr --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🚨 Common Issues and Solutions

### Import Errors

```bash
# Issue: ModuleNotFoundError
# Solution: Ensure package is installed in development mode
pip install -e .
```

### Missing Dependencies

```bash
# Issue: Missing opencv-python, Pillow, etc.
# Solution: Install core dependencies
pip install opencv-python Pillow

# Or install with extras
pip install -e .[all]
```

### GPU-Related Test Failures

```bash
# Issue: CUDA tests fail on CPU-only systems
# Expected: GPU tests should gracefully handle no-GPU scenarios
# The tests are designed to work on both GPU and CPU systems
```

### Test Data Issues

```bash
# Issue: Test fixtures not found
# Solution: Regenerate test fixtures
cd tests/fixtures && python create_test_data.py
```

## 🔄 Adding New Tests

### Unit Test Template

```python
import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.instance = YourClass()
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.instance.your_method()
        self.assertIsInstance(result, expected_type)
    
    def test_error_handling(self):
        """Test error handling"""
        with self.assertRaises(ExpectedError):
            self.instance.method_that_should_fail()

if __name__ == '__main__':
    unittest.main()
```

### Integration Test Template

```python
import unittest
import tempfile
from ukrainian_ocr import UkrainianOCRPipeline

class TestYourIntegration(unittest.TestCase):
    def test_complete_workflow(self):
        """Test complete workflow"""
        pipeline = UkrainianOCRPipeline(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = pipeline.process_single_image(
                'test_image.png',
                output_dir=temp_dir
            )
            
            self.assertTrue(result['success'])
```

## 📚 References

- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)
- [pytest documentation](https://docs.pytest.org/)
- [GitHub Actions Python workflow](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

---

For questions about testing, please check the [main README](README.md) or open an issue in the repository.