.PHONY: test test-unit test-integration test-import check-deps install install-dev clean help

# Default target
help:
	@echo "Ukrainian OCR Pipeline - Test Commands"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-import    - Test package imports"
	@echo "  check-deps     - Check dependencies"
	@echo "  install        - Install package in development mode"
	@echo "  install-dev    - Install with development dependencies"
	@echo "  clean          - Clean up test artifacts"
	@echo "  help           - Show this help message"

# Test targets
test:
	python run_tests.py --verbose

test-unit:
	python run_tests.py --unit --verbose

test-integration:
	python run_tests.py --integration --verbose

test-import:
	python run_tests.py --test tests.integration.test_import_integration --verbose

# Dependency and installation targets  
check-deps:
	python run_tests.py --check-deps

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Quick verification - test that package imports work
verify-install:
	@echo "Testing package installation..."
	@python -c "import ukrainian_ocr; print('✅ Main package imports')"
	@python -c "from ukrainian_ocr import UkrainianOCRPipeline, OCRConfig; print('✅ Core classes import')"
	@python -c "from ukrainian_ocr.utils.gpu import check_gpu_availability; gpu_info = check_gpu_availability(); print(f'✅ GPU utils work: CUDA available = {gpu_info[\"cuda_available\"]}')"
	@python -c "import ukrainian_ocr; config = ukrainian_ocr.OCRConfig(); config.update_for_cpu(); print('✅ Configuration system works')"
	@echo "Package installation verified successfully!"

# Development workflow
dev-setup: install-dev
	@echo "Development environment set up!"
	@echo "Run 'make test' to run all tests"
	@echo "Run 'make verify-install' to verify the installation"

# Continuous Integration simulation
ci-test: check-deps verify-install test
	@echo "CI test pipeline completed successfully!"