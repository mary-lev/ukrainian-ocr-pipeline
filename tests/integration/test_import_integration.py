"""
Integration tests for package imports and module availability
"""

import unittest
import sys
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPackageImports(unittest.TestCase):
    """Test that all package imports work correctly"""
    
    def test_main_package_import(self):
        """Test main package import"""
        import ukrainian_ocr
        
        # Should have version info
        self.assertTrue(hasattr(ukrainian_ocr, '__version__'))
        self.assertTrue(hasattr(ukrainian_ocr, '__author__'))
        self.assertIsInstance(ukrainian_ocr.__version__, str)
    
    def test_core_components_import(self):
        """Test core component imports"""
        from ukrainian_ocr import (
            UkrainianOCRPipeline,
            OCRConfig,
            BatchProcessor,
            KrakenSegmenter,
            TrOCRProcessor,
            NERExtractor,
            ALTOEnhancer
        )
        
        # Should be importable classes
        self.assertTrue(callable(UkrainianOCRPipeline))
        self.assertTrue(callable(OCRConfig))
        self.assertTrue(callable(BatchProcessor))
        self.assertTrue(callable(KrakenSegmenter))
        self.assertTrue(callable(TrOCRProcessor))
        self.assertTrue(callable(NERExtractor))
        self.assertTrue(callable(ALTOEnhancer))
    
    def test_utility_imports(self):
        """Test utility function imports"""
        from ukrainian_ocr import (
            check_gpu_availability,
            optimize_for_device,
            ModelManager
        )
        
        # Should be importable functions/classes
        self.assertTrue(callable(check_gpu_availability))
        self.assertTrue(callable(optimize_for_device))
        self.assertTrue(callable(ModelManager))
    
    def test_submodule_imports(self):
        """Test submodule imports"""
        # Core modules
        from ukrainian_ocr.core import (
            pipeline,
            segmentation,
            ocr,
            ner,
            enhancement,
            batch_processor
        )
        
        # Utils modules
        from ukrainian_ocr.utils import (
            gpu,
            io,
            models,
            visualization
        )
        
        # Configuration
        from ukrainian_ocr import config
        
        # All should be modules
        import types
        self.assertIsInstance(pipeline, types.ModuleType)
        self.assertIsInstance(gpu, types.ModuleType)
        self.assertIsInstance(config, types.ModuleType)
    
    def test_direct_class_imports(self):
        """Test direct class imports from submodules"""
        # Core classes
        from ukrainian_ocr.core.pipeline import UkrainianOCRPipeline
        from ukrainian_ocr.core.segmentation import KrakenSegmenter
        from ukrainian_ocr.core.ocr import TrOCRProcessor
        from ukrainian_ocr.core.ner import NERExtractor
        from ukrainian_ocr.core.enhancement import ALTOEnhancer
        from ukrainian_ocr.core.batch_processor import BatchProcessor
        
        # Utility classes
        from ukrainian_ocr.utils.gpu import check_gpu_availability
        from ukrainian_ocr.utils.io import IOUtils
        from ukrainian_ocr.utils.models import ModelManager
        from ukrainian_ocr.utils.visualization import Visualizer
        
        # Config classes
        from ukrainian_ocr.config import (
            OCRPipelineConfig,
            SegmentationConfig,
            OCRProcessorConfig,
            NERConfig,
            PostProcessingConfig
        )
        
        # All should be classes or functions
        self.assertTrue(callable(UkrainianOCRPipeline))
        self.assertTrue(callable(KrakenSegmenter))
        self.assertTrue(callable(TrOCRProcessor))
        self.assertTrue(callable(NERExtractor))
        self.assertTrue(callable(ALTOEnhancer))
        self.assertTrue(callable(BatchProcessor))
        self.assertTrue(callable(check_gpu_availability))
        self.assertTrue(callable(IOUtils))
        self.assertTrue(callable(ModelManager))
        self.assertTrue(callable(Visualizer))
        self.assertTrue(callable(OCRPipelineConfig))
    
    def test_backward_compatibility_imports(self):
        """Test backward compatibility imports"""
        # OCRConfig should be an alias for OCRPipelineConfig
        from ukrainian_ocr.config import OCRConfig, OCRPipelineConfig
        
        self.assertIs(OCRConfig, OCRPipelineConfig)
    
    def test_all_exports(self):
        """Test that __all__ exports are correct"""
        import ukrainian_ocr
        
        # Should have __all__ defined
        self.assertTrue(hasattr(ukrainian_ocr, '__all__'))
        self.assertIsInstance(ukrainian_ocr.__all__, list)
        
        # All listed items should be importable
        for item_name in ukrainian_ocr.__all__:
            self.assertTrue(
                hasattr(ukrainian_ocr, item_name),
                f"Item '{item_name}' in __all__ but not available in module"
            )


class TestClassInstantiation(unittest.TestCase):
    """Test that classes can be instantiated without errors"""
    
    def test_config_instantiation(self):
        """Test configuration class instantiation"""
        from ukrainian_ocr.config import (
            OCRPipelineConfig,
            SegmentationConfig,
            OCRProcessorConfig,
            NERConfig,
            PostProcessingConfig
        )
        
        # Should be able to create instances
        main_config = OCRPipelineConfig()
        seg_config = SegmentationConfig()
        ocr_config = OCRProcessorConfig()
        ner_config = NERConfig()
        post_config = PostProcessingConfig()
        
        # Should have expected attributes
        self.assertEqual(main_config.device, "auto")
        self.assertEqual(seg_config.model_path, "kraken_model/blla.mlmodel")
        self.assertEqual(ocr_config.batch_size, 4)
        self.assertEqual(ner_config.backend, "spacy")
        self.assertTrue(post_config.extract_person_regions)
    
    def test_pipeline_instantiation(self):
        """Test pipeline instantiation"""
        from ukrainian_ocr import UkrainianOCRPipeline, OCRConfig
        
        # Should be able to create pipeline with default config
        pipeline = UkrainianOCRPipeline(verbose=False)
        self.assertIsInstance(pipeline, UkrainianOCRPipeline)
        
        # Should be able to create with custom config
        config = OCRConfig()
        config.update_for_cpu()
        
        pipeline_custom = UkrainianOCRPipeline(config=config, verbose=False)
        self.assertIsInstance(pipeline_custom, UkrainianOCRPipeline)
        self.assertEqual(pipeline_custom.device, 'cpu')
    
    def test_component_instantiation(self):
        """Test individual component instantiation"""
        from ukrainian_ocr.core.segmentation import KrakenSegmenter
        from ukrainian_ocr.core.ocr import TrOCRProcessor
        from ukrainian_ocr.core.ner import NERExtractor
        from ukrainian_ocr.core.enhancement import ALTOEnhancer
        from ukrainian_ocr.core.batch_processor import BatchProcessor
        
        # Should be able to create components
        segmenter = KrakenSegmenter()
        ocr_processor = TrOCRProcessor(device="cpu", batch_size=1)
        ner_extractor = NERExtractor()
        alto_enhancer = ALTOEnhancer()
        batch_processor = BatchProcessor()
        
        # Should have expected attributes
        self.assertIsInstance(segmenter, KrakenSegmenter)
        self.assertIsInstance(ocr_processor, TrOCRProcessor)
        self.assertIsInstance(ner_extractor, NERExtractor)
        self.assertIsInstance(alto_enhancer, ALTOEnhancer)
        self.assertIsInstance(batch_processor, BatchProcessor)
    
    def test_utility_instantiation(self):
        """Test utility class instantiation"""
        from ukrainian_ocr.utils.io import IOUtils
        from ukrainian_ocr.utils.models import ModelManager
        from ukrainian_ocr.utils.visualization import Visualizer
        
        # Should be able to create utility instances
        io_utils = IOUtils()
        model_manager = ModelManager()
        visualizer = Visualizer()
        
        self.assertIsInstance(io_utils, IOUtils)
        self.assertIsInstance(model_manager, ModelManager)
        self.assertIsInstance(visualizer, Visualizer)
    
    def test_function_calls(self):
        """Test that utility functions can be called"""
        from ukrainian_ocr.utils.gpu import check_gpu_availability, optimize_for_device
        
        # Should be able to call GPU utilities
        gpu_info = check_gpu_availability()
        self.assertIsInstance(gpu_info, dict)
        self.assertIn('cuda_available', gpu_info)
        
        # Should not raise exception
        optimize_for_device('cpu')


class TestModuleReloading(unittest.TestCase):
    """Test module reloading and import consistency"""
    
    def test_module_reload(self):
        """Test that modules can be reloaded without issues"""
        import ukrainian_ocr
        
        # Reload the module
        importlib.reload(ukrainian_ocr)
        
        # Should still work
        from ukrainian_ocr import UkrainianOCRPipeline
        pipeline = UkrainianOCRPipeline(verbose=False)
        self.assertIsInstance(pipeline, UkrainianOCRPipeline)
    
    def test_import_consistency(self):
        """Test that multiple imports give consistent results"""
        # Import multiple ways
        import ukrainian_ocr as ua1
        from ukrainian_ocr import UkrainianOCRPipeline as Pipeline1
        
        import ukrainian_ocr as ua2
        from ukrainian_ocr import UkrainianOCRPipeline as Pipeline2
        
        # Should be the same
        self.assertIs(ua1, ua2)
        self.assertIs(Pipeline1, Pipeline2)
    
    def test_submodule_import_consistency(self):
        """Test submodule import consistency"""
        # Import config in different ways
        from ukrainian_ocr import config as config1
        from ukrainian_ocr.config import OCRPipelineConfig as Config1
        
        import ukrainian_ocr.config as config2
        from ukrainian_ocr.config import OCRPipelineConfig as Config2
        
        # Should be consistent
        self.assertIs(config1, config2)
        self.assertIs(Config1, Config2)


class TestOptionalDependencies(unittest.TestCase):
    """Test behavior with optional dependencies"""
    
    def test_missing_optional_dependencies(self):
        """Test that missing optional dependencies are handled gracefully"""
        # The package should work even if optional dependencies are missing
        # This tests the graceful degradation behavior
        
        from ukrainian_ocr.core.ner import NERExtractor
        from ukrainian_ocr.core.segmentation import KrakenSegmenter
        
        # Should create instances even if models aren't available
        ner = NERExtractor(backend="spacy")
        segmenter = KrakenSegmenter()
        
        # Should use placeholder implementations
        self.assertIsInstance(ner, NERExtractor)
        self.assertIsInstance(segmenter, KrakenSegmenter)
    
    def test_gpu_utilities_without_cuda(self):
        """Test GPU utilities work without CUDA"""
        from ukrainian_ocr.utils.gpu import (
            check_gpu_availability,
            optimize_for_device,
            get_optimal_batch_size
        )
        
        # Should work even without CUDA
        gpu_info = check_gpu_availability()
        self.assertIsInstance(gpu_info, dict)
        self.assertIn('cuda_available', gpu_info)
        
        # Should handle CPU-only case
        optimize_for_device('cpu')
        batch_size = get_optimal_batch_size('cpu')
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)


class TestPackageMetadata(unittest.TestCase):
    """Test package metadata and structure"""
    
    def test_package_metadata(self):
        """Test that package has proper metadata"""
        import ukrainian_ocr
        
        # Should have version info
        self.assertTrue(hasattr(ukrainian_ocr, '__version__'))
        self.assertTrue(hasattr(ukrainian_ocr, '__author__'))
        self.assertTrue(hasattr(ukrainian_ocr, '__package_name__'))
        self.assertTrue(hasattr(ukrainian_ocr, '__description__'))
        
        # Check types
        self.assertIsInstance(ukrainian_ocr.__version__, str)
        self.assertIsInstance(ukrainian_ocr.__author__, str)
        self.assertIsInstance(ukrainian_ocr.__package_name__, str)
        self.assertIsInstance(ukrainian_ocr.__description__, str)
        
        # Check values make sense
        self.assertIn('.', ukrainian_ocr.__version__)  # Should have version format
        self.assertEqual(ukrainian_ocr.__package_name__, 'ukrainian-ocr-pipeline')
    
    def test_module_docstrings(self):
        """Test that modules have proper docstrings"""
        import ukrainian_ocr
        from ukrainian_ocr.core import pipeline, segmentation, ocr
        from ukrainian_ocr.utils import gpu, io
        
        # Main package should have docstring
        self.assertIsNotNone(ukrainian_ocr.__doc__)
        
        # Core modules should have docstrings
        self.assertIsNotNone(pipeline.__doc__)
        self.assertIsNotNone(segmentation.__doc__)
        self.assertIsNotNone(ocr.__doc__)
        
        # Utility modules should have docstrings
        self.assertIsNotNone(gpu.__doc__)
        self.assertIsNotNone(io.__doc__)


if __name__ == '__main__':
    unittest.main(verbosity=2)