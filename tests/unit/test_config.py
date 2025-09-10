"""
Unit tests for configuration module
"""

import unittest
import tempfile
import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.config import (
    SegmentationConfig,
    OCRProcessorConfig,
    NERConfig,
    PostProcessingConfig,
    OCRPipelineConfig
)


class TestComponentConfigs(unittest.TestCase):
    """Test individual component configurations"""
    
    def test_segmentation_config_defaults(self):
        """Test SegmentationConfig default values"""
        config = SegmentationConfig()
        
        self.assertEqual(config.model_path, "kraken_model/blla.mlmodel")
        self.assertIsNone(config.device)
    
    def test_segmentation_config_custom(self):
        """Test SegmentationConfig with custom values"""
        config = SegmentationConfig(
            model_path="custom_model.mlmodel",
            device="cpu"
        )
        
        self.assertEqual(config.model_path, "custom_model.mlmodel")
        self.assertEqual(config.device, "cpu")
    
    def test_ocr_processor_config_defaults(self):
        """Test OCRProcessorConfig default values"""
        config = OCRProcessorConfig()
        
        self.assertEqual(config.model_path, "cyrillic-trocr/trocr-handwritten-cyrillic")
        self.assertIsNone(config.device)
        self.assertEqual(config.batch_size, 4)
        self.assertFalse(config.preprocessing)
        self.assertEqual(config.num_beams, 1)
        self.assertEqual(config.max_length, 256)
    
    def test_ocr_processor_config_custom(self):
        """Test OCRProcessorConfig with custom values"""
        config = OCRProcessorConfig(
            model_path="custom-trocr-model",
            device="cuda",
            batch_size=8,
            preprocessing=True,
            num_beams=5,
            max_length=512
        )
        
        self.assertEqual(config.model_path, "custom-trocr-model")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.batch_size, 8)
        self.assertTrue(config.preprocessing)
        self.assertEqual(config.num_beams, 5)
        self.assertEqual(config.max_length, 512)
    
    def test_ner_config_defaults(self):
        """Test NERConfig default values"""
        config = NERConfig()
        
        self.assertEqual(config.backend, "spacy")
        self.assertEqual(config.model_name, "uk_core_news_sm")
        self.assertEqual(config.confidence_threshold, 0.5)
    
    def test_ner_config_custom(self):
        """Test NERConfig with custom values"""
        config = NERConfig(
            backend="transformers",
            model_name="bert-base-multilingual-cased",
            confidence_threshold=0.8
        )
        
        self.assertEqual(config.backend, "transformers")
        self.assertEqual(config.model_name, "bert-base-multilingual-cased")
        self.assertEqual(config.confidence_threshold, 0.8)
    
    def test_post_processing_config_defaults(self):
        """Test PostProcessingConfig default values"""
        config = PostProcessingConfig()
        
        self.assertTrue(config.extract_person_regions)
        self.assertEqual(config.clustering_eps, 300)
        self.assertEqual(config.min_samples, 3)
        self.assertEqual(config.region_padding, 50)
    
    def test_post_processing_config_custom(self):
        """Test PostProcessingConfig with custom values"""
        config = PostProcessingConfig(
            extract_person_regions=False,
            clustering_eps=200,
            min_samples=5,
            region_padding=100
        )
        
        self.assertFalse(config.extract_person_regions)
        self.assertEqual(config.clustering_eps, 200)
        self.assertEqual(config.min_samples, 5)
        self.assertEqual(config.region_padding, 100)


class TestOCRPipelineConfig(unittest.TestCase):
    """Test main pipeline configuration"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        config = OCRPipelineConfig()
        
        # Check main settings
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.batch_size, 4)
        self.assertTrue(config.verbose)
        self.assertTrue(config.save_intermediate)
        
        # Check component configs exist
        self.assertIsInstance(config.segmentation, SegmentationConfig)
        self.assertIsInstance(config.ocr, OCRProcessorConfig)
        self.assertIsInstance(config.ner, NERConfig)
        self.assertIsInstance(config.post_processing, PostProcessingConfig)
    
    def test_custom_initialization(self):
        """Test initialization with custom values"""
        custom_segmentation = SegmentationConfig(model_path="custom.mlmodel")
        custom_ocr = OCRProcessorConfig(batch_size=8)
        custom_ner = NERConfig(backend="transformers")
        custom_post = PostProcessingConfig(extract_person_regions=False)
        
        config = OCRPipelineConfig(
            segmentation=custom_segmentation,
            ocr=custom_ocr,
            ner=custom_ner,
            post_processing=custom_post,
            device="cuda",
            batch_size=16,
            verbose=False,
            save_intermediate=False
        )
        
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.batch_size, 16)
        self.assertFalse(config.verbose)
        self.assertFalse(config.save_intermediate)
        
        # Check custom component configs
        self.assertEqual(config.segmentation.model_path, "custom.mlmodel")
        self.assertEqual(config.ocr.batch_size, 8)
        self.assertEqual(config.ner.backend, "transformers")
        self.assertFalse(config.post_processing.extract_person_regions)
    
    def test_from_dict(self):
        """Test configuration from dictionary"""
        config_dict = {
            'device': 'cpu',
            'batch_size': 2,
            'verbose': False,
            'save_intermediate': True,
            'segmentation': {
                'model_path': 'test_segmentation.mlmodel',
                'device': 'cpu'
            },
            'ocr': {
                'model_path': 'test-trocr-model',
                'batch_size': 1,
                'preprocessing': True
            },
            'ner': {
                'backend': 'transformers',
                'model_name': 'bert-multilingual',
                'confidence_threshold': 0.7
            },
            'post_processing': {
                'extract_person_regions': False,
                'clustering_eps': 250
            }
        }
        
        config = OCRPipelineConfig.from_dict(config_dict)
        
        # Check main settings
        self.assertEqual(config.device, 'cpu')
        self.assertEqual(config.batch_size, 2)
        self.assertFalse(config.verbose)
        self.assertTrue(config.save_intermediate)
        
        # Check component settings
        self.assertEqual(config.segmentation.model_path, 'test_segmentation.mlmodel')
        self.assertEqual(config.segmentation.device, 'cpu')
        
        self.assertEqual(config.ocr.model_path, 'test-trocr-model')
        self.assertEqual(config.ocr.batch_size, 1)
        self.assertTrue(config.ocr.preprocessing)
        
        self.assertEqual(config.ner.backend, 'transformers')
        self.assertEqual(config.ner.model_name, 'bert-multilingual')
        self.assertEqual(config.ner.confidence_threshold, 0.7)
        
        self.assertFalse(config.post_processing.extract_person_regions)
        self.assertEqual(config.post_processing.clustering_eps, 250)
    
    def test_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = OCRPipelineConfig(device='cpu', batch_size=2)
        config_dict = config.to_dict()
        
        # Check structure
        self.assertIsInstance(config_dict, dict)
        
        required_keys = [
            'segmentation', 'ocr', 'ner', 'post_processing',
            'device', 'batch_size', 'verbose', 'save_intermediate'
        ]
        
        for key in required_keys:
            self.assertIn(key, config_dict)
        
        # Check values
        self.assertEqual(config_dict['device'], 'cpu')
        self.assertEqual(config_dict['batch_size'], 2)
        
        # Check component dicts
        self.assertIsInstance(config_dict['segmentation'], dict)
        self.assertIsInstance(config_dict['ocr'], dict)
        self.assertIsInstance(config_dict['ner'], dict)
        self.assertIsInstance(config_dict['post_processing'], dict)
    
    def test_round_trip_dict_conversion(self):
        """Test that dict conversion is lossless"""
        original_config = OCRPipelineConfig(
            device='cuda',
            batch_size=8,
            verbose=False
        )
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = OCRPipelineConfig.from_dict(config_dict)
        
        # Should be identical
        self.assertEqual(original_config.device, restored_config.device)
        self.assertEqual(original_config.batch_size, restored_config.batch_size)
        self.assertEqual(original_config.verbose, restored_config.verbose)
        
        # Check component configs
        self.assertEqual(
            original_config.segmentation.model_path,
            restored_config.segmentation.model_path
        )
        self.assertEqual(
            original_config.ocr.batch_size,
            restored_config.ocr.batch_size
        )


class TestConfigurationFileOperations(unittest.TestCase):
    """Test configuration file operations"""
    
    def setUp(self):
        """Set up test configuration"""
        self.test_config = OCRPipelineConfig(
            device='cpu',
            batch_size=2,
            verbose=False
        )
        
        self.test_config.ocr.model_path = 'test-model'
        self.test_config.ner.backend = 'transformers'
    
    def test_save_load_json(self):
        """Test saving and loading JSON configuration"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            # Save configuration
            self.test_config.save_to_file(json_path)
            self.assertTrue(os.path.exists(json_path))
            
            # Load configuration
            loaded_config = OCRPipelineConfig.from_file(json_path)
            
            # Verify values
            self.assertEqual(loaded_config.device, 'cpu')
            self.assertEqual(loaded_config.batch_size, 2)
            self.assertFalse(loaded_config.verbose)
            self.assertEqual(loaded_config.ocr.model_path, 'test-model')
            self.assertEqual(loaded_config.ner.backend, 'transformers')
            
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    def test_save_load_yaml(self):
        """Test saving and loading YAML configuration"""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Save configuration
            self.test_config.save_to_file(yaml_path)
            self.assertTrue(os.path.exists(yaml_path))
            
            # Load configuration  
            loaded_config = OCRPipelineConfig.from_file(yaml_path)
            
            # Verify values
            self.assertEqual(loaded_config.device, 'cpu')
            self.assertEqual(loaded_config.batch_size, 2)
            self.assertEqual(loaded_config.ocr.model_path, 'test-model')
            
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file"""
        with self.assertRaises(FileNotFoundError):
            OCRPipelineConfig.from_file('non_existent_config.json')
    
    def test_save_create_directory(self):
        """Test saving configuration with directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'subdir', 'config.json')
            
            # Should create directory and save
            self.test_config.save_to_file(config_path)
            
            self.assertTrue(os.path.exists(config_path))
            
            # Should be able to load
            loaded_config = OCRPipelineConfig.from_file(config_path)
            self.assertEqual(loaded_config.device, 'cpu')
    
    def test_invalid_file_format(self):
        """Test handling of invalid file formats"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            txt_path = f.name
        
        try:
            # Should raise error for unsupported format
            with self.assertRaises(ValueError):
                self.test_config.save_to_file(txt_path)
                
            with self.assertRaises(ValueError):
                OCRPipelineConfig.from_file(txt_path)
                
        finally:
            if os.path.exists(txt_path):
                os.unlink(txt_path)


class TestConfigurationOptimizations(unittest.TestCase):
    """Test configuration optimization methods"""
    
    def test_update_for_colab(self):
        """Test Colab optimization"""
        config = OCRPipelineConfig()
        config.update_for_colab()
        
        # Should set appropriate device
        self.assertIn(config.device, ['cpu', 'cuda'])
        
        # Should optimize settings
        self.assertTrue(config.verbose)
        self.assertTrue(config.save_intermediate)
        self.assertFalse(config.ocr.preprocessing)
        self.assertEqual(config.ocr.num_beams, 1)
        
        # Batch size should be reasonable
        self.assertGreaterEqual(config.batch_size, 1)
        self.assertLessEqual(config.batch_size, 8)
    
    def test_update_for_cpu(self):
        """Test CPU optimization"""
        config = OCRPipelineConfig()
        config.update_for_cpu()
        
        # Should set CPU device
        self.assertEqual(config.device, 'cpu')
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.ocr.device, 'cpu')
        self.assertEqual(config.ocr.batch_size, 1)
        self.assertEqual(config.segmentation.device, 'cpu')
    
    def test_update_for_gpu(self):
        """Test GPU optimization"""
        config = OCRPipelineConfig()
        
        # Test with different GPU memory sizes
        memory_sizes = [4.0, 8.0, 16.0]
        
        for memory_gb in memory_sizes:
            config.update_for_gpu(memory_gb)
            
            self.assertEqual(config.device, 'cuda')
            self.assertEqual(config.ocr.device, 'cuda')
            self.assertEqual(config.segmentation.device, 'cuda')
            
            # Batch size should scale with memory
            self.assertGreaterEqual(config.batch_size, 1)
            
            if memory_gb >= 16:
                self.assertEqual(config.batch_size, 8)
            elif memory_gb >= 8:
                self.assertEqual(config.batch_size, 4)
            else:
                self.assertEqual(config.batch_size, 2)
    
    def test_update_for_gpu_auto_detect(self):
        """Test GPU optimization with auto-detection"""
        config = OCRPipelineConfig()
        config.update_for_gpu()  # No memory specified
        
        self.assertEqual(config.device, 'cuda')
        self.assertGreaterEqual(config.batch_size, 1)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility features"""
    
    def test_ocr_config_alias(self):
        """Test that OCRConfig is an alias for OCRPipelineConfig"""
        from ukrainian_ocr.config import OCRConfig
        
        # Should be the same class
        self.assertIs(OCRConfig, OCRPipelineConfig)
        
        # Should work identically
        config1 = OCRConfig()
        config2 = OCRPipelineConfig()
        
        self.assertEqual(config1.device, config2.device)
        self.assertEqual(config1.batch_size, config2.batch_size)
        self.assertEqual(config1.ocr.model_path, config2.ocr.model_path)


if __name__ == '__main__':
    unittest.main()