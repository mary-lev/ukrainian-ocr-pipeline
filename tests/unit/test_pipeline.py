"""
Unit tests for main pipeline
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.core.pipeline import UkrainianOCRPipeline
from ukrainian_ocr.config import OCRPipelineConfig
from tests.fixtures.create_test_data import create_test_image


class TestUkrainianOCRPipeline(unittest.TestCase):
    """Test cases for UkrainianOCRPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OCRPipelineConfig()
        self.config.update_for_cpu()  # Ensure CPU testing
        
        self.pipeline = UkrainianOCRPipeline(
            config=self.config,
            device="cpu",
            verbose=False  # Reduce test output
        )
        
        # Create test image file
        self.test_image = create_test_image()
        self.temp_image_file = tempfile.NamedTemporaryFile(
            suffix='.png', 
            delete=False
        )
        
        import cv2
        cv2.imwrite(self.temp_image_file.name, self.test_image)
        self.temp_image_file.close()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_image_file.name):
            os.unlink(self.temp_image_file.name)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.device, "cpu")
        self.assertIsInstance(self.pipeline.config, OCRPipelineConfig)
        self.assertIsNotNone(self.pipeline.stats)
        
        # Components should be None initially (lazy loading)
        self.assertIsNone(self.pipeline.segmenter)
        self.assertIsNone(self.pipeline.ocr_processor)
        self.assertIsNone(self.pipeline.ner_extractor)
    
    def test_pipeline_initialization_with_dict_config(self):
        """Test pipeline initialization with dictionary config"""
        config_dict = {
            'device': 'cpu',
            'batch_size': 2,
            'ocr': {
                'model_path': 'test-model',
                'batch_size': 1
            }
        }
        
        pipeline = UkrainianOCRPipeline(config=config_dict, verbose=False)
        
        self.assertEqual(pipeline.config.device, 'cpu')
        self.assertEqual(pipeline.config.batch_size, 2)
        self.assertEqual(pipeline.config.ocr.model_path, 'test-model')
    
    def test_device_setup_auto(self):
        """Test automatic device setup"""
        pipeline = UkrainianOCRPipeline(device='auto', verbose=False)
        
        # Should choose cpu or cuda based on availability
        self.assertIn(pipeline.device, ['cpu', 'cuda'])
    
    def test_batch_size_determination(self):
        """Test automatic batch size determination"""
        # CPU pipeline should have batch_size = 1
        cpu_pipeline = UkrainianOCRPipeline(device='cpu', verbose=False)
        self.assertEqual(cpu_pipeline.batch_size, 1)
        
        # Auto pipeline should determine based on device
        auto_pipeline = UkrainianOCRPipeline(device='auto', verbose=False)
        self.assertGreaterEqual(auto_pipeline.batch_size, 1)
    
    def test_component_initialization_lazy_loading(self):
        """Test that components are loaded lazily"""
        pipeline = UkrainianOCRPipeline(verbose=False)
        
        # Components should be None initially
        self.assertIsNone(pipeline.segmenter)
        self.assertIsNone(pipeline.ocr_processor)
        self.assertIsNone(pipeline.ner_extractor)
        
        # Initialize components
        pipeline._init_components()
        
        # Now components should be loaded
        self.assertIsNotNone(pipeline.segmenter)
        self.assertIsNotNone(pipeline.ocr_processor)
        self.assertIsNotNone(pipeline.ner_extractor)
        self.assertIsNotNone(pipeline.alto_enhancer)
    
    def test_process_single_image_basic(self):
        """Test basic single image processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.pipeline.process_single_image(
                self.temp_image_file.name,
                output_dir=temp_dir,
                save_intermediate=False
            )
            
            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('image_path', result)
            self.assertIn('processing_time', result)
            self.assertIn('lines_detected', result)
            self.assertIn('lines_with_text', result)
            self.assertIn('output_paths', result)
            
            # Should be successful
            self.assertTrue(result['success'])
            
            # Should have reasonable values
            self.assertGreater(result['processing_time'], 0)
            self.assertGreaterEqual(result['lines_detected'], 0)
            self.assertGreaterEqual(result['lines_with_text'], 0)
    
    def test_process_single_image_with_intermediate(self):
        """Test single image processing with intermediate files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.pipeline.process_single_image(
                self.temp_image_file.name,
                output_dir=temp_dir,
                save_intermediate=True
            )
            
            self.assertTrue(result['success'])
            
            # Check that output paths are provided
            output_paths = result['output_paths']
            self.assertIsInstance(output_paths, dict)
    
    def test_process_single_image_nonexistent_file(self):
        """Test processing with non-existent image file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.pipeline.process_single_image(
                'non_existent_file.png',
                output_dir=temp_dir
            )
            
            # Should fail gracefully
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('processing_time', result)
    
    def test_process_batch_single_image(self):
        """Test batch processing with single image"""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.pipeline.process_batch(
                [self.temp_image_file.name],
                output_dir=temp_dir,
                save_intermediate=False
            )
            
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 1)
            
            result = results[0]
            self.assertTrue(result['success'])
    
    def test_process_batch_multiple_images(self):
        """Test batch processing with multiple images"""
        # Create second test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_image2 = f.name
        
        try:
            import cv2
            test_image2 = create_test_image(width=600, height=400)
            cv2.imwrite(temp_image2, test_image2)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                results = self.pipeline.process_batch(
                    [self.temp_image_file.name, temp_image2],
                    output_dir=temp_dir,
                    save_intermediate=False
                )
                
                self.assertEqual(len(results), 2)
                
                for result in results:
                    self.assertIn('success', result)
                    self.assertIn('processing_time', result)
        
        finally:
            if os.path.exists(temp_image2):
                os.unlink(temp_image2)
    
    def test_stats_tracking(self):
        """Test that statistics are tracked correctly"""
        initial_stats = self.pipeline.get_stats()
        
        self.assertEqual(initial_stats['images_processed'], 0)
        self.assertEqual(initial_stats['total_processing_time'], 0.0)
        self.assertEqual(initial_stats['average_time_per_image'], 0.0)
        
        # Process an image
        with tempfile.TemporaryDirectory() as temp_dir:
            self.pipeline.process_single_image(
                self.temp_image_file.name,
                output_dir=temp_dir
            )
        
        updated_stats = self.pipeline.get_stats()
        
        self.assertEqual(updated_stats['images_processed'], 1)
        self.assertGreater(updated_stats['total_processing_time'], 0)
        self.assertGreater(updated_stats['average_time_per_image'], 0)
    
    def test_cleanup(self):
        """Test pipeline cleanup"""
        # Should not raise exception
        self.pipeline.cleanup()
    
    def test_create_alto_xml_placeholder(self):
        """Test ALTO XML creation placeholder"""
        # This is a placeholder method, should not crash
        result = self.pipeline._create_alto_xml(
            Path(self.temp_image_file.name),
            self.test_image,
            []
        )
        
        # Currently returns None (placeholder)
        self.assertIsNone(result)
    
    def test_extract_person_regions_placeholder(self):
        """Test person region extraction placeholder"""
        # This is a placeholder method, should not crash
        result = self.pipeline._extract_person_regions(
            'dummy_alto.xml',
            Path(self.temp_image_file.name),
            Path('dummy_output')
        )
        
        # Currently returns None (placeholder)  
        self.assertIsNone(result)


class TestPipelineWithDifferentConfigs(unittest.TestCase):
    """Test pipeline with different configurations"""
    
    def test_colab_optimization(self):
        """Test Colab-optimized configuration"""
        config = OCRPipelineConfig()
        config.update_for_colab()
        
        pipeline = UkrainianOCRPipeline(config=config, verbose=False)
        
        # Should be optimized for available device
        self.assertIn(pipeline.device, ['cpu', 'cuda'])
        self.assertTrue(config.verbose)
        self.assertTrue(config.save_intermediate)
    
    def test_cpu_optimization(self):
        """Test CPU-optimized configuration"""
        config = OCRPipelineConfig()
        config.update_for_cpu()
        
        pipeline = UkrainianOCRPipeline(config=config, verbose=False)
        
        self.assertEqual(pipeline.device, 'cpu')
        self.assertEqual(pipeline.batch_size, 1)
        self.assertEqual(config.ocr.device, 'cpu')
    
    def test_custom_configuration(self):
        """Test pipeline with custom configuration"""
        config = OCRPipelineConfig(
            device='cpu',
            batch_size=3,
            verbose=False
        )
        
        config.ocr.model_path = 'custom-trocr-model'
        config.ner.backend = 'transformers'
        
        pipeline = UkrainianOCRPipeline(config=config, verbose=False)
        
        self.assertEqual(pipeline.config.ocr.model_path, 'custom-trocr-model')
        self.assertEqual(pipeline.config.ner.backend, 'transformers')


if __name__ == '__main__':
    unittest.main()