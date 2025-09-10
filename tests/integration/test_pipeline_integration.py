"""
Integration tests for complete pipeline workflows
"""

import unittest
import tempfile
import os
import sys
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr import UkrainianOCRPipeline, OCRConfig
from ukrainian_ocr.utils.gpu import check_gpu_availability
from tests.fixtures.create_test_data import create_test_image, create_sample_lines_data


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete OCR pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire class"""
        # Create test images
        cls.test_images = {}
        
        # Standard test image
        cls.test_images['standard'] = create_test_image(800, 600, 5)
        
        # Small image
        cls.test_images['small'] = create_test_image(400, 300, 3)
        
        # Large image
        cls.test_images['large'] = create_test_image(1200, 900, 8)
        
        # Empty image
        cls.test_images['empty'] = create_test_image(400, 200, 0)
        
        # Save images to temporary files
        cls.temp_image_files = {}
        for name, image in cls.test_images.items():
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f'_{name}.png',
                delete=False
            )
            cv2.imwrite(temp_file.name, image)
            temp_file.close()
            cls.temp_image_files[name] = temp_file.name
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        for temp_file in cls.temp_image_files.values():
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def setUp(self):
        """Set up for each test"""
        # Create CPU-optimized pipeline for consistent testing
        self.config = OCRConfig()
        self.config.update_for_cpu()
        
        self.pipeline = UkrainianOCRPipeline(
            config=self.config,
            device='cpu',
            verbose=False  # Reduce test output
        )
    
    def test_complete_single_image_workflow(self):
        """Test complete workflow for a single image"""
        with tempfile.TemporaryDirectory() as output_dir:
            result = self.pipeline.process_single_image(
                self.temp_image_files['standard'],
                output_dir=output_dir,
                save_intermediate=True
            )
            
            # Check result structure
            self.assertTrue(result['success'])
            self.assertIn('processing_time', result)
            self.assertIn('lines_detected', result)
            self.assertIn('lines_with_text', result)
            self.assertIn('output_paths', result)
            
            # Check processing metrics
            self.assertGreater(result['processing_time'], 0)
            self.assertGreaterEqual(result['lines_detected'], 0)
            self.assertGreaterEqual(result['lines_with_text'], 0)
            
            # Check output paths
            output_paths = result['output_paths']
            self.assertIsInstance(output_paths, dict)
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow"""
        image_paths = [
            self.temp_image_files['standard'],
            self.temp_image_files['small']
        ]
        
        with tempfile.TemporaryDirectory() as output_dir:
            results = self.pipeline.process_batch(
                image_paths,
                output_dir=output_dir,
                save_intermediate=False
            )
            
            # Should return results for all images
            self.assertEqual(len(results), len(image_paths))
            
            # Check each result
            for i, result in enumerate(results):
                self.assertIn('success', result)
                self.assertIn('image_path', result)
                self.assertEqual(
                    result['image_path'],
                    str(Path(image_paths[i]).absolute())
                )
                
                # Should have processing metrics
                self.assertIn('processing_time', result)
                self.assertIn('lines_detected', result)
    
    def test_different_image_sizes(self):
        """Test pipeline with different image sizes"""
        test_cases = ['small', 'standard', 'large']
        
        with tempfile.TemporaryDirectory() as output_dir:
            for image_name in test_cases:
                with self.subTest(image=image_name):
                    result = self.pipeline.process_single_image(
                        self.temp_image_files[image_name],
                        output_dir=output_dir
                    )
                    
                    # Should succeed regardless of size
                    self.assertTrue(result['success'], f"Failed for {image_name} image")
                    
                    # Should have reasonable processing time
                    self.assertGreater(result['processing_time'], 0)
    
    def test_empty_image_handling(self):
        """Test pipeline with empty/blank image"""
        with tempfile.TemporaryDirectory() as output_dir:
            result = self.pipeline.process_single_image(
                self.temp_image_files['empty'],
                output_dir=output_dir
            )
            
            # Should handle gracefully
            self.assertTrue(result['success'])
            
            # May have few or no lines detected
            self.assertGreaterEqual(result['lines_detected'], 0)
    
    def test_error_recovery(self):
        """Test pipeline error recovery"""
        with tempfile.TemporaryDirectory() as output_dir:
            # Test with non-existent file
            result = self.pipeline.process_single_image(
                'non_existent_image.png',
                output_dir=output_dir
            )
            
            # Should fail gracefully
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('processing_time', result)
    
    def test_component_initialization_workflow(self):
        """Test that components are properly initialized during workflow"""
        # Components should be None initially
        self.assertIsNone(self.pipeline.segmenter)
        self.assertIsNone(self.pipeline.ocr_processor)
        self.assertIsNone(self.pipeline.ner_extractor)
        
        with tempfile.TemporaryDirectory() as output_dir:
            # Process an image
            result = self.pipeline.process_single_image(
                self.temp_image_files['standard'],
                output_dir=output_dir
            )
            
            # Components should now be initialized
            self.assertIsNotNone(self.pipeline.segmenter)
            self.assertIsNotNone(self.pipeline.ocr_processor)
            self.assertIsNotNone(self.pipeline.ner_extractor)
            self.assertIsNotNone(self.pipeline.alto_enhancer)
            
            self.assertTrue(result['success'])
    
    def test_statistics_tracking_integration(self):
        """Test that statistics are properly tracked across operations"""
        initial_stats = self.pipeline.get_stats()
        
        self.assertEqual(initial_stats['images_processed'], 0)
        self.assertEqual(initial_stats['total_processing_time'], 0.0)
        
        with tempfile.TemporaryDirectory() as output_dir:
            # Process multiple images
            image_paths = [
                self.temp_image_files['standard'],
                self.temp_image_files['small']
            ]
            
            results = self.pipeline.process_batch(
                image_paths,
                output_dir=output_dir
            )
            
            # Check statistics were updated
            final_stats = self.pipeline.get_stats()
            
            self.assertEqual(final_stats['images_processed'], 2)
            self.assertGreater(final_stats['total_processing_time'], 0)
            self.assertGreater(final_stats['average_time_per_image'], 0)
    
    def test_configuration_propagation(self):
        """Test that configuration is properly propagated to components"""
        # Create custom configuration
        custom_config = OCRConfig()
        custom_config.update_for_cpu()
        custom_config.ocr.batch_size = 2
        custom_config.ner.backend = 'transformers'
        custom_config.ner.confidence_threshold = 0.8
        
        custom_pipeline = UkrainianOCRPipeline(
            config=custom_config,
            device='cpu',
            verbose=False
        )
        
        # Initialize components
        custom_pipeline._init_components()
        
        # Check that configuration was passed to components
        self.assertEqual(custom_pipeline.ocr_processor.batch_size, 2)
        self.assertEqual(custom_pipeline.ner_extractor.backend, 'transformers')


class TestPipelineWithDifferentConfigurations(unittest.TestCase):
    """Test pipeline with different configuration setups"""
    
    def setUp(self):
        """Set up test image"""
        self.test_image = create_test_image()
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(self.temp_file.name, self.test_image)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test image"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_cpu_optimized_pipeline(self):
        """Test CPU-optimized pipeline configuration"""
        config = OCRConfig()
        config.update_for_cpu()
        
        pipeline = UkrainianOCRPipeline(
            config=config,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as output_dir:
            result = pipeline.process_single_image(
                self.temp_file.name,
                output_dir=output_dir
            )
            
            self.assertTrue(result['success'])
            self.assertEqual(pipeline.device, 'cpu')
            self.assertEqual(pipeline.batch_size, 1)
    
    def test_colab_optimized_pipeline(self):
        """Test Colab-optimized pipeline configuration"""
        config = OCRConfig()
        config.update_for_colab()
        
        pipeline = UkrainianOCRPipeline(
            config=config,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as output_dir:
            result = pipeline.process_single_image(
                self.temp_file.name,
                output_dir=output_dir
            )
            
            self.assertTrue(result['success'])
            # Device should be optimally chosen
            self.assertIn(pipeline.device, ['cpu', 'cuda'])
    
    def test_custom_configuration_pipeline(self):
        """Test pipeline with custom configuration"""
        config = OCRConfig(
            device='cpu',
            batch_size=3,
            verbose=False
        )
        
        # Customize component settings
        config.ocr.preprocessing = True
        config.ner.confidence_threshold = 0.7
        config.post_processing.extract_person_regions = False
        
        pipeline = UkrainianOCRPipeline(
            config=config,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as output_dir:
            result = pipeline.process_single_image(
                self.temp_file.name,
                output_dir=output_dir
            )
            
            self.assertTrue(result['success'])
            self.assertEqual(pipeline.batch_size, 3)
            self.assertEqual(pipeline.config.ner.confidence_threshold, 0.7)
    
    def test_dictionary_configuration_pipeline(self):
        """Test pipeline initialized with dictionary configuration"""
        config_dict = {
            'device': 'cpu',
            'batch_size': 1,
            'verbose': False,
            'ocr': {
                'batch_size': 1,
                'preprocessing': False
            },
            'ner': {
                'backend': 'spacy',
                'confidence_threshold': 0.6
            }
        }
        
        pipeline = UkrainianOCRPipeline(
            config=config_dict,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as output_dir:
            result = pipeline.process_single_image(
                self.temp_file.name,
                output_dir=output_dir
            )
            
            self.assertTrue(result['success'])
            self.assertEqual(pipeline.config.ner.confidence_threshold, 0.6)


class TestPipelinePerformance(unittest.TestCase):
    """Performance and load tests for the pipeline"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.config = OCRConfig()
        self.config.update_for_cpu()  # Consistent testing
        
        self.pipeline = UkrainianOCRPipeline(
            config=self.config,
            verbose=False
        )
        
        # Create test images
        self.test_images = []
        for i in range(3):  # Small batch for testing
            image = create_test_image(600, 400, 4)
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f'_perf_{i}.png',
                delete=False
            )
            cv2.imwrite(temp_file.name, image)
            temp_file.close()
            self.test_images.append(temp_file.name)
    
    def tearDown(self):
        """Clean up performance test fixtures"""
        for img_path in self.test_images:
            if os.path.exists(img_path):
                os.unlink(img_path)
    
    def test_batch_processing_performance(self):
        """Test batch processing performance metrics"""
        with tempfile.TemporaryDirectory() as output_dir:
            start_time = self.pipeline.get_stats()['total_processing_time']
            
            results = self.pipeline.process_batch(
                self.test_images,
                output_dir=output_dir
            )
            
            end_stats = self.pipeline.get_stats()
            processing_time = end_stats['total_processing_time'] - start_time
            
            # Check performance metrics
            self.assertEqual(len(results), len(self.test_images))
            self.assertGreater(processing_time, 0)
            
            # All images should be processed successfully
            successful = sum(1 for r in results if r['success'])
            self.assertEqual(successful, len(self.test_images))
            
            # Average time should be reasonable
            avg_time = end_stats['average_time_per_image']
            self.assertGreater(avg_time, 0)
            self.assertLess(avg_time, 300)  # Less than 5 minutes per image
    
    def test_memory_management(self):
        """Test that memory usage is reasonable"""
        initial_stats = self.pipeline.get_stats()
        
        with tempfile.TemporaryDirectory() as output_dir:
            # Process multiple images to test memory management
            for img_path in self.test_images:
                result = self.pipeline.process_single_image(
                    img_path,
                    output_dir=output_dir
                )
                self.assertTrue(result['success'])
        
        # Clean up (should not raise exceptions)
        self.pipeline.cleanup()
        
        final_stats = self.pipeline.get_stats()
        self.assertEqual(
            final_stats['images_processed'],
            initial_stats['images_processed'] + len(self.test_images)
        )
    
    def test_concurrent_processing_safety(self):
        """Test that pipeline handles concurrent-like operations safely"""
        with tempfile.TemporaryDirectory() as output_dir:
            # Simulate rapid successive processing
            results = []
            for img_path in self.test_images:
                result = self.pipeline.process_single_image(
                    img_path,
                    output_dir=output_dir
                )
                results.append(result)
            
            # All should succeed
            for result in results:
                self.assertTrue(result['success'])
            
            # Statistics should be consistent
            stats = self.pipeline.get_stats()
            self.assertEqual(stats['images_processed'], len(self.test_images))


if __name__ == '__main__':
    # Run tests with appropriate verbosity
    unittest.main(verbosity=2)