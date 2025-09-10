"""
Unit tests for I/O utilities
"""

import unittest
import tempfile
import os
import json
import numpy as np
import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.utils.io import IOUtils
from tests.fixtures.create_test_data import create_test_image


class TestIOUtils(unittest.TestCase):
    """Test cases for IOUtils"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = create_test_image()
        
        # Create temporary test image file
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(self.temp_image.name, self.test_image)
        self.temp_image.close()
        
        self.test_data = {
            'test': 'data',
            'numbers': [1, 2, 3],
            'nested': {'key': 'value'},
            'unicode': 'Тестові дані'
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_image.name):
            os.unlink(self.temp_image.name)
    
    def test_load_image_success(self):
        """Test successful image loading"""
        loaded_image = IOUtils.load_image(self.temp_image.name)
        
        self.assertIsInstance(loaded_image, np.ndarray)
        self.assertEqual(len(loaded_image.shape), 3)  # Color image
        self.assertEqual(loaded_image.shape, self.test_image.shape)
    
    def test_load_image_nonexistent(self):
        """Test loading non-existent image"""
        result = IOUtils.load_image('non_existent_image.png')
        
        self.assertIsNone(result)
    
    def test_load_image_invalid_format(self):
        """Test loading invalid image format"""
        # Create a text file with image extension
        with tempfile.NamedTemporaryFile(suffix='.png', mode='w', delete=False) as f:
            f.write('This is not an image')
            invalid_image_path = f.name
        
        try:
            result = IOUtils.load_image(invalid_image_path)
            self.assertIsNone(result)
        finally:
            os.unlink(invalid_image_path)
    
    def test_save_image_success(self):
        """Test successful image saving"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            output_path = f.name
        
        try:
            success = IOUtils.save_image(self.test_image, output_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(output_path))
            
            # Verify saved image can be loaded
            loaded = IOUtils.load_image(output_path)
            self.assertIsNotNone(loaded)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_save_image_create_directory(self):
        """Test image saving with directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'subdir', 'test_image.png')
            
            success = IOUtils.save_image(self.test_image, output_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(output_path))
    
    def test_find_images_in_directory(self):
        """Test finding images in directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            image_files = []
            for i, ext in enumerate(['.jpg', '.png', '.tiff']):
                img_path = os.path.join(temp_dir, f'test_{i}{ext}')
                cv2.imwrite(img_path, self.test_image)
                image_files.append(img_path)
            
            # Create non-image file
            text_file = os.path.join(temp_dir, 'readme.txt')
            with open(text_file, 'w') as f:
                f.write('Not an image')
            
            # Find images
            found_images = IOUtils.find_images(temp_dir, recursive=False)
            
            self.assertEqual(len(found_images), 3)
            for img_path in image_files:
                self.assertIn(str(Path(img_path).absolute()), found_images)
    
    def test_find_images_recursive(self):
        """Test recursive image finding"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectory with image
            sub_dir = os.path.join(temp_dir, 'subdir')
            os.makedirs(sub_dir)
            
            img1_path = os.path.join(temp_dir, 'image1.png')
            img2_path = os.path.join(sub_dir, 'image2.jpg')
            
            cv2.imwrite(img1_path, self.test_image)
            cv2.imwrite(img2_path, self.test_image)
            
            # Test recursive
            found_recursive = IOUtils.find_images(temp_dir, recursive=True)
            self.assertEqual(len(found_recursive), 2)
            
            # Test non-recursive
            found_non_recursive = IOUtils.find_images(temp_dir, recursive=False)
            self.assertEqual(len(found_non_recursive), 1)
    
    def test_find_images_nonexistent_directory(self):
        """Test finding images in non-existent directory"""
        result = IOUtils.find_images('non_existent_directory')
        
        self.assertEqual(result, [])
    
    def test_save_json_success(self):
        """Test successful JSON saving"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            success = IOUtils.save_json(self.test_data, json_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(json_path))
            
            # Verify JSON content
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data, self.test_data)
            
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    def test_save_json_create_directory(self):
        """Test JSON saving with directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, 'subdir', 'test.json')
            
            success = IOUtils.save_json(self.test_data, json_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(json_path))
    
    def test_load_json_success(self):
        """Test successful JSON loading"""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(self.test_data, f, ensure_ascii=False)
            json_path = f.name
        
        try:
            loaded_data = IOUtils.load_json(json_path)
            
            self.assertIsNotNone(loaded_data)
            self.assertEqual(loaded_data, self.test_data)
            
        finally:
            os.unlink(json_path)
    
    def test_load_json_nonexistent(self):
        """Test loading non-existent JSON"""
        result = IOUtils.load_json('non_existent.json')
        
        self.assertIsNone(result)
    
    def test_load_json_invalid_format(self):
        """Test loading invalid JSON"""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            f.write('Invalid JSON content {')
            invalid_json_path = f.name
        
        try:
            result = IOUtils.load_json(invalid_json_path)
            self.assertIsNone(result)
        finally:
            os.unlink(invalid_json_path)
    
    def test_create_output_filename(self):
        """Test output filename creation"""
        input_path = '/path/to/input/document.png'
        output_dir = '/output/directory'
        
        # Basic filename
        output_path = IOUtils.create_output_filename(input_path, output_dir)
        expected = os.path.join(output_dir, 'document.png')
        self.assertEqual(output_path, expected)
        
        # With suffix
        output_path_suffix = IOUtils.create_output_filename(
            input_path, output_dir, suffix='processed'
        )
        expected_suffix = os.path.join(output_dir, 'document_processed.png')
        self.assertEqual(output_path_suffix, expected_suffix)
        
        # With new extension
        output_path_ext = IOUtils.create_output_filename(
            input_path, output_dir, extension='xml'
        )
        expected_ext = os.path.join(output_dir, 'document.xml')
        self.assertEqual(output_path_ext, expected_ext)
        
        # With both suffix and extension
        output_path_both = IOUtils.create_output_filename(
            input_path, output_dir, suffix='enhanced', extension='xml'
        )
        expected_both = os.path.join(output_dir, 'document_enhanced.xml')
        self.assertEqual(output_path_both, expected_both)
    
    def test_get_file_info_existing_file(self):
        """Test file info for existing file"""
        info = IOUtils.get_file_info(self.temp_image.name)
        
        self.assertTrue(info['exists'])
        self.assertGreater(info['size_bytes'], 0)
        self.assertGreater(info['size_mb'], 0)
        self.assertEqual(info['name'], os.path.basename(self.temp_image.name))
        self.assertEqual(info['suffix'], '.png')
        self.assertIn('stem', info)
        self.assertIn('parent', info)
        self.assertIn('absolute', info)
    
    def test_get_file_info_nonexistent_file(self):
        """Test file info for non-existent file"""
        info = IOUtils.get_file_info('non_existent_file.txt')
        
        self.assertFalse(info['exists'])
    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_file = os.path.join(temp_dir, 'test.txt')
            log_file = os.path.join(temp_dir, 'test.log')
            
            with open(test_file, 'w') as f:
                f.write('test')
            with open(log_file, 'w') as f:
                f.write('log content')
            
            # Cleanup without keeping logs
            IOUtils.cleanup_temp_files(temp_dir, keep_logs=False)
            
            self.assertFalse(os.path.exists(test_file))
            self.assertFalse(os.path.exists(log_file))
    
    def test_cleanup_temp_files_keep_logs(self):
        """Test temporary file cleanup keeping logs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_file = os.path.join(temp_dir, 'test.txt')
            log_file = os.path.join(temp_dir, 'test.log')
            
            with open(test_file, 'w') as f:
                f.write('test')
            with open(log_file, 'w') as f:
                f.write('log content')
            
            # Cleanup keeping logs
            IOUtils.cleanup_temp_files(temp_dir, keep_logs=True)
            
            self.assertFalse(os.path.exists(test_file))
            self.assertTrue(os.path.exists(log_file))


class TestIOUtilsEdgeCases(unittest.TestCase):
    """Test edge cases for IOUtils"""
    
    def test_save_image_invalid_array(self):
        """Test saving invalid image array"""
        # Wrong dimensions
        invalid_array = np.ones((10,))  # 1D array
        
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            result = IOUtils.save_image(invalid_array, f.name)
            # Should handle gracefully
            self.assertIsInstance(result, bool)
    
    def test_unicode_handling(self):
        """Test Unicode handling in file operations"""
        unicode_data = {
            'українська': 'мова',
            'тест': ['один', 'два', 'три'],
            'nested': {'ключ': 'значення'}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            # Save and load Unicode JSON
            success = IOUtils.save_json(unicode_data, json_path)
            self.assertTrue(success)
            
            loaded = IOUtils.load_json(json_path)
            self.assertEqual(loaded, unicode_data)
            
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    def test_large_file_handling(self):
        """Test handling of reasonably large files"""
        # Create larger test image
        large_image = create_test_image(width=2000, height=1500)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            large_image_path = f.name
        
        try:
            # Save and load large image
            success = IOUtils.save_image(large_image, large_image_path)
            self.assertTrue(success)
            
            loaded = IOUtils.load_image(large_image_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, large_image.shape)
            
            # Check file info
            info = IOUtils.get_file_info(large_image_path)
            self.assertTrue(info['exists'])
            self.assertGreater(info['size_mb'], 1)  # Should be > 1MB
            
        finally:
            if os.path.exists(large_image_path):
                os.unlink(large_image_path)


if __name__ == '__main__':
    unittest.main()