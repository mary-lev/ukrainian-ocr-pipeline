"""
Unit tests for OCR module
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.core.ocr import TrOCRProcessor
from tests.fixtures.create_test_data import create_test_image, create_sample_lines_data


class TestTrOCRProcessor(unittest.TestCase):
    """Test cases for TrOCRProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = TrOCRProcessor(device="cpu", batch_size=2)
        self.test_image = create_test_image()
        self.sample_lines = create_sample_lines_data()
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = TrOCRProcessor(
            model_path="test-trocr-model",
            device="cpu",
            batch_size=4
        )
        
        self.assertEqual(processor.model_path, "test-trocr-model")
        self.assertEqual(processor.device, "cpu")
        self.assertEqual(processor.batch_size, 4)
        self.assertIsNotNone(processor.model)  # Placeholder model loaded
        self.assertIsNotNone(processor.processor)
    
    def test_process_lines_adds_text(self):
        """Test that process_lines adds text to line dictionaries"""
        lines = self.sample_lines.copy()
        result_lines = self.processor.process_lines(self.test_image, lines)
        
        self.assertEqual(len(result_lines), len(lines))
        
        for line in result_lines:
            self.assertIn('text', line)
            self.assertIn('confidence', line)
            self.assertIsInstance(line['text'], str)
            self.assertIsInstance(line['confidence'], (int, float))
            self.assertGreaterEqual(line['confidence'], 0.0)
            self.assertLessEqual(line['confidence'], 1.0)
    
    def test_process_lines_with_batch_size(self):
        """Test processing with different batch sizes"""
        lines = self.sample_lines * 3  # 9 lines total
        
        # Test with batch_size=2
        result_lines = self.processor.process_lines(
            self.test_image, lines, batch_size=2
        )
        
        self.assertEqual(len(result_lines), len(lines))
        
        # All lines should have text
        for line in result_lines:
            self.assertIn('text', line)
    
    def test_extract_line_region(self):
        """Test line region extraction"""
        line = self.sample_lines[0]
        
        line_image = self.processor._extract_line_region(self.test_image, line)
        
        if line_image is not None:
            self.assertIsInstance(line_image, np.ndarray)
            self.assertEqual(len(line_image.shape), 3)  # Should be color image
            self.assertGreater(line_image.size, 0)
    
    def test_extract_line_region_invalid_bbox(self):
        """Test line region extraction with invalid bbox"""
        # Invalid bbox
        invalid_line = {'bbox': [100, 100, 50, 50]}  # x2 < x1
        result = self.processor._extract_line_region(self.test_image, invalid_line)
        self.assertIsNone(result)
        
        # Missing bbox
        no_bbox_line = {'id': 'test'}
        result = self.processor._extract_line_region(self.test_image, no_bbox_line)
        self.assertIsNone(result)
    
    def test_recognize_text_single_image(self):
        """Test single image text recognition"""
        # Create a small test image
        test_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        
        result = self.processor.recognize_text(test_img)
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['text'], str)
        self.assertIsInstance(result['confidence'], (int, float))
    
    def test_recognize_text_with_path(self):
        """Test text recognition with image path"""
        import tempfile
        import cv2
        
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            cv2.imwrite(temp_path, self.test_image)
            result = self.processor.recognize_text(temp_path)
            
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
            self.assertIn('confidence', result)
            
        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_recognize_text_invalid_input(self):
        """Test text recognition with invalid input"""
        # Test with None
        result = self.processor.recognize_text(None)
        self.assertEqual(result['text'], '')
        self.assertEqual(result['confidence'], 0.0)
        
        # Test with non-existent path
        result = self.processor.recognize_text('non_existent_file.png')
        self.assertEqual(result['text'], '')
        self.assertEqual(result['confidence'], 0.0)
    
    def test_process_empty_lines_list(self):
        """Test processing with empty lines list"""
        result = self.processor.process_lines(self.test_image, [])
        self.assertEqual(result, [])
    
    def test_placeholder_text_generation(self):
        """Test that placeholder generates Ukrainian text"""
        text = self.processor._recognize_text(np.ones((50, 100, 3)))
        
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        
        # Should contain Cyrillic characters (basic check)
        has_cyrillic = any(ord(char) > 127 for char in text)
        self.assertTrue(has_cyrillic, "Generated text should contain Cyrillic characters")


if __name__ == '__main__':
    unittest.main()