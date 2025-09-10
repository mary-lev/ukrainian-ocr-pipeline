"""
Unit tests for segmentation module
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.core.segmentation import KrakenSegmenter
from tests.fixtures.create_test_data import create_test_image, create_sample_lines_data


class TestKrakenSegmenter(unittest.TestCase):
    """Test cases for KrakenSegmenter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.segmenter = KrakenSegmenter()
        self.test_image = create_test_image()
        self.expected_lines = create_sample_lines_data()
    
    def test_segmenter_initialization(self):
        """Test segmenter initialization"""
        segmenter = KrakenSegmenter(
            model_path="test_model.mlmodel",
            device="cpu"
        )
        
        self.assertEqual(segmenter.model_path, "test_model.mlmodel")
        self.assertEqual(segmenter.device, "cpu")
        self.assertIsNone(segmenter.model)  # Model not loaded yet
    
    def test_segment_image_returns_lines(self):
        """Test that segment_image returns a list of line dictionaries"""
        lines = self.segmenter.segment_image(self.test_image)
        
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0)  # Should return some lines
        
        # Check line structure
        for line in lines:
            self.assertIn('id', line)
            self.assertIn('bbox', line)
            self.assertIn('polygon', line)
            self.assertIn('baseline', line)
    
    def test_segment_image_bbox_format(self):
        """Test that bounding boxes have correct format"""
        lines = self.segmenter.segment_image(self.test_image)
        
        for line in lines:
            bbox = line['bbox']
            self.assertEqual(len(bbox), 4)  # [x1, y1, x2, y2]
            self.assertLessEqual(bbox[0], bbox[2])  # x1 <= x2
            self.assertLessEqual(bbox[1], bbox[3])  # y1 <= y2
    
    def test_segment_empty_image(self):
        """Test segmentation with empty/blank image"""
        empty_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        lines = self.segmenter.segment_image(empty_image)
        
        # Should still return some result (placeholder implementation)
        self.assertIsInstance(lines, list)
    
    def test_segment_invalid_image(self):
        """Test segmentation with invalid image data"""
        # Test with None
        lines = self.segmenter.segment_image(None)
        self.assertEqual(lines, [])
        
        # Test with wrong shape
        invalid_image = np.ones((10,), dtype=np.uint8)
        lines = self.segmenter.segment_image(invalid_image)
        self.assertIsInstance(lines, list)
    
    def test_save_visualization(self):
        """Test visualization saving"""
        import tempfile
        import os
        
        lines = self.segmenter.segment_image(self.test_image)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            # Should not raise exception
            self.segmenter.save_visualization(self.test_image, lines, output_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
            
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == '__main__':
    unittest.main()