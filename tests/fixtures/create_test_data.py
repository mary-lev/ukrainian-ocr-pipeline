"""
Create test fixtures for Ukrainian OCR Pipeline tests
"""

import numpy as np
import cv2
import os
from pathlib import Path

def create_test_image(width: int = 800, height: int = 600, text_lines: int = 5) -> np.ndarray:
    """Create a simple test image with text-like rectangular regions"""
    
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some text-like rectangular regions (dark on white)
    line_height = 30
    line_spacing = height // (text_lines + 1)
    
    for i in range(text_lines):
        y_start = line_spacing * (i + 1) - line_height // 2
        y_end = y_start + line_height
        
        # Vary line widths
        line_width = width - (50 + (i % 3) * 100)
        x_start = 50
        x_end = x_start + line_width
        
        # Draw dark rectangle to simulate text
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        
        # Add some "character" variations
        for j in range(x_start, x_end, 20):
            if np.random.random() > 0.3:  # 70% chance of "character"
                cv2.rectangle(image, 
                            (j, y_start + 5), 
                            (j + 15, y_end - 5), 
                            (255, 255, 255), -1)
    
    return image

def create_sample_lines_data() -> list:
    """Create sample line data for testing segmentation"""
    return [
        {
            'id': 'line_0',
            'bbox': [50, 80, 600, 110],
            'polygon': [[50, 80], [600, 80], [600, 110], [50, 110]],
            'baseline': [[50, 95], [600, 95]]
        },
        {
            'id': 'line_1', 
            'bbox': [50, 140, 550, 170],
            'polygon': [[50, 140], [550, 140], [550, 170], [50, 170]],
            'baseline': [[50, 155], [550, 155]]
        },
        {
            'id': 'line_2',
            'bbox': [50, 200, 580, 230], 
            'polygon': [[50, 200], [580, 200], [580, 230], [50, 230]],
            'baseline': [[50, 215], [580, 215]]
        }
    ]

def create_sample_ocr_results() -> list:
    """Create sample OCR results for testing"""
    lines = create_sample_lines_data()
    
    sample_texts = [
        "Андрей Моисеевич Орехов",
        "село Песчаное, Харківська губернія", 
        "1920 року народження"
    ]
    
    for i, line in enumerate(lines):
        line['text'] = sample_texts[i] if i < len(sample_texts) else f"Sample text {i}"
        line['confidence'] = 0.85 + (i * 0.05)  # Varying confidence scores
    
    return lines

def create_sample_entities() -> dict:
    """Create sample NER entities for testing"""
    return {
        'line_0': {
            'text': 'Андрей Моисеевич Орехов',
            'entities': [
                {
                    'text': 'Андрей Моисеевич Орехов',
                    'label': 'PERSON',
                    'start': 0,
                    'end': 22,
                    'confidence': 0.9
                }
            ]
        },
        'line_1': {
            'text': 'село Песчаное, Харківська губернія',
            'entities': [
                {
                    'text': 'Песчаное',
                    'label': 'LOCATION', 
                    'start': 5,
                    'end': 13,
                    'confidence': 0.8
                },
                {
                    'text': 'Харківська губернія',
                    'label': 'LOCATION',
                    'start': 15,
                    'end': 34,
                    'confidence': 0.85
                }
            ]
        }
    }

def setup_test_fixtures():
    """Create all test fixtures in the fixtures directory"""
    
    fixtures_dir = Path(__file__).parent
    
    # Create test image
    test_image = create_test_image()
    cv2.imwrite(str(fixtures_dir / 'test_document.png'), test_image)
    
    # Create small test image  
    small_image = create_test_image(width=400, height=300, text_lines=3)
    cv2.imwrite(str(fixtures_dir / 'small_document.png'), small_image)
    
    # Create empty image
    empty_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(fixtures_dir / 'empty_document.png'), empty_image)
    
    print("Test fixtures created successfully!")

if __name__ == '__main__':
    setup_test_fixtures()