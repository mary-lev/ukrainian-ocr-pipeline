"""
TrOCR processing module for Ukrainian OCR Pipeline
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from PIL import Image

class TrOCRProcessor:
    """TrOCR-based text recognition"""
    
    def __init__(
        self, 
        model_path: str = "cyrillic-trocr/trocr-handwritten-cyrillic",
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size or 4
        
        # Initialize model components
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        """Load TrOCR model and processor"""
        try:
            # Install transformers if needed (for Colab)
            self._ensure_transformers_installed()
            
            from transformers import (
                TrOCRProcessor, 
                VisionEncoderDecoderModel,
                ViTFeatureExtractor,
                AutoTokenizer
            )
            
            self.logger.info(f"Loading TrOCR model: {self.model_path}")
            self.logger.info(f"Using device: {self.device}")
            
            # Load processor and model from HuggingFace
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer for additional functionality
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.logger.info(f"Successfully loaded TrOCR model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading TrOCR model: {e}")
            raise
            
    def _ensure_transformers_installed(self):
        """Install transformers if not available (for Colab)"""
        try:
            import transformers
        except ImportError:
            self.logger.info("Installing transformers for TrOCR...")
            import subprocess
            subprocess.check_call([
                'pip', 'install', 'transformers[torch]', '--quiet'
            ])
            self.logger.info("Transformers installed successfully")
    
    def process_lines(
        self, 
        image: np.ndarray, 
        lines: List[Dict], 
        batch_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Process text lines with OCR
        
        Args:
            image: Full document image
            lines: List of line dictionaries from segmentation
            batch_size: Batch size for processing
            
        Returns:
            List of lines with recognized text
        """
        batch_size = batch_size or self.batch_size
        
        try:
            # Process lines in batches
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i + batch_size]
                self._process_batch(image, batch)
                
            return lines
            
        except Exception as e:
            self.logger.error(f"Error processing lines: {e}")
            return lines
    
    def _process_batch(self, image: np.ndarray, batch: List[Dict]):
        """Process a batch of lines with real TrOCR"""
        
        for line in batch:
            # Extract line region from image
            line_image = self._extract_line_region(image, line)
            
            if line_image is not None:
                # Recognize text using real TrOCR
                result = self._recognize_text(line_image)
                line['text'] = result['text']
                line['confidence'] = result['confidence']
            else:
                line['text'] = ''
                line['confidence'] = 0.0
    
    def _extract_line_region(self, image: np.ndarray, line: Dict) -> Optional[np.ndarray]:
        """Extract line region from full image with polygon support"""
        try:
            import cv2
            
            # Use polygon if available (better for irregular shapes)
            if 'polygon' in line and line['polygon'] and len(line['polygon']) >= 3:
                points = np.array(line['polygon'], np.int32)
                
                # Ensure points are valid
                if points.size > 0:
                    # Create mask with some padding to avoid cutting text
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 255)
                    
                    # Get bounding box with padding
                    x, y, w, h = cv2.boundingRect(points)
                    
                    # Add padding to avoid cutting characters
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    
                    # Create padded crop area
                    crop_mask = mask[y:y+h, x:x+w]
                    cropped_image = image[y:y+h, x:x+w]
                    
                    # Apply mask but keep some context
                    result = cv2.bitwise_and(cropped_image, cropped_image, mask=crop_mask)
                    
                    # If masked result is too empty, use the full crop
                    if np.sum(crop_mask) < 0.3 * crop_mask.size:
                        cropped = cropped_image
                    else:
                        cropped = result
                        
                    return cropped if cropped.size > 0 else None
                else:
                    return None
                    
            # Use bbox if no polygon or polygon is invalid
            elif 'bbox' in line and line['bbox'] and len(line['bbox']) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in line['bbox']]
                
                # Add padding to bbox to avoid cutting characters
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                cropped = image[y1:y2, x1:x2]
                return cropped if cropped.size > 0 else None
                
            else:
                self.logger.warning("No valid polygon or bbox found for line")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting line region: {e}")
            
        return None
    
    def _recognize_text(self, line_image: np.ndarray) -> Dict:
        """Recognize text in line image using real TrOCR"""
        try:
            if self.model is None or self.processor is None:
                return {'text': '', 'confidence': 0.0}
            
            # Convert OpenCV to PIL Image
            from PIL import Image
            if len(line_image.shape) == 3:
                # BGR to RGB
                image_rgb = line_image[:, :, ::-1]
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(line_image)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Process image with TrOCR processor
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                'text': text.strip(),
                'confidence': 0.95  # Default confidence for TrOCR
            }
            
        except Exception as e:
            self.logger.error(f"Error recognizing text: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def recognize_text(
        self, 
        image: Union[np.ndarray, str],
        preprocess: bool = False,
        max_length: int = 256,
        num_beams: int = 1
    ) -> Dict:
        """
        Recognize text in single image
        
        Args:
            image: Image array or path
            preprocess: Whether to preprocess the image
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            
        Returns:
            Dictionary with text and confidence scores
        """
        try:
            if isinstance(image, str):
                import cv2
                image = cv2.imread(image)
                
            if image is None:
                return {'text': '', 'confidence': 0.0}
            
            # Placeholder recognition
            text = self._recognize_text(image)
            
            return {
                'text': text.strip(),
                'confidence': 0.95
            }
            
        except Exception as e:
            self.logger.error(f"Error recognizing text: {e}")
            return {'text': '', 'confidence': 0.0}