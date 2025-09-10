"""
Kraken segmentation module for Ukrainian OCR Pipeline
eScriptorium-compatible implementation
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
from PIL import Image

class KrakenSegmenter:
    """Kraken-based text line segmentation with eScriptorium compatibility"""
    
    DEFAULT_PARAMS = {
        'text_direction': 'horizontal-lr',
        'script_detection': False,
        'line_height': 35,
        'pad': 8,
        'scale': None,
        'binarize': False,
        'raise_on_error': False,
        'autocast': False,
        'fallback_line_height': 35,
        'min_line_confidence': 0.3,
        'merge_regions': True,
        'bbox_padding': 8
    }
    
    def __init__(self, model_path: str = "kraken_model/blla.mlmodel", device: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device or 'cpu'
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the Kraken BLLA model"""
        try:
            # Check if Kraken is installed
            try:
                from kraken.lib import vgsl
                kraken_available = True
            except ImportError:
                self.logger.warning("Kraken not installed, will download models automatically")
                kraken_available = False
            
            if kraken_available and self.model_path and os.path.exists(self.model_path):
                # Load the BLLA segmentation model properly
                from kraken.lib import vgsl
                self.model = vgsl.TorchVGSLModel.load_model(self.model_path)
                self.logger.info(f"Loaded BLLA model: {self.model_path}")
                self.logger.info(f"Model type: {self.model.model_type}")
            else:
                self.logger.info("Model file not found, will use Kraken's default model")
                self.model = None
                
        except Exception as e:
            self.logger.error(f"Error loading BLLA model: {e}")
            self.model = None
            
    def segment_image(self, image: np.ndarray) -> List[Dict]:
        """
        Segment image into text lines using real Kraken BLLA
        
        Args:
            image: Input image array (OpenCV format)
            
        Returns:
            List of line dictionaries with coordinates and polygons
        """
        try:
            # Install kraken if needed (for Colab)
            self._ensure_kraken_installed()
            
            from kraken import blla
            from PIL import Image as PILImage
            
            # Convert OpenCV to PIL
            if len(image.shape) == 3:
                # BGR to RGB
                image_rgb = image[:, :, ::-1]
                pil_image = PILImage.fromarray(image_rgb)
            else:
                pil_image = PILImage.fromarray(image)
            
            # Convert to RGB if needed
            if pil_image.mode == 'RGBA':
                background = PILImage.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[3])
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            self.logger.info("Running Kraken BLLA segmentation...")
            
            # Run segmentation with BLLA model with error handling
            seg_result = None
            device_tried = self.device
            
            try:
                self.logger.info(f"Running Kraken segmentation on {device_tried}")
                seg_result = blla.segment(
                    pil_image,
                    text_direction=self.DEFAULT_PARAMS['text_direction'],
                    model=self.model,  # None uses default model
                    device=device_tried,
                    raise_on_error=False,  # Don't raise on topology errors
                    autocast=self.DEFAULT_PARAMS['autocast']
                )
            except Exception as seg_error:
                error_str = str(seg_error)
                self.logger.warning(f"Segmentation error on {device_tried}: {seg_error}")
                
                # Check for cuDNN/CUDA-specific issues
                cuda_issues = any(issue in error_str for issue in [
                    'cuDNN', 'CUDNN', 'CUDA', 'out of memory', 'runtime version'
                ])
                
                if cuda_issues and device_tried != 'cpu':
                    self.logger.warning("CUDA/cuDNN compatibility issue detected, retrying with CPU...")
                    device_tried = 'cpu'
                    
                    # Try again with CPU
                    try:
                        seg_result = blla.segment(
                            pil_image,
                            text_direction='horizontal-lr',  # Force horizontal
                            model=None,  # Use default model
                            device='cpu',  # Force CPU to avoid GPU issues
                            raise_on_error=False,
                            autocast=False
                        )
                        self.logger.info("Segmentation successful on CPU fallback")
                    except Exception as cpu_error:
                        self.logger.error(f"Segmentation failed even on CPU: {cpu_error}")
                        return self._create_emergency_fallback_segmentation(image)
                else:
                    # Non-CUDA error, try conservative settings on same device
                    try:
                        self.logger.info(f"Retrying segmentation with conservative settings on {device_tried}...")
                        seg_result = blla.segment(
                            pil_image,
                            text_direction='horizontal-lr',  # Force horizontal
                            model=None,  # Use default model
                            device=device_tried,
                            raise_on_error=False,
                            autocast=False
                        )
                    except Exception as retry_error:
                        self.logger.error(f"Segmentation failed completely: {retry_error}")
                        return self._create_emergency_fallback_segmentation(image)
            
            # Check if segmentation was successful
            if seg_result is None:
                self.logger.error("Segmentation failed - no result obtained")
                return self._create_emergency_fallback_segmentation(image)
            
            # Convert to format expected by OCR pipeline
            lines = self._convert_kraken_output(seg_result)
            
            # Filter out any problematic lines
            valid_lines = [line for line in lines if line.get('bbox') and line['bbox'] != [0, 0, 0, 0]]
            
            self.logger.info(f"Segmentation complete: {len(valid_lines)} valid text lines detected")
            if len(valid_lines) != len(lines):
                self.logger.warning(f"Filtered out {len(lines) - len(valid_lines)} invalid lines")
            
            return valid_lines
            
        except Exception as e:
            self.logger.error(f"Error in segmentation: {e}")
            self.logger.warning("Falling back to placeholder segmentation")
            return self._placeholder_segmentation(image)
    
    def _ensure_kraken_installed(self):
        """Install Kraken if not available (for Colab)"""
        try:
            import kraken
        except ImportError:
            self.logger.info("Installing Kraken for segmentation...")
            import subprocess
            subprocess.check_call([
                'pip', 'install', 'kraken[pytorch]', '--quiet'
            ])
            self.logger.info("Kraken installed successfully")
            
    def _convert_kraken_output(self, kraken_output) -> List[Dict]:
        """Convert Kraken output to pipeline format with polygon validation"""
        lines = []
        
        if hasattr(kraken_output, 'lines'):
            for idx, line in enumerate(kraken_output.lines):
                try:
                    # Get baseline (required)
                    baseline = line.baseline if hasattr(line, 'baseline') else []
                    
                    # Get polygon (bounding polygon) with validation
                    raw_polygon = line.boundary if hasattr(line, 'boundary') else []
                    polygon = self._validate_and_fix_polygon(raw_polygon, idx)
                    
                    # Calculate bounding box
                    bbox = self._polygon_to_bbox(polygon) if polygon else self._baseline_to_bbox(baseline)
                    
                    # Skip lines with invalid geometry
                    if not bbox or bbox == [0, 0, 0, 0]:
                        self.logger.warning(f"Skipping line {idx} due to invalid geometry")
                        continue
                    
                    # Format line data
                    line_data = {
                        'id': f'line_{idx}',
                        'baseline': self._format_points(baseline),
                        'polygon': self._format_points(polygon),
                        'bbox': bbox,
                        'text': '',  # Will be filled by OCR
                        'confidence': getattr(line, 'confidence', 1.0)
                    }
                    
                    lines.append(line_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing line {idx}: {e}")
                    # Create a fallback line from baseline if available
                    if hasattr(line, 'baseline') and line.baseline:
                        try:
                            baseline = line.baseline
                            bbox = self._baseline_to_bbox(baseline)
                            
                            line_data = {
                                'id': f'line_{idx}',
                                'baseline': self._format_points(baseline),
                                'polygon': self._create_polygon_from_baseline(baseline),
                                'bbox': bbox,
                                'text': '',
                                'confidence': getattr(line, 'confidence', 0.5)
                            }
                            lines.append(line_data)
                            self.logger.info(f"Created fallback line {idx} from baseline")
                        except Exception as fallback_error:
                            self.logger.warning(f"Failed to create fallback for line {idx}: {fallback_error}")
        
        return lines
    
    def _validate_and_fix_polygon(self, polygon: List, line_idx: int) -> List:
        """Validate and fix polygon geometry to prevent topology errors"""
        if not polygon or len(polygon) < 3:
            return []
        
        try:
            # Convert to numpy array for easier manipulation
            import numpy as np
            points = np.array(polygon)
            
            # Remove duplicate consecutive points
            if len(points) > 1:
                # Calculate distances between consecutive points
                diffs = np.diff(points, axis=0)
                distances = np.sqrt(np.sum(diffs**2, axis=1))
                
                # Keep points that are far enough apart (> 1 pixel)
                keep_mask = distances > 1.0
                if len(keep_mask) > 0:
                    # Always keep the first point
                    keep_indices = [0] + [i+1 for i, keep in enumerate(keep_mask) if keep]
                    points = points[keep_indices]
            
            # Check for minimum number of points
            if len(points) < 3:
                self.logger.debug(f"Line {line_idx}: Too few points after deduplication")
                return []
            
            # Check for self-intersections (basic check)
            try:
                # Simple convex hull to remove self-intersections
                from scipy.spatial import ConvexHull
                if len(points) >= 3:
                    hull = ConvexHull(points)
                    points = points[hull.vertices]
            except:
                # If ConvexHull fails, try to remove obvious self-intersections
                points = self._simple_polygon_fix(points)
            
            # Convert back to list format
            return [[float(p[0]), float(p[1])] for p in points]
            
        except Exception as e:
            self.logger.debug(f"Polygon validation failed for line {line_idx}: {e}")
            # Return empty list to trigger fallback to baseline
            return []
    
    def _simple_polygon_fix(self, points):
        """Simple polygon fixing without external dependencies"""
        try:
            import numpy as np
            
            # Remove points that are too close to each other
            if len(points) < 3:
                return points
                
            # Calculate centroid
            centroid = np.mean(points, axis=0)
            
            # Sort points by angle from centroid to create a simple polygon
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            
            return points[sorted_indices]
        except:
            # Ultimate fallback - just return original points
            return points
    
    def _create_polygon_from_baseline(self, baseline: List) -> List[List[float]]:
        """Create a simple rectangular polygon from baseline"""
        if not baseline or len(baseline) < 2:
            return []
        
        try:
            # Convert baseline to points
            points = self._format_points(baseline)
            if len(points) < 2:
                return []
            
            # Create a rectangle around the baseline
            # Assuming baseline goes from left to right
            left_x = min(p[0] for p in points)
            right_x = max(p[0] for p in points)
            top_y = min(p[1] for p in points) - 15  # Above baseline
            bottom_y = max(p[1] for p in points) + 5  # Below baseline
            
            # Create rectangle polygon
            return [
                [left_x, top_y],
                [right_x, top_y],
                [right_x, bottom_y],
                [left_x, bottom_y]
            ]
            
        except Exception as e:
            self.logger.debug(f"Failed to create polygon from baseline: {e}")
            return []
    
    def _create_emergency_fallback_segmentation(self, image: np.ndarray) -> List[Dict]:
        """Create emergency fallback segmentation when Kraken completely fails"""
        self.logger.warning("Creating emergency fallback segmentation")
        
        try:
            height, width = image.shape[:2]
            
            # Create horizontal strips across the image
            num_strips = max(10, height // 200)  # At least 10 strips, or one per 200 pixels
            strip_height = height // num_strips
            
            lines = []
            for i in range(num_strips):
                y_start = i * strip_height
                y_end = min((i + 1) * strip_height, height)
                
                # Skip very small strips
                if y_end - y_start < 20:
                    continue
                
                # Create a reasonable text line region
                x_margin = width // 20  # 5% margin on each side
                x_start = x_margin
                x_end = width - x_margin
                
                # Ensure reasonable dimensions
                if x_end - x_start < 50:  # Minimum width
                    x_start = 0
                    x_end = width
                
                line_data = {
                    'id': f'fallback_line_{i}',
                    'bbox': [x_start, y_start, x_end, y_end],
                    'polygon': [
                        [x_start, y_start],
                        [x_end, y_start],
                        [x_end, y_end],
                        [x_start, y_end]
                    ],
                    'baseline': [
                        [x_start, y_end - 5],
                        [x_end, y_end - 5]
                    ],
                    'text': '',
                    'confidence': 0.3  # Low confidence for fallback
                }
                
                lines.append(line_data)
            
            self.logger.info(f"Created {len(lines)} fallback segmentation regions")
            return lines
            
        except Exception as e:
            self.logger.error(f"Even fallback segmentation failed: {e}")
            # Return minimal single-region segmentation
            return [{
                'id': 'emergency_line_0',
                'bbox': [0, 0, image.shape[1] if len(image.shape) > 1 else 1000, 
                        image.shape[0] if len(image.shape) > 0 else 1000],
                'polygon': [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                'baseline': [[0, 500], [1000, 500]],
                'text': '',
                'confidence': 0.1
            }]
    
    def _format_points(self, points: List) -> List[List[float]]:
        """Format points to [[x1,y1], [x2,y2], ...] format"""
        if not points:
            return []
        try:
            formatted = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    formatted.append([float(point[0]), float(point[1])])
            return formatted
        except:
            return []
    
    def _polygon_to_bbox(self, polygon: List) -> List[float]:
        """Convert polygon to bounding box [x1, y1, x2, y2]"""
        if not polygon:
            return [0, 0, 0, 0]
        try:
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            
            padding = self.DEFAULT_PARAMS.get('bbox_padding', 8)
            
            return [
                float(min(x_coords)) - padding,
                float(min(y_coords)) - padding,
                float(max(x_coords)) + padding,
                float(max(y_coords)) + padding
            ]
        except:
            return [0, 0, 0, 0]
    
    def _baseline_to_bbox(self, baseline: List, height: int = 35) -> List[float]:
        """Convert baseline to bbox with estimated height"""
        if not baseline:
            return [0, 0, 0, 0]
        try:
            x_coords = [p[0] for p in baseline]
            y_coords = [p[1] for p in baseline]
            
            padding = self.DEFAULT_PARAMS.get('bbox_padding', 8)
            
            return [
                float(min(x_coords)) - padding,
                float(min(y_coords)) - height,
                float(max(x_coords)) + padding,
                float(max(y_coords)) + padding
            ]
        except:
            return [0, 0, 0, 0]
    
    def _placeholder_segmentation(self, image: np.ndarray) -> List[Dict]:
        """Fallback placeholder segmentation for testing"""
        self.logger.warning("Using placeholder segmentation - install Kraken for real results")
        
        height, width = image.shape[:2]
        lines = []
        num_lines = 10  # Simulate 10 text lines
        
        for i in range(num_lines):
            y = int((i + 1) * height / (num_lines + 1))
            line = {
                'id': f'line_{i}',
                'bbox': [50, y - 20, width - 50, y + 20],
                'polygon': [
                    [50, y - 20], [width - 50, y - 20], 
                    [width - 50, y + 20], [50, y + 20]
                ],
                'baseline': [[50, y], [width - 50, y]],
                'text': '',
                'confidence': 1.0
            }
            lines.append(line)
            
        return lines
    
    def save_visualization(self, image: np.ndarray, lines: List[Dict], output_path: str):
        """Save segmentation visualization"""
        import cv2
        
        vis_image = image.copy()
        
        for line in lines:
            # Prioritize polygon over bounding box for more accurate visualization
            polygon = line.get('polygon', [])
            if polygon and len(polygon) >= 3:
                # Draw polygon boundary (blue)
                pts = np.array(polygon, np.int32)
                cv2.polylines(vis_image, [pts], True, (255, 0, 0), 2)
            else:
                # Fallback to bounding box if no polygon available (green)
                bbox = line.get('bbox', [])
                if len(bbox) >= 4:
                    cv2.rectangle(vis_image, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 255, 0), 2)
            
            # Optional: Draw baseline if available (red, thinner)
            baseline = line.get('baseline', [])
            if baseline and len(baseline) >= 2:
                pts = np.array(baseline, np.int32)
                cv2.polylines(vis_image, [pts], False, (0, 0, 255), 1)
        
        cv2.imwrite(str(output_path), vis_image)
        self.logger.info(f"Saved segmentation visualization to: {output_path}")