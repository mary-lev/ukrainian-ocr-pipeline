"""
Visualization utilities for Ukrainian OCR Pipeline
"""

import logging
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class Visualizer:
    """Creates visualizations for OCR pipeline results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def visualize_segmentation(
        self,
        image: np.ndarray,
        lines: List[Dict],
        output_path: Optional[str] = None,
        show_baselines: bool = True,
        show_polygons: bool = True,
        show_bboxes: bool = True
    ) -> np.ndarray:
        """
        Visualize text line segmentation results
        
        Args:
            image: Input image
            lines: List of line dictionaries with coordinates
            output_path: Path to save visualization
            show_baselines: Whether to draw baselines
            show_polygons: Whether to draw polygons
            show_bboxes: Whether to draw bounding boxes
            
        Returns:
            Visualization image
        """
        
        vis_image = image.copy()
        
        for i, line in enumerate(lines):
            line_id = line.get('id', f'line_{i}')
            
            # Draw bounding box
            if show_bboxes:
                bbox = line.get('bbox')
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add line ID label
                    cv2.putText(vis_image, line_id, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw polygon
            if show_polygons:
                polygon = line.get('polygon')
                if polygon:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.polylines(vis_image, [pts], True, (255, 0, 0), 2)
            
            # Draw baseline
            if show_baselines:
                baseline = line.get('baseline')
                if baseline and len(baseline) >= 2:
                    pt1 = tuple(map(int, baseline[0]))
                    pt2 = tuple(map(int, baseline[-1]))
                    cv2.line(vis_image, pt1, pt2, (0, 0, 255), 3)
        
        # Add legend
        if show_bboxes or show_polygons or show_baselines:
            self._add_legend(vis_image, show_bboxes, show_polygons, show_baselines)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"Segmentation visualization saved: {output_path}")
        
        return vis_image
    
    def visualize_ocr_results(
        self,
        image: np.ndarray,
        lines: List[Dict],
        output_path: Optional[str] = None,
        show_text: bool = True,
        show_confidence: bool = True,
        min_confidence: float = 0.5
    ) -> np.ndarray:
        """
        Visualize OCR recognition results
        
        Args:
            image: Input image
            lines: List of lines with OCR results
            output_path: Path to save visualization
            show_text: Whether to overlay recognized text
            show_confidence: Whether to show confidence scores
            min_confidence: Minimum confidence to display
            
        Returns:
            Visualization image
        """
        
        vis_image = image.copy()
        font_scale = 0.6
        font_thickness = 1
        
        for line in lines:
            text = line.get('text', '')
            confidence = line.get('confidence', 0.0)
            
            if confidence < min_confidence:
                continue
            
            # Get line coordinates
            bbox = line.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 165, 255)  # Orange for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Overlay text
            if show_text and text:
                # Create background for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_bg_end = (x1 + text_size[0] + 5, y1 - text_size[1] - 5)
                
                cv2.rectangle(vis_image, (x1, y1), text_bg_end, (255, 255, 255), -1)
                cv2.putText(vis_image, text, (x1 + 2, y1 - 2), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
            
            # Show confidence score
            if show_confidence:
                conf_text = f"{confidence:.2f}"
                cv2.putText(vis_image, conf_text, (x2 - 50, y2 + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add legend
        self._add_ocr_legend(vis_image)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"OCR visualization saved: {output_path}")
        
        return vis_image
    
    def visualize_ner_results(
        self,
        image: np.ndarray,
        lines: List[Dict],
        entities_by_line: Dict,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize NER (Named Entity Recognition) results
        
        Args:
            image: Input image
            lines: List of line dictionaries
            entities_by_line: Dictionary mapping line IDs to entities
            output_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        
        vis_image = image.copy()
        
        # Define colors for different entity types
        entity_colors = {
            'PERSON': (0, 255, 0),      # Green
            'LOCATION': (255, 0, 0),    # Blue
            'ORGANIZATION': (0, 165, 255), # Orange
            'DATE': (255, 255, 0),      # Cyan
            'MISC': (128, 0, 128)       # Purple
        }
        
        for line in lines:
            line_id = line.get('id')
            if not line_id or line_id not in entities_by_line:
                continue
            
            bbox = line.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
                
            entities = entities_by_line[line_id].get('entities', [])
            if not entities:
                continue
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
            
            # Collect entity types for this line
            entity_types = list(set(entity['label'] for entity in entities))
            
            # Draw colored border based on entity types
            for i, entity_type in enumerate(entity_types):
                color = entity_colors.get(entity_type, (128, 128, 128))
                thickness = 3 if entity_type == 'PERSON' else 2
                
                # Offset multiple entity type borders
                offset = i * 3
                cv2.rectangle(vis_image, 
                            (x1 - offset, y1 - offset), 
                            (x2 + offset, y2 + offset), 
                            color, thickness)
            
            # Add entity type labels
            label_y = y1 - 10
            for entity_type in entity_types:
                color = entity_colors.get(entity_type, (128, 128, 128))
                cv2.putText(vis_image, entity_type, (x1, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                label_y -= 15
        
        # Add NER legend
        self._add_ner_legend(vis_image, entity_colors)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"NER visualization saved: {output_path}")
        
        return vis_image
    
    def visualize_person_regions(
        self,
        image: np.ndarray,
        person_regions: List[Dict],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detected person-dense regions
        
        Args:
            image: Input image
            person_regions: List of person-dense regions
            output_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        
        vis_image = image.copy()
        
        for i, region in enumerate(person_regions):
            # Get region bounds
            y_min = int(region.get('y_min', 0))
            y_max = int(region.get('y_max', image.shape[0]))
            
            # Calculate x bounds from lines in region
            lines = region.get('lines', [])
            x_coords = []
            
            for line in lines:
                element = line.get('element')
                if element is not None:
                    # Try to get coordinates
                    hpos = element.get('HPOS')
                    width = element.get('WIDTH')
                    if hpos and width:
                        x_coords.extend([int(hpos), int(hpos) + int(width)])
            
            if x_coords:
                x_min = max(0, min(x_coords) - 50)
                x_max = min(image.shape[1], max(x_coords) + 50)
            else:
                x_min, x_max = 0, image.shape[1]
            
            # Draw semi-transparent overlay
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
            cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0, vis_image)
            
            # Draw border
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Add label
            label = f"Person Region {i+1} ({len(lines)} lines)"
            cv2.putText(vis_image, label, (x_min + 10, y_min + 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"Person regions visualization saved: {output_path}")
        
        return vis_image
    
    def extract_person_region_crop(
        self,
        image: np.ndarray,
        region: Dict,
        output_path: Optional[str] = None,
        padding: int = 50
    ) -> Optional[np.ndarray]:
        """
        Extract cropped image of person-dense region
        
        Args:
            image: Input image
            region: Person-dense region dictionary
            output_path: Path to save cropped image
            padding: Padding around region
            
        Returns:
            Cropped image or None if failed
        """
        
        try:
            # Get region bounds
            y_min = max(0, int(region.get('y_min', 0)) - padding)
            y_max = min(image.shape[0], int(region.get('y_max', image.shape[0])) + padding)
            
            # Calculate x bounds from lines
            lines = region.get('lines', [])
            x_coords = []
            
            for line in lines:
                element = line.get('element')
                if element is not None:
                    hpos = element.get('HPOS')
                    width = element.get('WIDTH')
                    if hpos and width:
                        x_coords.extend([int(hpos), int(hpos) + int(width)])
            
            if x_coords:
                x_min = max(0, min(x_coords) - padding)
                x_max = min(image.shape[1], max(x_coords) + padding)
            else:
                x_min, x_max = 0, image.shape[1]
            
            # Extract region
            cropped = image[y_min:y_max, x_min:x_max]
            
            if cropped.size == 0:
                self.logger.warning("Empty cropped region")
                return None
            
            # Save if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, cropped)
                self.logger.info(f"Person region crop saved: {output_path}")
            
            return cropped
            
        except Exception as e:
            self.logger.error(f"Error extracting person region crop: {e}")
            return None
    
    def _add_legend(self, image: np.ndarray, show_bboxes: bool, show_polygons: bool, show_baselines: bool):
        """Add legend to segmentation visualization"""
        legend_items = []
        
        if show_bboxes:
            legend_items.append(("Bounding Box", (0, 255, 0)))
        if show_polygons:
            legend_items.append(("Polygon", (255, 0, 0)))
        if show_baselines:
            legend_items.append(("Baseline", (0, 0, 255)))
        
        # Draw legend background
        legend_height = len(legend_items) * 25 + 10
        cv2.rectangle(image, (10, 10), (200, 10 + legend_height), (255, 255, 255), -1)
        cv2.rectangle(image, (10, 10), (200, 10 + legend_height), (0, 0, 0), 2)
        
        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y = 30 + i * 25
            cv2.line(image, (20, y), (40, y), color, 3)
            cv2.putText(image, label, (50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def _add_ocr_legend(self, image: np.ndarray):
        """Add legend to OCR visualization"""
        legend_items = [
            ("High Conf (>0.8)", (0, 255, 0)),
            ("Med Conf (>0.6)", (0, 165, 255)),
            ("Low Conf (<0.6)", (0, 0, 255))
        ]
        
        # Draw legend
        legend_height = len(legend_items) * 25 + 10
        cv2.rectangle(image, (10, 10), (180, 10 + legend_height), (255, 255, 255), -1)
        cv2.rectangle(image, (10, 10), (180, 10 + legend_height), (0, 0, 0), 2)
        
        for i, (label, color) in enumerate(legend_items):
            y = 30 + i * 25
            cv2.rectangle(image, (20, y - 5), (35, y + 5), color, -1)
            cv2.putText(image, label, (45, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def _add_ner_legend(self, image: np.ndarray, entity_colors: Dict[str, Tuple[int, int, int]]):
        """Add legend to NER visualization"""
        legend_items = [(entity_type, color) for entity_type, color in entity_colors.items()]
        
        # Draw legend
        legend_height = len(legend_items) * 25 + 10
        cv2.rectangle(image, (10, 10), (150, 10 + legend_height), (255, 255, 255), -1)
        cv2.rectangle(image, (10, 10), (150, 10 + legend_height), (0, 0, 0), 2)
        
        for i, (entity_type, color) in enumerate(legend_items):
            y = 30 + i * 25
            cv2.rectangle(image, (20, y - 5), (35, y + 5), color, -1)
            cv2.putText(image, entity_type, (45, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)