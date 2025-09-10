"""
Input/Output utilities for Ukrainian OCR Pipeline
"""

import logging
import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np
import cv2

class IOUtils:
    """Input/Output utility functions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array or None if failed
        """
        try:
            if not os.path.exists(image_path):
                logging.getLogger(__name__).error(f"Image not found: {image_path}")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                logging.getLogger(__name__).error(f"Failed to load image: {image_path}")
                return None
                
            return image
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """
        Save image to file
        
        Args:
            image: Image array
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, image)
            
            if success:
                logging.getLogger(__name__).info(f"Image saved to: {output_path}")
                return True
            else:
                logging.getLogger(__name__).error(f"Failed to save image: {output_path}")
                return False
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error saving image {output_path}: {e}")
            return False
    
    @staticmethod
    def find_images(directory: str, recursive: bool = True) -> List[str]:
        """
        Find all image files in directory
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_paths = []
        
        directory = Path(directory)
        
        if not directory.exists():
            logging.getLogger(__name__).error(f"Directory not found: {directory}")
            return []
        
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
            
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path.absolute()))
        
        image_paths.sort()
        logging.getLogger(__name__).info(f"Found {len(image_paths)} images in {directory}")
        
        return image_paths
    
    @staticmethod
    def save_json(data: Dict, output_path: str) -> bool:
        """
        Save data as JSON file
        
        Args:
            data: Data to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logging.getLogger(__name__).info(f"JSON saved to: {output_path}")
            return True
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error saving JSON {output_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Dict]:
        """
        Load JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data or None if failed
        """
        try:
            if not os.path.exists(file_path):
                logging.getLogger(__name__).error(f"JSON file not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading JSON {file_path}: {e}")
            return None
    
    @staticmethod
    def save_alto_xml(tree: ET.ElementTree, output_path: str) -> bool:
        """
        Save ALTO XML file
        
        Args:
            tree: XML ElementTree
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            logging.getLogger(__name__).info(f"ALTO XML saved to: {output_path}")
            return True
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error saving ALTO XML {output_path}: {e}")
            return False
    
    @staticmethod
    def load_alto_xml(file_path: str) -> Optional[ET.ElementTree]:
        """
        Load ALTO XML file
        
        Args:
            file_path: Path to ALTO XML file
            
        Returns:
            XML ElementTree or None if failed
        """
        try:
            if not os.path.exists(file_path):
                logging.getLogger(__name__).error(f"ALTO XML file not found: {file_path}")
                return None
            
            tree = ET.parse(file_path)
            return tree
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading ALTO XML {file_path}: {e}")
            return None
    
    @staticmethod
    def create_output_filename(
        input_path: str, 
        output_dir: str, 
        suffix: str = '', 
        extension: str = None
    ) -> str:
        """
        Create output filename based on input filename
        
        Args:
            input_path: Input file path
            output_dir: Output directory
            suffix: Suffix to add to filename
            extension: New file extension (keep original if None)
            
        Returns:
            Output file path
        """
        input_path = Path(input_path)
        
        # Create base filename
        base_name = input_path.stem
        if suffix:
            base_name = f"{base_name}_{suffix}"
        
        # Use provided extension or keep original
        if extension:
            if not extension.startswith('.'):
                extension = f".{extension}"
            output_filename = f"{base_name}{extension}"
        else:
            output_filename = f"{base_name}{input_path.suffix}"
        
        return os.path.join(output_dir, output_filename)
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """
        Get file information
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                return {'exists': False}
            
            stat = path_obj.stat()
            
            return {
                'exists': True,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / 1024 / 1024,
                'name': path_obj.name,
                'stem': path_obj.stem,
                'suffix': path_obj.suffix,
                'parent': str(path_obj.parent),
                'absolute': str(path_obj.absolute())
            }
            
        except Exception as e:
            return {'exists': False, 'error': str(e)}
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str, keep_logs: bool = True):
        """
        Clean up temporary files
        
        Args:
            temp_dir: Temporary directory to clean
            keep_logs: Whether to keep log files
        """
        try:
            temp_path = Path(temp_dir)
            
            if not temp_path.exists():
                return
            
            for file_path in temp_path.rglob('*'):
                if file_path.is_file():
                    # Skip log files if requested
                    if keep_logs and file_path.suffix.lower() in {'.log', '.txt'}:
                        continue
                    
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Could not delete {file_path}: {e}")
            
            # Remove empty directories
            for dir_path in sorted(temp_path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                    except Exception:
                        pass
                        
        except Exception as e:
            logging.getLogger(__name__).error(f"Error cleaning up temp files: {e}")