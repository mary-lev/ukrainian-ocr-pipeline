"""
Main Ukrainian OCR Pipeline optimized for cloud inference
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import torch
from tqdm.auto import tqdm

from ..config import OCRConfig
from .segmentation import KrakenSegmenter
from .ocr import TrOCRProcessor
from .ner import NERExtractor
from .enhancement import ALTOEnhancer
from ..utils.gpu import check_gpu_availability, optimize_for_device
from ..utils.io import IOUtils

class UkrainianOCRPipeline:
    """
    High-performance Ukrainian OCR Pipeline optimized for cloud inference
    
    Features:
    - GPU acceleration for TrOCR and NER
    - Batch processing for multiple images
    - Memory management for large documents
    - Progress tracking with tqdm
    - Automatic model downloading
    """
    
    def __init__(
        self, 
        config: Optional[Union[str, Dict, OCRConfig]] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the OCR pipeline
        
        Args:
            config: Configuration file path, dict, or OCRConfig object
            device: Device to use ('cuda', 'cpu', 'auto'). Auto-detects if None
            batch_size: Batch size for processing. Auto-determined if None
            verbose: Enable verbose logging and progress bars
        """
        
        # Setup logging
        self.logger = self._setup_logging(verbose)
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Setup device and optimization
        self.device = self._setup_device(device)
        self.batch_size = batch_size or self._determine_batch_size()
        
        # Initialize components (lazy loading)
        self.segmenter = None
        self.ocr_processor = None 
        self.ner_extractor = None
        self.alto_enhancer = None
        
        # Performance tracking
        self.stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'average_time_per_image': 0.0
        }
        
        self.logger.info(f"Ukrainian OCR Pipeline initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        
    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return logger
        
    def _load_config(self, config: Optional[Union[str, Dict, OCRConfig]]) -> OCRConfig:
        """Load and validate configuration"""
        if isinstance(config, OCRConfig):
            return config
        elif isinstance(config, dict):
            return OCRConfig.from_dict(config)
        elif isinstance(config, str):
            return OCRConfig.from_file(config)
        else:
            return OCRConfig()  # Default configuration
            
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup computation device with optimization"""
        if device == 'auto' or device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Optimize for device
        optimize_for_device(device)
        
        # Log GPU info if available
        if device == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return device
        
    def _determine_batch_size(self) -> int:
        """Automatically determine optimal batch size based on device"""
        if self.device == 'cuda' and torch.cuda.is_available():
            # GPU batch sizes based on memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:
                return 8  # High-end GPU
            elif gpu_memory_gb >= 8:
                return 4  # Mid-range GPU  
            else:
                return 2  # Budget GPU
        else:
            return 1  # CPU processing
            
    def _init_components(self):
        """Initialize pipeline components with lazy loading"""
        if not self.segmenter:
            self.logger.info("Loading Kraken segmentation model...")
            self.segmenter = KrakenSegmenter(
                model_path=self.config.segmentation.model_path,
                device=self.device
            )
            
        if not self.ocr_processor:
            self.logger.info("Loading TrOCR model...")
            self.ocr_processor = TrOCRProcessor(
                model_path=self.config.ocr.model_path,
                device=self.device,
                batch_size=self.batch_size
            )
            
        if not self.ner_extractor:
            self.logger.info("Loading NER model...")
            self.ner_extractor = NERExtractor(
                backend=self.config.ner.backend,
                model_name=self.config.ner.model_name
            )
            
        if not self.alto_enhancer:
            self.alto_enhancer = ALTOEnhancer()
    
    def process_single_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_intermediate: bool = False
    ) -> Dict:
        """
        Process a single image through the complete pipeline
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            save_intermediate: Save intermediate processing steps
            
        Returns:
            Dictionary with processing results and paths
        """
        start_time = time.time()
        
        # Initialize components
        self._init_components()
        
        # Setup paths
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent / "ocr_output"
        output_dir = Path(output_dir)
        
        # Create output structure  
        paths = {
            'alto_basic': output_dir / f"{image_path.stem}.xml",
            'alto_enhanced': output_dir / f"{image_path.stem}_enhanced.xml",
            'visualization': output_dir / f"{image_path.stem}_segmentation.png",
            'person_regions': output_dir / f"{image_path.stem}_person_regions.png"
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Load image
            self.logger.info(f"Processing: {image_path.name}")
            image = IOUtils.load_image(str(image_path))
            
            # Step 2: Segmentation
            self.logger.info("Segmenting text lines...")
            lines = self.segmenter.segment_image(image)
            
            if save_intermediate:
                self.segmenter.save_visualization(
                    image, lines, paths['visualization']
                )
            
            # Step 3: OCR Recognition
            self.logger.info(f"Recognizing text ({len(lines)} lines)...")
            lines_with_text = self.ocr_processor.process_lines(
                image, lines, batch_size=self.batch_size
            )
            
            # Step 4: Generate ALTO XML
            self.logger.info("Generating ALTO XML...")
            alto_xml = self._create_alto_xml(
                image_path, image, lines_with_text
            )
            
            # Save basic ALTO
            # IOUtils.save_alto_xml(alto_xml, str(paths['alto_basic']))
            # Placeholder for ALTO XML creation and saving
            
            # Step 5: NER Enhancement
            self.logger.info("Extracting named entities...")
            # Extract entities from text lines
            entities_by_line = {}
            for line in lines_with_text:
                line_text = line.get('text', '')
                if line_text.strip():
                    entities = self.ner_extractor.extract_entities(line_text)
                    if entities:
                        entities_by_line[line.get('id', f"line_{len(entities_by_line)}")] = {
                            'text': line_text,
                            'entities': entities
                        }
            
            # Step 5.5: Generate enhanced ALTO with NER entities
            enhanced_alto = None  # Initialize as None by default
            
            if entities_by_line:
                try:
                    # For now, create a placeholder enhanced ALTO structure
                    # In a full implementation, this would create actual enhanced ALTO XML files
                    enhanced_alto = {
                        'basic_alto_path': str(paths['alto_basic']),
                        'enhanced_alto_path': str(paths['alto_enhanced']),
                        'entities_count': sum(len(data['entities']) for data in entities_by_line.values()),
                        'lines_with_entities': len(entities_by_line)
                    }
                    self.logger.info(f"Enhanced ALTO placeholder created with {enhanced_alto['entities_count']} entities")
                except Exception as e:
                    self.logger.warning(f"Could not create enhanced ALTO: {e}")
                    enhanced_alto = None
            
            # Step 6: Extract person-dense regions
            person_region_path = None
            if self.config.post_processing.extract_person_regions:
                self.logger.info("Extracting person-dense regions...")
                person_region_path = self._extract_person_regions(
                    enhanced_alto, image_path, paths['person_regions']
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            # Prepare results
            results = {
                'success': True,
                'image_path': str(image_path),
                'processing_time': processing_time,
                'lines_detected': len(lines),
                'lines_with_text': len([l for l in lines_with_text if l.get('text')]),
                'entities_extracted': len(entities_by_line) if entities_by_line else 0,
                'total_entities': sum(len(data['entities']) for data in entities_by_line.values()) if entities_by_line else 0,
                'output_paths': {
                    'alto_basic': str(paths['alto_basic']) if 'alto_basic' in paths else None,
                    'alto_enhanced': str(paths['alto_enhanced']) if enhanced_alto else None,
                    'visualization': str(paths['visualization']) if save_intermediate and 'visualization' in paths else None
                }
            }
            
            if self.config.post_processing.extract_person_regions and person_region_path:
                results['output_paths']['person_regions'] = str(person_region_path)
                
            self.logger.info(f"Processing complete: {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return {
                'success': False,
                'image_path': str(image_path),
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        save_intermediate: bool = False,
        max_workers: Optional[int] = None
    ) -> List[Dict]:
        """
        Process multiple images in batch with optimized GPU utilization
        
        Args:
            image_paths: List of image paths to process
            output_dir: Output directory for all results
            save_intermediate: Save intermediate steps for each image
            max_workers: Maximum parallel workers (None for auto)
            
        Returns:
            List of processing results for each image
        """
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        results = []
        failed_count = 0
        
        # Process with progress bar
        with tqdm(image_paths, desc="Processing images") as pbar:
            for image_path in pbar:
                result = self.process_single_image(
                    image_path, output_dir, save_intermediate
                )
                results.append(result)
                
                if not result['success']:
                    failed_count += 1
                    
                # Update progress bar description
                pbar.set_postfix({
                    'Success': len(results) - failed_count,
                    'Failed': failed_count,
                    'Avg Time': f"{self.stats['average_time_per_image']:.1f}s"
                })
        
        # Log summary
        success_count = len(results) - failed_count
        self.logger.info(f"Batch processing complete:")
        self.logger.info(f"  Successful: {success_count}/{len(image_paths)}")
        self.logger.info(f"  Failed: {failed_count}/{len(image_paths)}")
        self.logger.info(f"  Total time: {self.stats['total_processing_time']:.1f}s")
        self.logger.info(f"  Average per image: {self.stats['average_time_per_image']:.1f}s")
        
        return results
    
    def _create_alto_xml(self, image_path: Path, image, lines: List[Dict]) -> str:
        """Create ALTO XML from processing results"""
        # Implementation similar to existing ALTO creation
        # This would be moved from the existing code
        pass
        
    def _extract_person_regions(self, alto_file: str, image_path: Path, output_dir: Path) -> str:
        """Extract person-dense regions from enhanced ALTO"""
        # Implementation similar to existing person region extraction
        # This would be moved from the existing code
        pass
        
    def _update_stats(self, processing_time: float):
        """Update processing statistics"""
        self.stats['images_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['average_time_per_image'] = (
            self.stats['total_processing_time'] / self.stats['images_processed']
        )
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()
        
    def cleanup(self):
        """Clean up GPU memory and resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Pipeline cleanup complete")