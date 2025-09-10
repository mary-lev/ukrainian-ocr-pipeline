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
from .config import OCRPipelineConfig
from .segmentation import KrakenSegmenter
from .ocr import TrOCRProcessor
from .ner import NERExtractor
from .surname_matcher import SurnameMatcher
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
        config: Optional[Union[str, Dict, OCRConfig, OCRPipelineConfig]] = None,
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
        self.surname_matcher = None
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
        
    def _load_config(self, config: Optional[Union[str, Dict, OCRConfig, OCRPipelineConfig]]) -> OCRPipelineConfig:
        """Load and validate configuration"""
        if isinstance(config, OCRPipelineConfig):
            return config
        elif isinstance(config, OCRConfig):
            # Convert old OCRConfig to new OCRPipelineConfig for backward compatibility
            return OCRPipelineConfig()
        elif isinstance(config, dict):
            return OCRPipelineConfig.from_dict(config)
        elif isinstance(config, str):
            return OCRPipelineConfig.from_file(config)
        else:
            return OCRPipelineConfig()  # Default configuration
            
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
            # Prepare backend config for NER
            backend_config = self.config.ner.backend_config or {}
            if self.config.ner.model_name:
                backend_config['model'] = self.config.ner.model_name
            if self.config.ner.api_key:
                backend_config['api_key'] = self.config.ner.api_key
                
            self.ner_extractor = NERExtractor(
                backend=self.config.ner.backend,
                backend_config=backend_config
            )
            
        if not self.surname_matcher and self.config.surname_matching.enabled:
            self.logger.info("Loading surname matcher...")
            self.surname_matcher = SurnameMatcher(
                surname_file=self.config.surname_matching.surname_file,
                surnames=self.config.surname_matching.surnames,
                threshold=self.config.surname_matching.threshold,
                use_phonetic=self.config.surname_matching.use_phonetic,
                min_length=self.config.surname_matching.min_length
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
            
            # Save basic ALTO XML
            with open(paths['alto_basic'], 'w', encoding='utf-8') as f:
                f.write(alto_xml)
            self.logger.info(f"Basic ALTO XML saved: {paths['alto_basic']}")
            
            # Step 5: NER Enhancement
            self.logger.info("Extracting named entities...")
            # Use the new NER system to extract entities from all lines
            ner_results = self.ner_extractor.extract_entities_from_lines(lines_with_text)
            
            entities_by_line = {}
            if ner_results.get('entities'):
                # Group entities by the lines they appear in
                for entity in ner_results['entities']:
                    source_line = entity.get('source_line', '')
                    if source_line:
                        # Find the line ID that matches this text
                        for line in lines_with_text:
                            if line.get('text') == source_line:
                                line_id = line.get('id', f"line_{len(entities_by_line)}")
                                if line_id not in entities_by_line:
                                    entities_by_line[line_id] = {
                                        'text': source_line,
                                        'entities': []
                                    }
                                entities_by_line[line_id]['entities'].append(entity)
                                break
            
            # Step 5.1: Surname Matching (if enabled)
            surname_matches = []
            if self.surname_matcher and self.config.surname_matching.enabled:
                self.logger.info("Finding surname matches...")
                surname_results = self.surname_matcher.find_in_lines(lines_with_text)
                surname_matches = surname_results
                
                # Export matches if requested
                if self.config.surname_matching.export_matches and surname_matches:
                    matches_file = paths['base'] / f"{paths['base'].stem}_surname_matches.json"
                    self.surname_matcher.export_matches(surname_matches, str(matches_file))
                    paths['surname_matches'] = matches_file
            
            # Step 5.5: Generate enhanced ALTO with NER entities
            enhanced_alto = None  # Initialize as None by default
            
            if entities_by_line:
                try:
                    # Use ALTOEnhancer for sophisticated NER enhancement
                    enhanced_alto_path = self.alto_enhancer.enhance_alto_with_ner(
                        str(paths['alto_basic']), entities_by_line, str(paths['alto_enhanced'])
                    )
                    
                    entities_count = sum(len(data['entities']) for data in entities_by_line.values())
                    enhanced_alto = {
                        'basic_alto_path': str(paths['alto_basic']),
                        'enhanced_alto_path': str(paths['alto_enhanced']),
                        'entities_count': entities_count,
                        'lines_with_entities': len(entities_by_line)
                    }
                    self.logger.info(f"Enhanced ALTO XML created with {entities_count} entities")
                    
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
                'ner_backend': ner_results.get('backend', 'unknown') if ner_results else 'none',
                'surnames_found': len(surname_matches) if surname_matches else 0,
                'unique_surnames': len(set(m.matched_surname for m in surname_matches)) if surname_matches else 0,
                'output_paths': {
                    'alto_basic': str(paths['alto_basic']) if 'alto_basic' in paths else None,
                    'alto_enhanced': str(paths['alto_enhanced']) if enhanced_alto else None,
                    'visualization': str(paths['visualization']) if save_intermediate and 'visualization' in paths else None,
                    'surname_matches': str(paths['surname_matches']) if 'surname_matches' in paths else None
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
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        # Create ALTO structure
        alto = ET.Element('alto', {
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xmlns': 'http://www.loc.gov/standards/alto/ns-v4#',
            'xsi:schemaLocation': 'http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd'
        })
        
        # Add Description
        description = ET.SubElement(alto, 'Description')
        measurement_unit = ET.SubElement(description, 'MeasurementUnit')
        measurement_unit.text = 'pixel'
        
        source_info = ET.SubElement(description, 'sourceImageInformation')
        filename = ET.SubElement(source_info, 'fileName')
        filename.text = image_path.name
        
        # Add Tags
        tags = ET.SubElement(alto, 'Tags')
        # Add text block type
        block_tag = ET.SubElement(tags, 'OtherTag')
        block_tag.set('ID', 'BT1')
        block_tag.set('LABEL', 'text')
        block_tag.set('DESCRIPTION', 'block type text')
        # Add line type
        line_tag = ET.SubElement(tags, 'OtherTag')
        line_tag.set('ID', 'LT1')
        line_tag.set('LABEL', 'default')
        line_tag.set('DESCRIPTION', 'line type default')
        
        # Add Layout
        layout = ET.SubElement(alto, 'Layout')
        page = ET.SubElement(layout, 'Page')
        page.set('ID', f'page_{image_path.stem}')
        page.set('PHYSICAL_IMG_NR', '1')
        
        # Get image dimensions
        if hasattr(image, 'shape'):
            height, width = image.shape[:2]
        else:
            height, width = 3000, 2000  # Default dimensions
            
        page.set('WIDTH', str(width))
        page.set('HEIGHT', str(height))
        
        # Add PrintSpace
        print_space = ET.SubElement(page, 'PrintSpace')
        print_space.set('HPOS', '0')
        print_space.set('VPOS', '0')
        print_space.set('WIDTH', str(width))
        print_space.set('HEIGHT', str(height))
        
        # Add TextBlock
        text_block = ET.SubElement(print_space, 'TextBlock')
        text_block.set('ID', f'block_{image_path.stem}')
        text_block.set('TAGREFS', 'BT1')
        
        # Add text lines
        for idx, line in enumerate(lines):
            text_content = line.get('text', '').strip()
            if not text_content:
                continue
                
            text_line = ET.SubElement(text_block, 'TextLine')
            text_line.set('ID', f'line_{idx}')
            text_line.set('TAGREFS', 'LT1')
            
            # Get line coordinates
            bbox = line.get('bbox', [0, 0, 100, 30])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
            else:
                x1, y1, x2, y2 = 0, idx * 30, 100, (idx + 1) * 30
                
            text_line.set('HPOS', str(x1))
            text_line.set('VPOS', str(y1))
            text_line.set('WIDTH', str(x2 - x1))
            text_line.set('HEIGHT', str(y2 - y1))
            
            # Add baseline
            baseline_points = line.get('baseline', [])
            if baseline_points and len(baseline_points) >= 2:
                try:
                    baseline_str = ' '.join(f"{int(p[0])} {int(p[1])}" for p in baseline_points)
                    text_line.set('BASELINE', baseline_str)
                except (ValueError, IndexError):
                    # Create default baseline from bbox
                    y_base = y2 - 5  # Slightly above bottom
                    text_line.set('BASELINE', f"{x1} {y_base} {x2} {y_base}")
            else:
                # Create default baseline from bbox
                y_base = y2 - 5  # Slightly above bottom
                text_line.set('BASELINE', f"{x1} {y_base} {x2} {y_base}")
            
            # Add Shape/Polygon
            shape = ET.SubElement(text_line, 'Shape')
            polygon = ET.SubElement(shape, 'Polygon')
            
            polygon_points = line.get('polygon', [])
            if polygon_points and len(polygon_points) >= 3:
                try:
                    points_str = ' '.join(f"{int(p[0])} {int(p[1])}" for p in polygon_points)
                    polygon.set('POINTS', points_str)
                except (ValueError, IndexError):
                    # Create polygon from bbox
                    points_str = f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"
                    polygon.set('POINTS', points_str)
            else:
                # Create polygon from bbox
                points_str = f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"
                polygon.set('POINTS', points_str)
                
            # Add String element
            string = ET.SubElement(text_line, 'String')
            string.set('CONTENT', text_content)
            string.set('HPOS', str(x1))
            string.set('VPOS', str(y1))
            string.set('WIDTH', str(x2 - x1))
            string.set('HEIGHT', str(y2 - y1))
            
            # Add confidence if available
            confidence = line.get('confidence', 0.0)
            if confidence > 0:
                string.set('WC', f"{confidence:.2f}")
                
        # Convert to pretty XML
        rough_string = ET.tostring(alto, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
        
        
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