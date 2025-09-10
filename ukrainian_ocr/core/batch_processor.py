"""
Batch processing utilities for Ukrainian OCR Pipeline
"""

import logging
import time
import os
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class BatchProcessor:
    """Handles batch processing of images with progress tracking"""
    
    def __init__(self, max_workers: Optional[int] = None, verbose: bool = True):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.verbose = verbose
        
    def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        output_dir: Optional[str] = None,
        description: str = "Processing",
        **kwargs
    ) -> List[Dict]:
        """
        Process items in batch with progress tracking
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            output_dir: Output directory for results
            description: Progress bar description
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processing results
        """
        results = []
        
        if self.verbose:
            pbar = tqdm(total=len(items), desc=description)
        else:
            pbar = None
        
        start_time = time.time()
        
        try:
            if self.max_workers == 1:
                # Sequential processing
                for item in items:
                    try:
                        result = processor_func(item, output_dir=output_dir, **kwargs)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing {item}: {e}")
                        results.append({
                            'success': False,
                            'error': str(e),
                            'item': item
                        })
                    
                    if pbar:
                        pbar.update(1)
                        
            else:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_item = {
                        executor.submit(processor_func, item, output_dir=output_dir, **kwargs): item
                        for item in items
                    }
                    
                    for future in as_completed(future_to_item):
                        item = future_to_item[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.logger.error(f"Error processing {item}: {e}")
                            results.append({
                                'success': False,
                                'error': str(e),
                                'item': item
                            })
                        
                        if pbar:
                            pbar.update(1)
            
        finally:
            if pbar:
                pbar.close()
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get('success', False))
        
        if self.verbose:
            self.logger.info(f"Batch processing complete: {successful}/{len(items)} successful")
            self.logger.info(f"Total time: {total_time:.1f}s, Average: {total_time/len(items):.1f}s per item")
        
        return results
    
    def process_images_batch(
        self,
        image_paths: List[str],
        pipeline_func: Callable,
        output_dir: str,
        batch_size: int = 4,
        **kwargs
    ) -> List[Dict]:
        """
        Process images in batches for memory efficiency
        
        Args:
            image_paths: List of image file paths
            pipeline_func: Pipeline processing function
            output_dir: Output directory
            batch_size: Number of images per batch
            **kwargs: Additional pipeline arguments
            
        Returns:
            List of processing results
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            if self.verbose:
                batch_num = i // batch_size + 1
                total_batches = (len(image_paths) + batch_size - 1) // batch_size
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")
            
            # Process current batch
            batch_results = self.process_batch(
                batch_paths,
                pipeline_func,
                output_dir=output_dir,
                description=f"Batch {i//batch_size + 1}",
                **kwargs
            )
            
            all_results.extend(batch_results)
            
            # Optional memory cleanup between batches
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
        return all_results
    
    def validate_inputs(self, image_paths: List[str]) -> List[str]:
        """
        Validate input image paths
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of valid image paths
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        valid_paths = []
        
        for path in image_paths:
            path_obj = Path(path)
            
            if not path_obj.exists():
                self.logger.warning(f"File not found: {path}")
                continue
                
            if path_obj.suffix.lower() not in valid_extensions:
                self.logger.warning(f"Unsupported file format: {path}")
                continue
                
            valid_paths.append(str(path_obj.absolute()))
        
        if self.verbose:
            self.logger.info(f"Validated {len(valid_paths)}/{len(image_paths)} input files")
        
        return valid_paths
    
    def create_output_structure(self, output_dir: str, image_paths: List[str]) -> Dict[str, str]:
        """
        Create organized output directory structure
        
        Args:
            output_dir: Base output directory
            image_paths: List of input image paths
            
        Returns:
            Dictionary mapping output types to directories
        """
        output_dirs = {
            'alto': os.path.join(output_dir, 'alto_xml'),
            'enhanced_alto': os.path.join(output_dir, 'enhanced_alto_xml'),
            'visualizations': os.path.join(output_dir, 'visualizations'),
            'person_regions': os.path.join(output_dir, 'person_regions'),
            'logs': os.path.join(output_dir, 'logs')
        }
        
        # Create directories
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        if self.verbose:
            self.logger.info(f"Created output structure in: {output_dir}")
        
        return output_dirs
    
    def generate_summary_report(self, results: List[Dict], output_dir: str) -> str:
        """
        Generate processing summary report
        
        Args:
            results: List of processing results
            output_dir: Output directory
            
        Returns:
            Path to summary report
        """
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        total_time = sum(r.get('processing_time', 0) for r in successful)
        avg_time = total_time / len(successful) if successful else 0
        
        report_lines = [
            "Ukrainian OCR Pipeline - Processing Summary",
            "=" * 50,
            f"Total images processed: {len(results)}",
            f"Successful: {len(successful)}",
            f"Failed: {len(failed)}",
            f"Success rate: {len(successful)/len(results)*100:.1f}%",
            f"Total processing time: {total_time:.1f}s",
            f"Average time per image: {avg_time:.1f}s",
            "",
        ]
        
        if failed:
            report_lines.extend([
                "Failed Images:",
                "-" * 20,
            ])
            for result in failed:
                report_lines.append(f"- {result.get('item', 'Unknown')}: {result.get('error', 'Unknown error')}")
            report_lines.append("")
        
        if successful:
            report_lines.extend([
                "Processing Statistics:",
                "-" * 20,
            ])
            
            total_lines = sum(r.get('lines_detected', 0) for r in successful)
            total_text_lines = sum(r.get('lines_with_text', 0) for r in successful)
            
            report_lines.extend([
                f"Total text lines detected: {total_lines}",
                f"Lines with recognized text: {total_text_lines}",
                f"Text recognition rate: {total_text_lines/total_lines*100 if total_lines > 0 else 0:.1f}%",
            ])
        
        # Save report
        report_path = os.path.join(output_dir, 'processing_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        if self.verbose:
            self.logger.info(f"Summary report saved to: {report_path}")
        
        return report_path