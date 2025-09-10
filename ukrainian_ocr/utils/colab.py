"""
Simple Google Colab utilities for the Ukrainian OCR Pipeline
"""

import os
import zipfile
import torch
from pathlib import Path
from typing import List, Optional


def download_results(
    output_dir: str, 
    filename: Optional[str] = None, 
    include_patterns: Optional[List[str]] = None
) -> str:
    """
    Create a zip file of results and download it in Google Colab
    
    Args:
        output_dir: Directory containing the results to download
        filename: Custom filename for the zip (default: results.zip)
        include_patterns: List of file patterns to include (default: all files)
        
    Returns:
        Path to the created zip file
    """
    try:
        from google.colab import files
        
        if filename is None:
            filename = "ukrainian_ocr_results.zip"
            
        if include_patterns is None:
            include_patterns = ["*.xml", "*.png", "*.jpg", "*.json", "*.txt"]
        
        # Find all matching files
        output_path = Path(output_dir)
        all_files = []
        
        for pattern in include_patterns:
            matching_files = list(output_path.rglob(pattern))
            all_files.extend(matching_files)
        
        if not all_files:
            print(f"⚠️ No files found in {output_dir}")
            return None
        
        # Create zip file
        zip_path = f"/content/{filename}"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in all_files:
                relative_path = file_path.relative_to(output_path)
                zipf.write(file_path, relative_path)
                
        print(f"✅ Created zip with {len(all_files)} files")
        
        # Download the zip file
        files.download(zip_path)
        return zip_path
        
    except ImportError:
        print("❌ This function requires Google Colab")
        return None
    except Exception as e:
        print(f"❌ Error creating download: {e}")
        return None


def check_gpu():
    """Simple GPU check for Colab"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return True
    else:
        print("⚠️ No GPU detected - will use CPU")
        return False


def list_output_files(output_dir: str) -> List[str]:
    """List all output files in the results directory"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Directory not found: {output_dir}")
        return []
    
    all_files = []
    for file_path in output_path.rglob("*"):
        if file_path.is_file():
            all_files.append(str(file_path))
    
    # Group by file type
    file_types = {}
    for file_path in all_files:
        ext = Path(file_path).suffix.lower()
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(file_path)
    
    print(f"📁 Found {len(all_files)} files:")
    for ext, files in sorted(file_types.items()):
        print(f"  {ext or '(no ext)'}: {len(files)} files")
    
    return all_files


def get_processing_summary(result: dict) -> str:
    """Create a simple summary of processing results"""
    if not result.get('success', False):
        return f"❌ Failed: {result.get('error', 'Unknown error')}"
    
    summary = []
    summary.append(f"✅ Processed: {Path(result['image_path']).name}")
    summary.append(f"⏱️ Time: {result['processing_time']:.2f}s")
    summary.append(f"📄 Lines: {result['lines_detected']}")
    summary.append(f"📝 Text lines: {result['lines_with_text']}")
    
    if result.get('total_entities', 0) > 0:
        summary.append(f"🏷️ Entities: {result['total_entities']}")
    
    return " | ".join(summary)