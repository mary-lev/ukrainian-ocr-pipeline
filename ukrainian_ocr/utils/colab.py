"""
Google Colab utilities for the Ukrainian OCR Pipeline
"""

import os
import zipfile
import glob
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
        # Import Colab files module
        from google.colab import files
        
        # Set default filename
        if filename is None:
            filename = "ukrainian_ocr_results.zip"
            
        # Default patterns if not specified
        if include_patterns is None:
            include_patterns = ["*.xml", "*.png", "*.jpg", "*.json", "*.txt"]
        
        # Find all matching files
        output_path = Path(output_dir)
        all_files = []
        
        for pattern in include_patterns:
            matching_files = list(output_path.rglob(pattern))
            all_files.extend(matching_files)
        
        if not all_files:
            print(f"⚠️ No files found in {output_dir} matching patterns: {include_patterns}")
            return None
        
        # Create zip file
        zip_path = f"/content/{filename}"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in all_files:
                # Create relative path for the zip
                relative_path = file_path.relative_to(output_path)
                zipf.write(file_path, relative_path)
                
        print(f"✅ Created zip file with {len(all_files)} files")
        print(f"📦 Zip file: {zip_path}")
        
        # List contents of zip for verification
        print("\n📋 Contents:")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for name in sorted(zipf.namelist()):
                print(f"  - {name}")
        
        # Download the zip file
        files.download(zip_path)
        print("✅ Results downloaded successfully!")
        
        return zip_path
        
    except ImportError:
        print("❌ This function requires Google Colab environment")
        print(f"💡 Files are available in: {output_dir}")
        return None
    except Exception as e:
        print(f"❌ Error creating download: {e}")
        return None


def setup_colab_environment():
    """
    Setup Google Colab environment for Ukrainian OCR Pipeline
    
    Returns:
        Dictionary with environment information
    """
    env_info = {
        'is_colab': False,
        'has_gpu': False,
        'gpu_name': None,
        'python_version': None
    }
    
    try:
        # Check if running in Colab
        import google.colab
        env_info['is_colab'] = True
        print("✅ Running in Google Colab")
        
        # Check GPU availability
        import torch
        if torch.cuda.is_available():
            env_info['has_gpu'] = True
            env_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {env_info['gpu_name']} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ No GPU detected - will use CPU processing")
        
        # Get Python version
        import sys
        env_info['python_version'] = sys.version
        print(f"🐍 Python: {sys.version.split()[0]}")
        
        # Check PyTorch version
        print(f"🔥 PyTorch: {torch.__version__}")
        
    except ImportError:
        print("❌ Not running in Google Colab")
    
    return env_info


def list_output_files(output_dir: str) -> List[str]:
    """
    List all output files in the results directory
    
    Args:
        output_dir: Directory to scan for files
        
    Returns:
        List of file paths found
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return []
    
    # Find all files
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
    
    print(f"📁 Found {len(all_files)} files in {output_dir}:")
    
    for ext, files in sorted(file_types.items()):
        print(f"  {ext or '(no extension)'}: {len(files)} files")
        if len(files) <= 5:  # Show filenames if not too many
            for file_path in files:
                print(f"    - {Path(file_path).name}")
        else:
            print(f"    - {Path(files[0]).name}")
            print(f"    - ... and {len(files)-1} more")
    
    return all_files


def get_processing_results_summary(result: dict) -> str:
    """
    Create a formatted summary of processing results
    
    Args:
        result: Processing result dictionary from pipeline
        
    Returns:
        Formatted summary string
    """
    if not result.get('success', False):
        return f"❌ Processing failed: {result.get('error', 'Unknown error')}"
    
    summary = []
    summary.append(f"✅ Successfully processed: {Path(result['image_path']).name}")
    summary.append(f"⏱️ Processing time: {result['processing_time']:.2f}s")
    summary.append(f"📄 Lines detected: {result['lines_detected']}")
    summary.append(f"📝 Lines with text: {result['lines_with_text']}")
    
    if result.get('entities_extracted', 0) > 0:
        summary.append(f"🏷️ Entities found: {result['total_entities']} in {result['entities_extracted']} lines")
    
    # List output files
    output_paths = result.get('output_paths', {})
    if output_paths:
        summary.append("\n📁 Generated files:")
        for file_type, file_path in output_paths.items():
            if file_path and os.path.exists(file_path):
                summary.append(f"  - {file_type}: {Path(file_path).name}")
    
    return "\n".join(summary)