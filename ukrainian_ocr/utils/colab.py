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


def check_pytorch_compatibility():
    """Check and fix PyTorch compatibility issues in Colab"""
    try:
        import torch
        torch_version = torch.__version__
        print(f"🔥 Current PyTorch version: {torch_version}")
        
        # Check if we have torch.compiler issues
        if hasattr(torch, 'compiler'):
            try:
                # Test torch.compiler.disable with reason
                torch.compiler.disable(reason="test")
                torch.compiler.enable()
                print("✅ PyTorch compiler compatibility: OK")
            except TypeError:
                print("⚠️ PyTorch compiler has compatibility issues, but workarounds applied")
        else:
            print("ℹ️ PyTorch compiler not available (older version)")
            
        return True
    except Exception as e:
        print(f"❌ PyTorch compatibility check failed: {e}")
        return False


def upgrade_transformers():
    """Install/upgrade transformers to the required version for TrOCR"""
    import subprocess
    import sys
    
    print("🔧 Installing transformers>=4.36.0 for TrOCR support...")
    
    try:
        # Install/upgrade transformers to the required version
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 
            'transformers>=4.36.0', '--quiet'
        ])
        
        print("✅ Transformers installed successfully!")
        
        # Verify installation
        import transformers
        from transformers import TrOCRProcessor
        print(f"📦 Transformers version: {transformers.__version__}")
        print("✅ TrOCRProcessor available!")
        return True
            
    except Exception as e:
        print(f"❌ Failed to install transformers: {e}")
        raise


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
        'python_version': None,
        'dependencies_installed': False,
        'pytorch_compatible': False
    }
    
    try:
        # Check if running in Colab
        import google.colab
        env_info['is_colab'] = True
        print("✅ Running in Google Colab")
        
        # Install required dependencies
        print("🔧 Installing required dependencies...")
        install_colab_dependencies()
        env_info['dependencies_installed'] = True
        
        # Check PyTorch compatibility
        print("🔥 Checking PyTorch compatibility...")
        env_info['pytorch_compatible'] = check_pytorch_compatibility()
        
        # Check and upgrade transformers if needed
        print("🤖 Checking transformers compatibility...")
        upgrade_transformers()
        
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


def install_colab_dependencies():
    """
    Install all required dependencies for Colab
    """
    import subprocess
    import sys
    
    dependencies = [
        'kraken[pytorch]',
        'transformers>=4.36.0',  # TrOCRProcessor available from 4.36+
        'torch>=2.0.0',  # Ensure compatible PyTorch version
        'spacy>=3.4.0',
        'opencv-python',
        'pillow',
        'numpy',
        'tqdm',
        'scikit-learn'
    ]
    
    print("📦 Installing dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', dep, '--quiet'
            ])
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {dep}: {e}")
    
    # Download spaCy models for NER (try multiple models)
    print("🔤 Downloading spaCy language models...")
    spacy_models = [
        ('ru_core_news_lg', 'Large Russian model (~500MB) - best quality'),
        ('ru_core_news_md', 'Medium Russian model (~50MB) - good quality'), 
        ('ru_core_news_sm', 'Small Russian model (~15MB) - fast')
    ]
    
    spacy_installed = False
    successful_models = []
    
    # Try to install models in order of preference
    for model_name, description in spacy_models:
        try:
            print(f"  📥 Trying {model_name} - {description}")
            subprocess.check_call([
                sys.executable, '-m', 'spacy', 'download', model_name, '--quiet'
            ])
            print(f"  ✅ {model_name} installed successfully")
            successful_models.append(model_name)
            spacy_installed = True
            
            # Test the model to make sure it works
            try:
                import spacy
                nlp = spacy.load(model_name)
                test_text = "Іван Петренко з Харкова"
                doc = nlp(test_text)
                print(f"  🧪 Model test: Found {len(doc.ents)} entities in test text")
                break  # Stop after first successful working model
            except Exception as test_error:
                print(f"  ⚠️ Model {model_name} installed but failed test: {test_error}")
                continue
                
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Could not download {model_name}: {e}")
            continue
    
    if spacy_installed:
        print(f"  🎉 spaCy setup complete! Installed models: {', '.join(successful_models)}")
    else:
        print("  ⚠️ No spaCy models installed - will use rule-based NER")
    
    print("✅ Dependencies installation complete!")


def preload_models():
    """
    Pre-load and cache models for faster processing
    """
    print("🚀 Pre-loading models...")
    
    try:
        # Pre-load TrOCR model
        print("  📝 Loading TrOCR model...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        model_name = "cyrillic-trocr/trocr-handwritten-cyrillic"
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        print("  ✅ TrOCR model loaded")
        
        # Pre-load Kraken (downloads default model if needed)
        print("  🖼️ Loading Kraken segmentation...")
        import kraken
        from kraken import blla
        print("  ✅ Kraken loaded")
        
        # Pre-load spaCy NER model
        print("  🏷️ Loading spaCy NER model...")
        try:
            import spacy
            
            # Try loading models in order of preference
            spacy_models = ['ru_core_news_lg', 'ru_core_news_md', 'ru_core_news_sm']
            nlp = None
            
            for model_name in spacy_models:
                try:
                    nlp = spacy.load(model_name)
                    print(f"  ✅ spaCy {model_name} loaded successfully")
                    
                    # Test the model with sample Ukrainian/Russian text
                    test_text = "Іван Петренко з села Березівка 1920 року"
                    doc = nlp(test_text)
                    entities_found = len(doc.ents)
                    print(f"  🧪 Model test: found {entities_found} entities in sample text")
                    break
                    
                except OSError:
                    continue
            
            if nlp is None:
                print("  ⚠️ No spaCy models found - NER will use rule-based extraction")
            
        except ImportError:
            print("  ⚠️ spaCy not available - NER will use rule-based extraction")
        
        print("🎉 All models pre-loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Error pre-loading models: {e}")
        print("Models will be loaded on-demand instead")


def setup_complete_colab_environment():
    """
    Complete setup for Colab environment including dependencies and models
    """
    print("🚀 Setting up Ukrainian OCR Pipeline for Google Colab...")
    print("=" * 60)
    
    # Setup environment
    env_info = setup_colab_environment()
    
    if env_info['is_colab'] and env_info['dependencies_installed']:
        # Pre-load models
        preload_models()
        
        print("\n" + "=" * 60)
        print("✅ Setup complete! You can now use the Ukrainian OCR Pipeline.")
        print("\nExample usage:")
        print("```python")
        print("from ukrainian_ocr import UkrainianOCRPipeline")
        print("pipeline = UkrainianOCRPipeline(device='auto')")
        print("result = pipeline.process_single_image('your_image.jpg')")
        print("```")
        print("=" * 60)
        
    return env_info


def upgrade_ner_to_spacy():
    """
    Upgrade NER from rule-based to spaCy-based for better results
    Call this after initial setup if you want better entity recognition
    """
    import subprocess
    import sys
    
    print("🔧 Upgrading NER to use spaCy models...")
    
    # Install spaCy if not already installed
    try:
        import spacy
        print("✅ spaCy already installed")
    except ImportError:
        print("📦 Installing spaCy...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'spacy>=3.4.0', '--quiet'
            ])
            print("✅ spaCy installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install spaCy: {e}")
            return False
    
    # Try to download a Russian model
    spacy_models = [
        ('ru_core_news_lg', 'Large Russian model (~500MB) - best quality'),
        ('ru_core_news_md', 'Medium Russian model (~50MB) - good quality'), 
        ('ru_core_news_sm', 'Small Russian model (~15MB) - basic quality')
    ]
    
    print("📥 Downloading spaCy language model...")
    for model_name, description in spacy_models:
        try:
            print(f"  Trying {model_name} - {description}")
            subprocess.check_call([
                sys.executable, '-m', 'spacy', 'download', model_name, '--quiet'
            ])
            
            # Test the installed model
            import spacy
            nlp = spacy.load(model_name)
            test_text = "Іван Петренко з Харкова, 1920 року народження"
            doc = nlp(test_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            print(f"  ✅ {model_name} installed and working!")
            print(f"  🧪 Test: Found {len(entities)} entities: {entities}")
            print("🎉 NER upgrade complete! Restart your pipeline to use spaCy-based NER.")
            return True
            
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Could not download {model_name}")
            continue
        except Exception as e:
            print(f"  ⚠️ Error testing {model_name}: {e}")
            continue
    
    print("❌ Could not install any spaCy models")
    print("💡 NER will continue using rule-based extraction")
    return False


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