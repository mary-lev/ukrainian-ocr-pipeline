"""
Command Line Interface for Ukrainian OCR Pipeline
"""

import click
import os
import sys
from pathlib import Path
from typing import List

from .core.pipeline import UkrainianOCRPipeline
from .core.config import OCRConfig
from .utils.gpu import check_gpu_availability

@click.group()
@click.version_option()
def main():
    """Ukrainian OCR Pipeline - High-performance OCR for historical documents"""
    pass

@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='./ocr_output', help='Output directory')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--device', '-d', type=click.Choice(['cpu', 'cuda', 'auto']), default='auto', help='Processing device')
@click.option('--batch-size', '-b', type=int, help='Batch size for processing')
@click.option('--save-intermediate', '-i', is_flag=True, help='Save intermediate processing steps')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def process(input_path, output, config, device, batch_size, save_intermediate, verbose):
    """Process images or directory of images"""
    
    # Setup
    if verbose:
        click.echo("🇺🇦 Ukrainian OCR Pipeline")
        click.echo("=" * 40)
    
    # Load configuration
    if config:
        ocr_config = OCRConfig.from_file(config)
        if verbose:
            click.echo(f"📄 Loaded config from: {config}")
    else:
        ocr_config = OCRConfig()
    
    # Override config with CLI options
    if device != 'auto':
        ocr_config.device = device
    if batch_size:
        ocr_config.batch_size = batch_size
    ocr_config.verbose = verbose
    ocr_config.save_intermediate = save_intermediate
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    if verbose:
        if gpu_info['cuda_available']:
            click.echo(f"🎮 GPU detected: {gpu_info['gpu_names'][0]}")
        else:
            click.echo("💻 Using CPU (no GPU detected)")
    
    # Initialize pipeline
    try:
        pipeline = UkrainianOCRPipeline(config=ocr_config, verbose=verbose)
        if verbose:
            click.echo("✅ Pipeline initialized")
    except Exception as e:
        click.echo(f"❌ Error initializing pipeline: {e}", err=True)
        sys.exit(1)
    
    # Determine input files
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        image_files = [input_path]
    elif input_path.is_dir():
        # Directory - find image files
        extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            click.echo(f"❌ No image files found in: {input_path}", err=True)
            sys.exit(1)
    else:
        click.echo(f"❌ Invalid input path: {input_path}", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo(f"📁 Found {len(image_files)} image(s) to process")
    
    # Process images
    try:
        if len(image_files) == 1:
            # Single image
            result = pipeline.process_single_image(
                image_files[0], 
                output_dir=output,
                save_intermediate=save_intermediate
            )
            results = [result]
        else:
            # Batch processing
            results = pipeline.process_batch(
                image_files,
                output_dir=output, 
                save_intermediate=save_intermediate
            )
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        if verbose or failed > 0:
            click.echo(f"\n📊 Processing Summary:")
            click.echo(f"✅ Successful: {successful}/{len(results)}")
            if failed > 0:
                click.echo(f"❌ Failed: {failed}/{len(results)}")
        
        # Show stats
        stats = pipeline.get_stats()
        if verbose:
            click.echo(f"⏱️  Total time: {stats['total_processing_time']:.1f}s")
            click.echo(f"📈 Average per image: {stats['average_time_per_image']:.1f}s")
        
        # Cleanup
        pipeline.cleanup()
        
        if successful > 0:
            click.echo(f"🎉 Results saved to: {output}")
        
    except KeyboardInterrupt:
        click.echo("\n⛔ Processing interrupted by user")
        pipeline.cleanup()
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Processing error: {e}", err=True)
        pipeline.cleanup()
        sys.exit(1)

@main.command()
@click.argument('alto_file', type=click.Path(exists=True))
@click.argument('image_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='./person_regions', help='Output directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def extract_persons(alto_file, image_file, output, verbose):
    """Extract person-dense regions from enhanced ALTO file"""
    
    from .core.enhancement import PersonRegionExtractor
    
    if verbose:
        click.echo("🎯 Extracting Person-Dense Regions")
        click.echo("=" * 40)
    
    try:
        extractor = PersonRegionExtractor()
        
        # Extract person lines
        person_lines = extractor.extract_person_lines(alto_file)
        
        if verbose:
            click.echo(f"👤 Found {len(person_lines)} person lines")
        
        if len(person_lines) == 0:
            click.echo("⚠️  No person entities found in ALTO file")
            return
        
        # Find dense region
        dense_region = extractor.find_dense_region(person_lines)
        
        if dense_region:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract region
            base_name = Path(image_file).stem
            crop_path = output_dir / f"{base_name}_person_region.jpg"
            viz_path = output_dir / f"{base_name}_visualization.jpg"
            
            # Extract cropped region
            extractor.extract_region_image(image_file, dense_region, crop_path)
            
            # Create visualization
            extractor.visualize_regions(image_file, person_lines, dense_region, viz_path)
            
            if verbose:
                click.echo(f"📊 Dense region: {dense_region['num_lines']} person lines")
                click.echo(f"📐 Size: {dense_region['width']}x{dense_region['height']} pixels")
            
            click.echo(f"💾 Cropped region: {crop_path}")
            click.echo(f"🎨 Visualization: {viz_path}")
            
        else:
            click.echo("❌ No dense person region found")
            
    except Exception as e:
        click.echo(f"❌ Error extracting person regions: {e}", err=True)
        sys.exit(1)

@main.command()
def info():
    """Show system and GPU information"""
    
    click.echo("🇺🇦 Ukrainian OCR Pipeline - System Information")
    click.echo("=" * 50)
    
    # Package version
    try:
        from . import __version__
        click.echo(f"📦 Package version: {__version__}")
    except ImportError:
        click.echo("📦 Package version: Unknown")
    
    # Python version
    click.echo(f"🐍 Python version: {sys.version.split()[0]}")
    
    # GPU information
    gpu_info = check_gpu_availability()
    
    if gpu_info['cuda_available']:
        click.echo(f"🎮 GPU: Available")
        click.echo(f"   Count: {gpu_info['gpu_count']}")
        for i, (name, memory) in enumerate(zip(gpu_info['gpu_names'], gpu_info['gpu_memory'])):
            click.echo(f"   GPU {i}: {name} ({memory:.1f}GB)")
        click.echo(f"   Recommended device: {gpu_info['recommended_device']}")
    else:
        click.echo("💻 GPU: Not available (CPU only)")
    
    # Dependencies
    click.echo("\n📚 Dependencies:")
    deps = ['torch', 'transformers', 'spacy', 'kraken', 'opencv-python']
    
    for dep in deps:
        try:
            module = __import__(dep.replace('-', '_'))
            version = getattr(module, '__version__', 'Unknown')
            click.echo(f"   ✅ {dep}: {version}")
        except ImportError:
            click.echo(f"   ❌ {dep}: Not installed")

@main.command()
@click.option('--output', '-o', default='./config.yaml', help='Output configuration file')
def create_config(output):
    """Create a sample configuration file"""
    
    config = OCRConfig()
    config.save(output)
    
    click.echo(f"📄 Created configuration file: {output}")
    click.echo("✏️  Edit the file to customize settings")

if __name__ == '__main__':
    main()