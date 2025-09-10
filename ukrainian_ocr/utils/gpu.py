"""
GPU utilities and device optimization for Ukrainian OCR Pipeline
"""

import os
import logging
import torch
import psutil
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return detailed information
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'recommended_device': 'cpu'
    }
    
    if torch.cuda.is_available():
        gpu_info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(gpu_info['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            gpu_info['gpu_names'].append(props.name)
            gpu_info['gpu_memory'].append(props.total_memory / 1024**3)  # GB
            
        # Recommend CUDA if sufficient memory
        if gpu_info['gpu_memory'] and gpu_info['gpu_memory'][0] >= 4:
            gpu_info['recommended_device'] = 'cuda'
        
        logger.info(f"GPU detected: {gpu_info['gpu_names'][0]} ({gpu_info['gpu_memory'][0]:.1f}GB)")
    else:
        logger.info("No GPU detected, using CPU")
        
    return gpu_info

def optimize_for_device(device: str):
    """
    Optimize PyTorch settings for the target device
    
    Args:
        device: Target device ('cuda' or 'cpu')
    """
    
    if device == 'cuda' and torch.cuda.is_available():
        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Clear cache
        torch.cuda.empty_cache()
        
        logger.info("GPU optimizations applied")
        
    else:
        # CPU optimizations
        # Set number of threads for CPU inference
        num_cores = psutil.cpu_count(logical=False)
        torch.set_num_threads(min(num_cores, 8))  # Cap at 8 threads
        
        # Enable CPU optimizations
        torch.backends.mkldnn.enabled = True
        
        logger.info(f"CPU optimizations applied (using {torch.get_num_threads()} threads)")

def get_optimal_batch_size(device: str, model_type: str = 'trocr') -> int:
    """
    Determine optimal batch size based on device and model type
    
    Args:
        device: Target device
        model_type: Type of model ('trocr', 'ner', 'segmentation')
        
    Returns:
        Recommended batch size
    """
    
    if device == 'cuda' and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Batch size recommendations by GPU memory and model type
        batch_sizes = {
            'trocr': {
                16: 16,   # High-end GPU
                8: 8,     # Mid-range GPU
                4: 4,     # Budget GPU
                0: 2      # Fallback
            },
            'ner': {
                16: 32,   # NER models are generally lighter
                8: 16,
                4: 8,
                0: 4
            },
            'segmentation': {
                16: 4,    # Segmentation can be memory-intensive
                8: 2,
                4: 1,
                0: 1
            }
        }
        
        model_batch_sizes = batch_sizes.get(model_type, batch_sizes['trocr'])
        
        for memory_threshold in sorted(model_batch_sizes.keys(), reverse=True):
            if gpu_memory_gb >= memory_threshold:
                return model_batch_sizes[memory_threshold]
        
        return model_batch_sizes[0]  # Fallback
    
    else:
        # CPU batch sizes (generally smaller)
        return {
            'trocr': 1,
            'ner': 4,
            'segmentation': 1
        }.get(model_type, 1)

def monitor_gpu_memory() -> Dict[str, float]:
    """
    Monitor current GPU memory usage
    
    Returns:
        Dictionary with memory statistics (in GB)
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    memory_stats = {}
    
    for i in range(torch.cuda.device_count()):
        memory_stats[f'gpu_{i}'] = {
            'allocated': torch.cuda.memory_allocated(i) / 1024**3,
            'reserved': torch.cuda.memory_reserved(i) / 1024**3,
            'total': torch.cuda.get_device_properties(i).total_memory / 1024**3
        }
        
        # Calculate utilization
        allocated = memory_stats[f'gpu_{i}']['allocated']
        total = memory_stats[f'gpu_{i}']['total']
        memory_stats[f'gpu_{i}']['utilization'] = (allocated / total) * 100
    
    return memory_stats

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cleaned up")

def setup_mixed_precision() -> Optional[Any]:
    """
    Setup mixed precision training/inference if supported
    
    Returns:
        GradScaler for mixed precision or None if not supported
    """
    if torch.cuda.is_available():
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            logger.info("Mixed precision enabled")
            return scaler
        except ImportError:
            logger.warning("Mixed precision not supported")
            return None
    return None

class GPUMemoryManager:
    """Context manager for GPU memory management"""
    
    def __init__(self, device: str):
        self.device = device
        self.initial_memory = None
        
    def __enter__(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated()
            cleanup_gpu_memory()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == 'cuda' and torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_diff = (final_memory - self.initial_memory) / 1024**2  # MB
            
            if memory_diff > 100:  # If more than 100MB difference
                logger.warning(f"Memory usage increased by {memory_diff:.1f}MB")
            
            cleanup_gpu_memory()

# Utility functions for Colab
def is_colab_environment() -> bool:
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_colab_gpu():
    """Setup GPU in Google Colab environment"""
    if is_colab_environment():
        # Check if GPU is enabled in Colab
        gpu_info = check_gpu_availability()
        
        if not gpu_info['cuda_available']:
            logger.warning(
                "GPU not detected in Colab. "
                "Enable GPU: Runtime -> Change runtime type -> GPU"
            )
        else:
            logger.info("Colab GPU setup complete")
            
        return gpu_info
    else:
        logger.info("Not running in Colab environment")
        return check_gpu_availability()