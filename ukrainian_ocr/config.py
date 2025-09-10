"""
Configuration management for Ukrainian OCR Pipeline
"""

import os
import json
import yaml
from typing import Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class SegmentationConfig:
    """Configuration for text line segmentation"""
    model_path: str = "kraken_model/blla.mlmodel"
    device: Optional[str] = None
    
@dataclass 
class OCRProcessorConfig:
    """Configuration for OCR text recognition"""
    model_path: str = "cyrillic-trocr/trocr-handwritten-cyrillic"
    device: Optional[str] = None
    batch_size: int = 4
    preprocessing: bool = False
    num_beams: int = 1
    max_length: int = 256

@dataclass
class NERConfig:
    """Configuration for Named Entity Recognition"""
    backend: str = "spacy"  # "spacy" or "transformers"
    model_name: str = "uk_core_news_sm"
    confidence_threshold: float = 0.5

@dataclass
class PostProcessingConfig:
    """Configuration for post-processing steps"""
    extract_person_regions: bool = True
    clustering_eps: float = 300
    min_samples: int = 3
    region_padding: int = 50

@dataclass
class OCRPipelineConfig:
    """Main configuration for the OCR pipeline"""
    
    # Component configurations
    segmentation: SegmentationConfig
    ocr: OCRProcessorConfig
    ner: NERConfig
    post_processing: PostProcessingConfig
    
    # General settings
    device: str = "auto"
    batch_size: int = 4
    verbose: bool = True
    save_intermediate: bool = True
    
    def __init__(
        self,
        segmentation: Optional[SegmentationConfig] = None,
        ocr: Optional[OCRProcessorConfig] = None,
        ner: Optional[NERConfig] = None,
        post_processing: Optional[PostProcessingConfig] = None,
        device: str = "auto",
        batch_size: int = 4,
        verbose: bool = True,
        save_intermediate: bool = True
    ):
        self.segmentation = segmentation or SegmentationConfig()
        self.ocr = ocr or OCRProcessorConfig() 
        self.ner = ner or NERConfig()
        self.post_processing = post_processing or PostProcessingConfig()
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self.save_intermediate = save_intermediate
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'OCRPipelineConfig':
        """Load configuration from JSON or YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OCRPipelineConfig':
        """Create configuration from dictionary"""
        
        # Extract component configs
        segmentation_data = data.get('segmentation', {})
        ocr_data = data.get('ocr', {})
        ner_data = data.get('ner', {})
        post_processing_data = data.get('post_processing', {})
        
        # Create component configs
        segmentation = SegmentationConfig(**segmentation_data)
        ocr = OCRProcessorConfig(**ocr_data)
        ner = NERConfig(**ner_data)
        post_processing = PostProcessingConfig(**post_processing_data)
        
        # Create main config
        return cls(
            segmentation=segmentation,
            ocr=ocr,
            ner=ner,
            post_processing=post_processing,
            device=data.get('device', 'auto'),
            batch_size=data.get('batch_size', 4),
            verbose=data.get('verbose', True),
            save_intermediate=data.get('save_intermediate', True)
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'segmentation': asdict(self.segmentation),
            'ocr': asdict(self.ocr),
            'ner': asdict(self.ner),
            'post_processing': asdict(self.post_processing),
            'device': self.device,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
            'save_intermediate': self.save_intermediate
        }
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save configuration to JSON or YAML file"""
        config_path = Path(config_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        # Save based on file extension
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def update_for_colab(self):
        """Update configuration for optimal Google Colab performance"""
        import torch
        
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = 'cuda'
            # Optimize batch size based on GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:
                self.batch_size = 8
            elif gpu_memory_gb >= 8:
                self.batch_size = 4
            else:
                self.batch_size = 2
        else:
            self.device = 'cpu'
            self.batch_size = 1
        
        # Optimize OCR settings for speed
        self.ocr.device = self.device
        self.ocr.batch_size = self.batch_size
        self.ocr.preprocessing = False  # Skip preprocessing for speed
        self.ocr.num_beams = 1  # Use greedy search for speed
        
        # Enable progress tracking
        self.verbose = True
        self.save_intermediate = True
    
    def update_for_cpu(self):
        """Update configuration for CPU-only inference"""
        self.device = 'cpu'
        self.batch_size = 1
        self.ocr.device = 'cpu'
        self.ocr.batch_size = 1
        self.segmentation.device = 'cpu'
    
    def update_for_gpu(self, gpu_memory_gb: Optional[float] = None):
        """Update configuration for GPU inference"""
        self.device = 'cuda'
        
        # Determine batch size based on GPU memory
        if gpu_memory_gb is None:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            else:
                gpu_memory_gb = 4  # Default assumption
        
        if gpu_memory_gb >= 16:
            self.batch_size = 8
        elif gpu_memory_gb >= 8:
            self.batch_size = 4
        else:
            self.batch_size = 2
        
        self.ocr.device = 'cuda'
        self.ocr.batch_size = self.batch_size
        self.segmentation.device = 'cuda'

# For backward compatibility
OCRConfig = OCRPipelineConfig