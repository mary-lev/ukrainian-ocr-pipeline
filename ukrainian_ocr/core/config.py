"""
Configuration management for Ukrainian OCR Pipeline
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class SegmentationConfig:
    """Configuration for text segmentation"""
    model_path: str = "kraken_model/blla.mlmodel"
    padding: int = 8
    device: Optional[str] = None
    
@dataclass 
class OCRConfig:
    """Configuration for OCR recognition"""
    model_path: str = "cyrillic-trocr/trocr-handwritten-cyrillic"
    batch_size: Optional[int] = None
    device: Optional[str] = None
    max_length: int = 256
    num_beams: int = 1
    preprocessing: bool = False
    
@dataclass
class NERConfig:
    """Configuration for Named Entity Recognition"""
    backend: str = "spacy"
    backend_config: Dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None
    confidence_threshold: float = 0.7
    
    # Model mappings
    model_mappings: Dict[str, str] = field(default_factory=lambda: {
        "roberta_large_russian": "Eka-Korn/roberta-base-russian-v0-finetuned-ner",
        "spacy_russian": "ru_core_news_lg"
    })
    
    # False positive filtering
    false_positive_words: set = field(default_factory=lambda: {
        "ім'я", "прізвище", "по-батькові", "особа", "людина"
    })

@dataclass
class PostProcessingConfig:
    """Configuration for post-processing"""
    extract_person_regions: bool = True
    clustering_eps: int = 300
    clustering_min_samples: int = 3
    region_padding: int = 50
    
@dataclass
class OCRPipelineConfig:
    """Main configuration for OCR Pipeline"""
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig) 
    ner: NERConfig = field(default_factory=NERConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    # Global settings
    device: Optional[str] = None  # 'cuda', 'cpu', 'auto'
    batch_size: Optional[int] = None  # Auto-determined if None
    verbose: bool = True
    save_intermediate: bool = False
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "OCRPipelineConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OCRPipelineConfig":
        """Create configuration from dictionary"""
        
        # Create sub-configurations
        segmentation = SegmentationConfig(**config_dict.get('segmentation', {}))
        ocr = OCRConfig(**config_dict.get('ocr', {}))
        ner = NERConfig(**config_dict.get('ner', {}))
        post_processing = PostProcessingConfig(**config_dict.get('post_processing', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['segmentation', 'ocr', 'ner', 'post_processing']}
        
        return cls(
            segmentation=segmentation,
            ocr=ocr, 
            ner=ner,
            post_processing=post_processing,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'segmentation': self.segmentation.__dict__,
            'ocr': self.ocr.__dict__,
            'ner': self.ner.__dict__, 
            'post_processing': self.post_processing.__dict__,
            'device': self.device,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
            'save_intermediate': self.save_intermediate
        }
    
    def save(self, config_path: Union[str, Path]):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def update_for_colab(self):
        """Update configuration for Google Colab environment"""
        # Enable GPU if available
        if self.device is None:
            self.device = 'auto'
        
        # Optimize batch sizes for Colab
        if self.batch_size is None:
            self.batch_size = 4  # Conservative for Colab
            
        # Enable progress bars
        self.verbose = True
        
        # Use lighter models if needed
        if not self.ner.backend_config:
            self.ner.backend = "spacy"  # Generally more stable in Colab
    
    def update_for_production(self):
        """Update configuration for production deployment"""
        # Disable verbose logging in production
        self.verbose = False
        
        # Don't save intermediate files
        self.save_intermediate = False
        
        # Use larger batch sizes if GPU available
        if self.device == 'cuda':
            self.batch_size = self.batch_size or 8

# Alias for backward compatibility
OCRConfig = OCRPipelineConfig