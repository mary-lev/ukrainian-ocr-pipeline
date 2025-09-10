"""
Model management utilities for Ukrainian OCR Pipeline
"""

import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, snapshot_download

class ModelManager:
    """Manages model loading and caching"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.cache', 'ukrainian_ocr')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_trocr_model(self, model_name: str = "cyrillic-trocr/trocr-handwritten-cyrillic") -> Dict[str, Any]:
        """
        Load TrOCR model and processor
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Dictionary with model and processor
        """
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            self.logger.info(f"Loading TrOCR model: {model_name}")
            
            # Load processor and model
            processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
            model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Move to appropriate device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()
            
            self.logger.info(f"TrOCR model loaded successfully on {device}")
            
            return {
                'model': model,
                'processor': processor,
                'device': device
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load TrOCR model {model_name}: {e}")
            return {
                'model': None,
                'processor': None,
                'device': 'cpu',
                'error': str(e)
            }
    
    def load_kraken_model(self, model_path: str = "kraken_model/blla.mlmodel") -> Optional[Any]:
        """
        Load Kraken segmentation model
        
        Args:
            model_path: Path to Kraken model file
            
        Returns:
            Kraken model or None if failed
        """
        try:
            # Try to import kraken
            from kraken import models
            
            # Check if model file exists
            if os.path.exists(model_path):
                model = models.load_any(model_path)
                self.logger.info(f"Kraken model loaded: {model_path}")
                return model
            else:
                self.logger.warning(f"Kraken model not found: {model_path}")
                return None
                
        except ImportError:
            self.logger.warning("Kraken not installed, using placeholder segmentation")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load Kraken model: {e}")
            return None
    
    def load_spacy_model(self, model_name: str = "uk_core_news_sm") -> Optional[Any]:
        """
        Load spaCy NER model
        
        Args:
            model_name: spaCy model name
            
        Returns:
            spaCy model or None if failed
        """
        try:
            import spacy
            
            # Try to load the model
            nlp = spacy.load(model_name)
            self.logger.info(f"spaCy model loaded: {model_name}")
            return nlp
            
        except OSError:
            self.logger.warning(f"spaCy model not found: {model_name}")
            self.logger.info("Install with: python -m spacy download uk_core_news_sm")
            return None
        except ImportError:
            self.logger.warning("spaCy not installed")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            return None
    
    def load_transformers_ner(self, model_name: str = "dbmdz/bert-base-multilingual-cased") -> Optional[Any]:
        """
        Load Transformers NER pipeline
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Transformers NER pipeline or None if failed
        """
        try:
            from transformers import pipeline
            
            # Create NER pipeline
            ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                cache_dir=self.cache_dir
            )
            
            self.logger.info(f"Transformers NER loaded: {model_name}")
            return ner_pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to load Transformers NER: {e}")
            return None
    
    def download_model(self, repo_id: str, filename: Optional[str] = None) -> str:
        """
        Download model from HuggingFace Hub
        
        Args:
            repo_id: Repository ID on HuggingFace Hub
            filename: Specific file to download (optional)
            
        Returns:
            Path to downloaded model
        """
        try:
            if filename:
                # Download specific file
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=self.cache_dir
                )
            else:
                # Download entire repository
                model_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=self.cache_dir
                )
            
            self.logger.info(f"Model downloaded: {repo_id} -> {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to download model {repo_id}: {e}")
            raise
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about a model file
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with model information
        """
        info = {
            'exists': False,
            'path': model_path,
            'size_mb': 0,
            'type': 'unknown'
        }
        
        try:
            path_obj = Path(model_path)
            
            if path_obj.exists():
                info['exists'] = True
                info['size_mb'] = path_obj.stat().st_size / 1024 / 1024
                
                # Determine model type from extension
                suffix = path_obj.suffix.lower()
                if suffix == '.mlmodel':
                    info['type'] = 'kraken'
                elif suffix in ['.bin', '.safetensors']:
                    info['type'] = 'transformers'
                elif suffix == '.pkl':
                    info['type'] = 'spacy'
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def cleanup_cache(self, keep_recent_days: int = 7):
        """
        Clean up old cached models
        
        Args:
            keep_recent_days: Number of recent days to keep
        """
        try:
            import time
            
            cutoff_time = time.time() - (keep_recent_days * 24 * 3600)
            cache_path = Path(self.cache_dir)
            
            if not cache_path.exists():
                return
            
            cleaned_count = 0
            for file_path in cache_path.rglob('*'):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            cleaned_count += 1
                    except Exception:
                        pass
            
            self.logger.info(f"Cleaned up {cleaned_count} old cached files")
            
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")
    
    def list_available_models(self) -> Dict[str, list]:
        """
        List available models by type
        
        Returns:
            Dictionary mapping model types to available models
        """
        models = {
            'trocr': [
                'cyrillic-trocr/trocr-handwritten-cyrillic',
                'microsoft/trocr-base-handwritten',
                'microsoft/trocr-large-handwritten'
            ],
            'kraken': [
                'kraken_model/blla.mlmodel',
                'kraken_model/default.mlmodel'
            ],
            'spacy_ner': [
                'uk_core_news_sm',
                'uk_core_news_md',
                'uk_core_news_lg'
            ],
            'transformers_ner': [
                'dbmdz/bert-base-multilingual-cased',
                'xlm-roberta-base',
                'bert-base-multilingual-cased'
            ]
        }
        
        return models