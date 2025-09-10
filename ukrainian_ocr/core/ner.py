"""
NER (Named Entity Recognition) module for Ukrainian OCR Pipeline
Advanced multi-backend NER system supporting spaCy, Transformers, and OpenAI
"""

import os
import re
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Entity:
    """Named entity with metadata"""
    text: str
    label: str  # PERSON, LOCATION, ORG, etc.
    confidence: float
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    source_line: Optional[str] = None
    context: Optional[str] = None


class NERBackend(ABC):
    """Abstract base class for NER backends"""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Get backend identifier"""
        pass


class SpacyNERBackend(NERBackend):
    """spaCy-based NER backend"""
    
    def __init__(self, model_name: str = "ru_core_news_lg"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.nlp = None
        self.load_model()
        
    def load_model(self):
        """Load spaCy model"""
        try:
            import spacy
            
            # Try different Russian/Ukrainian models in order
            model_candidates = [
                self.model_name,
                "ru_core_news_lg", 
                "ru_core_news_md", 
                "ru_core_news_sm",
                "uk_core_news_sm"
            ]
            
            # Remove None and duplicates
            model_candidates = list(dict.fromkeys([m for m in model_candidates if m]))
            
            for model_name in model_candidates:
                try:
                    self.nlp = spacy.load(model_name)
                    self.logger.info(f"Loaded spaCy model: {model_name}")
                    self.model_name = model_name
                    return
                except OSError:
                    continue
            
            # If no models found, try auto-installation
            self.logger.warning("No spaCy models found, attempting auto-installation...")
            if self._auto_install_spacy():
                # Retry loading after installation
                return self.load_model()
            
            self.logger.error("No spaCy models available")
            self.nlp = None
            
        except ImportError:
            self.logger.error("spaCy not installed")
            self.nlp = None
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {e}")
            self.nlp = None
            
    def _auto_install_spacy(self) -> bool:
        """Try to auto-install spaCy model"""
        try:
            import subprocess
            import sys
            
            # Try to install ru_core_news_lg
            subprocess.check_call([
                sys.executable, '-m', 'spacy', 'download', 'ru_core_news_lg', '--quiet'
            ])
            self.logger.info("Auto-installed ru_core_news_lg")
            return True
        except:
            try:
                # Fallback to smaller model
                subprocess.check_call([
                    sys.executable, '-m', 'spacy', 'download', 'ru_core_news_sm', '--quiet'
                ])
                self.logger.info("Auto-installed ru_core_news_sm")
                return True
            except:
                return False
            
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using spaCy"""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    confidence=1.0,  # spaCy doesn't provide confidence scores directly
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                )
                entities.append(entity)
                
            return entities
        except Exception as e:
            self.logger.error(f"Error in spaCy extraction: {e}")
            return []
        
    def get_backend_name(self) -> str:
        return f"spacy_{self.model_name}"


class TransformersNERBackend(NERBackend):
    """Transformers-based NER backend for Russian models"""
    
    def __init__(self, model_name: str = "roberta_large_russian"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.pipeline = None
        self.load_model()
        
    def load_model(self):
        """Load transformers model"""
        try:
            from transformers import pipeline
            
            # Map model names to actual HF model IDs
            model_mapping = {
                "roberta_large_russian": "Eka-Korn/roberta-base-russian-v0-finetuned-ner",
                "deeppavlov_ner_bert": "DeepPavlov/rubert-base-cased-sentence", 
                "deeppavlov_ontonotes": "DeepPavlov/bert-base-cased-conversational"
            }
            
            actual_model = model_mapping.get(self.model_name, self.model_name)
            
            self.pipeline = pipeline(
                "ner", 
                model=actual_model,
                tokenizer=actual_model,
                aggregation_strategy="simple",
                device=-1  # CPU by default
            )
            
            self.logger.info(f"Loaded transformers model: {actual_model}")
            
        except Exception as e:
            self.logger.error(f"Error loading transformers model {self.model_name}: {e}")
            self.pipeline = None
            
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using transformers"""
        if not self.pipeline:
            return []
            
        try:
            results = self.pipeline(text)
            entities = []
            
            for result in results:
                # Map common NER labels
                label_mapping = {
                    'B-PER': 'PERSON', 'I-PER': 'PERSON',
                    'B-LOC': 'LOCATION', 'I-LOC': 'LOCATION',
                    'B-ORG': 'ORG', 'I-ORG': 'ORG',
                    'B-MISC': 'MISC', 'I-MISC': 'MISC'
                }
                
                label = label_mapping.get(result.get('entity_group', ''), result.get('entity_group', ''))
                
                entity = Entity(
                    text=result['word'],
                    label=label,
                    confidence=result['score'],
                    start_pos=result['start'],
                    end_pos=result['end']
                )
                entities.append(entity)
                
            return entities
            
        except Exception as e:
            self.logger.error(f"Error in transformers NER extraction: {e}")
            return []
            
    def get_backend_name(self) -> str:
        return f"transformers_{self.model_name}"


class OpenAINERBackend(NERBackend):
    """OpenAI GPT-based NER backend"""
    
    def __init__(self, model: str = "gpt-4o-2024-05-13", api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.client = None
        self.setup_client(api_key)
        
    def setup_client(self, api_key: Optional[str]):
        """Setup OpenAI client"""
        try:
            import openai
            
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                # Try to get from environment
                self.client = openai.OpenAI()
                
            self.logger.info(f"OpenAI client initialized for model: {self.model}")
            
        except Exception as e:
            self.logger.error(f"Error setting up OpenAI client: {e}")
            self.client = None
            
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using OpenAI API"""
        if not self.client:
            return []
            
        try:
            prompt = f"""
Extract all named entities from this Ukrainian/Russian historical document text. 
Return JSON format with entities and their types (PERSON, LOCATION, ORGANIZATION).

Text: "{text}"

Return format:
{{
  "entities": [
    {{"text": "entity_text", "label": "PERSON|LOCATION|ORGANIZATION", "confidence": 0.95}}
  ]
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            entities = []
            
            for ent_data in result.get('entities', []):
                entity = Entity(
                    text=ent_data['text'],
                    label=ent_data['label'],
                    confidence=ent_data.get('confidence', 0.9)
                )
                entities.append(entity)
                
            return entities
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI NER extraction: {e}")
            return []
            
    def get_backend_name(self) -> str:
        return f"openai_{self.model}"


class RuleBasedNERBackend(NERBackend):
    """Rule-based NER backend for Ukrainian/Russian text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Ukrainian/Russian person name patterns
        self.person_patterns = [
            # Full names: First Middle Last
            r'\b[А-ЯІЇЄЁ][а-яіїєё]{2,15}\s+[А-ЯІЇЄЁ][а-яіїєё]{2,15}\s+[А-ЯІЇЄЁ][а-яіїєё]{2,15}\b',
            # First Last
            r'\b[А-ЯІЇЄЁ][а-яіїєё]{2,15}\s+[А-ЯІЇЄЁ][а-яіїєё]{2,15}\b',
            # Common Ukrainian surnames
            r'\b[А-ЯІЇЄЁ][а-яіїєё]*(?:енко|ський|цький|ич|юк|як|ук|ко)\b'
        ]
        
        # Location patterns
        self.location_patterns = [
            r'\b(?:місто|село|селище|хутір|станція)\s+[А-ЯІЇЄЁ][а-яіїєё]{2,20}\b',
            r'\b[А-ЯІЇЄЁ][а-яіїєё]{3,20}(?:ськ|цьк|івка|івці|ичі|енко|ове|іно)\b'
        ]
        
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using rule-based patterns"""
        entities = []
        
        # Extract persons
        for pattern in self.person_patterns:
            for match in re.finditer(pattern, text):
                entity = Entity(
                    text=match.group(),
                    label='PERSON',
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                entities.append(entity)
                
        # Extract locations
        for pattern in self.location_patterns:
            for match in re.finditer(pattern, text):
                entity = Entity(
                    text=match.group(),
                    label='LOCATION',
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                entities.append(entity)
                
        return entities
        
    def get_backend_name(self) -> str:
        return "rule_based"


class NERExtractor:
    """Main NER extractor with multiple backend support"""
    
    def __init__(self, backend: str = "spacy", backend_config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.backend_config = backend_config or {}
        self.backend = self.load_backend(backend)
        
        # Common false positives to filter out  
        self.false_positive_words = {
            "ім'я",  # "name" in Ukrainian
            "прізвище",  # "surname" in Ukrainian
            "по-батькові",  # "patronymic" in Ukrainian
            "особа",  # "person" in Ukrainian
            "людина",  # "human" in Ukrainian
            "чоловік",  # "man" when used generically
            "жінка",  # "woman" when used generically
        }
        
    def load_backend(self, backend_name: str) -> NERBackend:
        """Load specified NER backend"""
        try:
            if backend_name == "spacy":
                model_name = self.backend_config.get("model", "ru_core_news_lg")
                return SpacyNERBackend(model_name)
                
            elif backend_name == "transformers":
                model_name = self.backend_config.get("model", "roberta_large_russian")
                return TransformersNERBackend(model_name)
                
            elif backend_name == "openai":
                model = self.backend_config.get("model", "gpt-4o-2024-05-13")
                api_key = self.backend_config.get("api_key")
                return OpenAINERBackend(model, api_key)
                
            elif backend_name == "rule_based":
                return RuleBasedNERBackend()
                
            else:
                self.logger.warning(f"Unknown backend: {backend_name}, falling back to rule-based")
                return RuleBasedNERBackend()
                
        except Exception as e:
            self.logger.error(f"Error loading backend {backend_name}: {e}")
            self.logger.info("Falling back to rule-based NER")
            return RuleBasedNERBackend()
    
    def extract_entities_from_text(self, text: str) -> List[Dict]:
        """Extract entities from text and return as dictionaries"""
        entities = self.backend.extract_entities(text)
        
        # Filter out false positives
        filtered_entities = []
        for entity in entities:
            if entity.text.lower() not in self.false_positive_words:
                filtered_entities.append(entity)
            else:
                self.logger.debug(f"Filtered out false positive: {entity.text}")
        
        # Convert to dictionary format
        return [
            {
                'text': entity.text,
                'label': entity.label,
                'confidence': entity.confidence,
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos,
                'context': entity.context,
                'source_line': entity.source_line
            }
            for entity in filtered_entities
        ]
    
    def extract_entities_from_lines(self, lines: List[Dict]) -> Dict:
        """Extract entities from text lines (main interface for pipeline)"""
        try:
            # Combine text from all lines
            all_text_lines = []
            full_text_parts = []
            
            for line in lines:
                text = line.get('text', '').strip()
                if text:
                    all_text_lines.append({'text': text, 'line_data': line})
                    full_text_parts.append(text)
            
            if not full_text_parts:
                return {
                    'entities': [],
                    'entities_by_type': {},
                    'total_entities': 0,
                    'backend': self.backend.get_backend_name()
                }
            
            # Extract entities from combined text
            full_text = ' '.join(full_text_parts)
            entities = self.backend.extract_entities(full_text)
            
            # Filter false positives
            filtered_entities = []
            for entity in entities:
                if entity.text.lower() not in self.false_positive_words:
                    filtered_entities.append(entity)
                else:
                    self.logger.debug(f"Filtered out false positive: {entity.text}")
            
            # Add context and source line information
            for entity in filtered_entities:
                entity.context = self.find_context(entity.text, all_text_lines)
                entity.source_line = self.find_source_line(entity.text, all_text_lines)
            
            # Group entities by type
            entities_by_type = {}
            for entity in filtered_entities:
                if entity.label not in entities_by_type:
                    entities_by_type[entity.label] = []
                entities_by_type[entity.label].append({
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'context': entity.context,
                    'source_line': entity.source_line
                })
            
            # Sort by confidence within each type
            for label in entities_by_type:
                entities_by_type[label].sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'entities': [
                    {
                        'text': entity.text,
                        'label': entity.label,
                        'confidence': entity.confidence,
                        'context': entity.context,
                        'source_line': entity.source_line
                    }
                    for entity in filtered_entities
                ],
                'entities_by_type': entities_by_type,
                'total_entities': len(filtered_entities),
                'backend': self.backend.get_backend_name(),
                'lines_processed': len(all_text_lines)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting entities from lines: {e}")
            return {
                'entities': [],
                'entities_by_type': {},
                'total_entities': 0,
                'backend': self.backend.get_backend_name(),
                'error': str(e)
            }
    
    def find_context(self, entity_text: str, text_lines: List[Dict], context_size: int = 2) -> str:
        """Find context around entity"""
        for i, line_data in enumerate(text_lines):
            if entity_text.lower() in line_data['text'].lower():
                start = max(0, i - context_size)
                end = min(len(text_lines), i + context_size + 1)
                context_lines = text_lines[start:end]
                return ' '.join(line['text'] for line in context_lines)
        return ""
        
    def find_source_line(self, entity_text: str, text_lines: List[Dict]) -> Optional[str]:
        """Find the specific line containing the entity"""
        for line_data in text_lines:
            if entity_text.lower() in line_data['text'].lower():
                return line_data['text']
        return None
    
    # Legacy methods for backward compatibility
    def _load_model(self):
        """Legacy method for backward compatibility"""
        pass
        
    def extract_entities(self, text: str) -> List[Dict]:
        """Legacy method - extract entities from single text"""
        return self.extract_entities_from_text(text)
        
    def _should_auto_install(self):
        """Legacy method for backward compatibility"""
        return False
        
    def _install_spacy(self):
        """Legacy method for backward compatibility"""
        return False
        
    def _extract_entities_rule_based(self, text: str) -> List[Dict]:
        """Legacy method for backward compatibility"""
        rule_backend = RuleBasedNERBackend()
        entities = rule_backend.extract_entities(text)
        return [
            {
                'text': entity.text,
                'label': entity.label,
                'confidence': entity.confidence,
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos
            }
            for entity in entities
        ]