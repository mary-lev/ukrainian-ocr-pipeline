"""
NER (Named Entity Recognition) module for Ukrainian OCR Pipeline
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Union
from pathlib import Path
import re

class NERExtractor:
    """Named Entity Recognition for Ukrainian text"""
    
    def __init__(self, backend: str = "spacy", model_name: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.backend = backend
        self.model_name = model_name
        self.nlp = None
        self._load_model()
        
    def _load_model(self):
        """Load NER model based on backend"""
        try:
            if self.backend == "spacy":
                import spacy
                model_name = self.model_name or "uk_core_news_sm"
                try:
                    self.nlp = spacy.load(model_name)
                except OSError:
                    self.logger.warning(f"Model {model_name} not found, using placeholder")
                    self.nlp = None
            elif self.backend == "transformers":
                from transformers import pipeline
                model_name = self.model_name or "dbmdz/bert-base-multilingual-cased-ner"
                try:
                    self.nlp = pipeline("ner", model=model_name, tokenizer=model_name)
                except Exception:
                    self.logger.warning(f"Transformers model {model_name} not found, using placeholder")
                    self.nlp = None
            else:
                self.logger.warning(f"Unknown backend {self.backend}, using placeholder")
                self.nlp = None
                
        except Exception as e:
            self.logger.error(f"Error loading NER model: {e}")
            self.nlp = None
    
    def extract_entities(self, text: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for entities
            
        Returns:
            List of entity dictionaries
        """
        if not self.nlp:
            return self._placeholder_extraction(text)
            
        try:
            if self.backend == "spacy":
                return self._extract_spacy(text, confidence_threshold)
            elif self.backend == "transformers":
                return self._extract_transformers(text, confidence_threshold)
            else:
                return self._placeholder_extraction(text)
                
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return self._placeholder_extraction(text)
    
    def _extract_spacy(self, text: str, confidence_threshold: float) -> List[Dict]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            confidence = getattr(ent, 'confidence', 0.9)
            if confidence >= confidence_threshold:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': confidence
                })
                
        return entities
    
    def _extract_transformers(self, text: str, confidence_threshold: float) -> List[Dict]:
        """Extract entities using Transformers"""
        results = self.nlp(text)
        entities = []
        
        for result in results:
            if result['score'] >= confidence_threshold:
                entities.append({
                    'text': result['word'],
                    'label': result['entity'],
                    'start': result['start'],
                    'end': result['end'],
                    'confidence': result['score']
                })
                
        return entities
    
    def _placeholder_extraction(self, text: str) -> List[Dict]:
        """Placeholder entity extraction for testing"""
        
        # Simple rule-based extraction for Ukrainian names and places
        entities = []
        
        # Common Ukrainian name patterns
        name_patterns = [
            r'\b[А-ЯІЇЄЁ][а-яіїєё]+\s+[А-ЯІЇЄЁ][а-яіїєё]+(?:ич|енко|ський|цький|ук|юк|ко)\b',
            r'\b[А-ЯІЇЄЁ][а-яіїєё]+(?:енко|ський|цький|ук|юк|ко)\b',
        ]
        
        # Common location patterns
        place_patterns = [
            r'\b(?:село|місто|район|область|губернія)\s+[А-ЯІЇЄЁ][а-яіїєё]+\b',
            r'\b[А-ЯІЇЄЁ][а-яіїєё]+(?:ськ|ський|град|город|ів|ово|ево|ине|енки)\b',
            r'\bХарків|Київ|Львів|Одеса|Дніпро|Запоріжжя\b'
        ]
        
        # Extract person names
        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': 'PERSON',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        # Extract locations
        for pattern in place_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': 'LOCATION',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7
                })
                
        return entities
    
    def extract_from_alto(self, alto_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Extract entities from ALTO XML file
        
        Args:
            alto_path: Path to ALTO file
            confidence_threshold: Minimum confidence for entities
            
        Returns:
            Dictionary with line-level entity annotations
        """
        try:
            tree = ET.parse(alto_path)
            root = tree.getroot()
            
            # Find all text lines
            lines_with_entities = {}
            
            for text_line in root.findall('.//TextLine'):
                line_id = text_line.get('ID')
                
                # Extract text from String elements
                strings = text_line.findall('.//String')
                line_text = ' '.join([s.get('CONTENT', '') for s in strings])
                
                if line_text.strip():
                    # Extract entities from line text
                    entities = self.extract_entities(line_text, confidence_threshold)
                    
                    if entities:
                        lines_with_entities[line_id] = {
                            'text': line_text,
                            'entities': entities
                        }
            
            return lines_with_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities from ALTO: {e}")
            return {}