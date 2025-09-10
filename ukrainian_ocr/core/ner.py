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
                try:
                    import spacy
                    # Try different Russian/Ukrainian models
                    model_candidates = [
                        self.model_name,
                        "ru_core_news_lg", 
                        "ru_core_news_md", 
                        "ru_core_news_sm",
                        "uk_core_news_sm"
                    ]
                    
                    # Remove None values
                    model_candidates = [m for m in model_candidates if m]
                    
                    for model_name in model_candidates:
                        try:
                            self.nlp = spacy.load(model_name)
                            self.logger.info(f"Loaded spaCy model: {model_name}")
                            return
                        except OSError:
                            continue
                    
                    self.logger.warning("No spaCy models found, using rule-based extraction")
                    self.nlp = None
                    
                except ImportError:
                    # Try to auto-install spaCy for better results
                    if self._should_auto_install():
                        self.logger.info("Attempting to install spaCy for better NER results...")
                        if self._install_spacy():
                            # Retry loading after installation
                            return self._load_model()
                    
                    self.logger.warning("spaCy not installed, using rule-based extraction")
                    self.nlp = None
                    
            elif self.backend == "transformers":
                try:
                    from transformers import pipeline
                    model_name = self.model_name or "dbmdz/bert-base-multilingual-cased-ner"
                    self.nlp = pipeline("ner", model=model_name, tokenizer=model_name)
                    self.logger.info(f"Loaded Transformers model: {model_name}")
                except ImportError:
                    self.logger.warning("Transformers not installed, using rule-based extraction")
                    self.nlp = None
                except Exception as e:
                    self.logger.warning(f"Transformers model loading failed: {e}, using rule-based extraction")
                    self.nlp = None
            else:
                self.logger.warning(f"Unknown NER backend '{self.backend}', using rule-based extraction")
                self.nlp = None
                
        except Exception as e:
            self.logger.warning(f"NER model loading failed: {e}, using rule-based extraction")
            self.nlp = None
            
        if self.nlp is None:
            self.logger.info("Using rule-based NER extraction (install spaCy for better results)")
    
    def _should_auto_install(self) -> bool:
        """Check if we should attempt auto-installation"""
        # Only auto-install in known environments (like Colab or Jupyter)
        try:
            # Check for Colab
            import google.colab
            return True
        except ImportError:
            pass
        
        # Check for Jupyter
        try:
            import IPython
            return True
        except ImportError:
            pass
        
        # Don't auto-install in production environments
        return False
    
    def _install_spacy(self) -> bool:
        """Attempt to install spaCy and a Russian model"""
        try:
            import subprocess
            import sys
            
            # Install spaCy
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'spacy', '--quiet'
            ])
            
            # Try to install a Russian model
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'spacy', 'download', 'ru_core_news_sm', '--quiet'
                ])
                self.logger.info("Successfully installed spaCy with ru_core_news_sm model")
            except subprocess.CalledProcessError:
                self.logger.warning("spaCy installed but model download failed")
                
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to install spaCy: {e}")
            return False
    
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
        """Rule-based entity extraction for Ukrainian/Russian text"""
        
        entities = []
        
        # Enhanced Ukrainian/Russian name patterns
        name_patterns = [
            # Full names with patronymic
            r'\b[А-ЯІЇЄЁA-Z][а-яіїєёa-z]+\s+[А-ЯІЇЄЁA-Z][а-яіїєёa-z]+(?:ич|енко|ський|цький|ук|юк|ко|ова|іна|евич|ович|івич)\b',
            # Last names with typical endings
            r'\b[А-ЯІЇЄЁA-Z][а-яіїєёa-z]+(?:енко|ський|цький|ук|юк|ко|ова|іна|евич|ович|івич|ев|ева|ин|ина|ський|цька)\b',
            # First + Last name combinations
            r'\b(?:Іван|Петро|Микола|Олександр|Василь|Андрій|Михайло|Дмитро|Сергій|Володимир|Анна|Марія|Катерина|Ольга|Наталія|Тетяна|Людмила|Ірина)\s+[А-ЯІЇЄЁA-Z][а-яіїєёa-z]+\b',
        ]
        
        # Enhanced location patterns
        place_patterns = [
            # Administrative units
            r'\b(?:село|с\.|місто|м\.|район|р-н|область|обл\.|губернія|губ\.)\s+[А-ЯІЇЄЁA-Z][а-яіїєёa-z-]+\b',
            # Cities with typical endings
            r'\b[А-ЯІЇЄЁA-Z][а-яіїєёa-z]+(?:ськ|ський|град|город|ів|ово|ево|ине|енки|ичі|ці|ка|ки)\b',
            # Major Ukrainian cities
            r'\b(?:Харків|Київ|Львів|Одеса|Дніпро|Запоріжжя|Вінниця|Чернігів|Суми|Полтава|Черкаси|Житомир|Рівне|Івано-Франківськ|Тернопіль|Луцьк|Ужгород|Чернівці|Кропивницький|Мелітополь|Кременчук|Білгород|Курськ|Воронеж|Москва|Петербург)\b',
            # Historical regions
            r'\b(?:Слобожанщина|Галичина|Волинь|Поділля|Буковина|Закарпаття)\b'
        ]
        
        # Date patterns
        date_patterns = [
            r'\b(?:1[8-9]\d{2}|20[0-2]\d)\s*(?:року|года|р\.)\b',
            r'\b\d{1,2}\s*(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\s*(?:1[8-9]\d{2}|20[0-2]\d)\b'
        ]
        
        # Extract person names
        for pattern in name_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Skip if it's likely a place name
                matched_text = match.group()
                if not any(place_word in matched_text.lower() for place_word in ['ськ', 'град', 'город']):
                    entities.append({
                        'text': matched_text,
                        'label': 'PERSON',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.85
                    })
        
        # Extract locations
        for pattern in place_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': 'LOCATION',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.80
                })
        
        # Extract dates
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': 'DATE',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.90
                })
        
        # Remove overlapping entities (keep higher confidence)
        entities = self._remove_overlapping_entities(entities)
                
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping higher confidence ones"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            # Check if this entity overlaps with any in filtered list
            overlaps = False
            for filtered_entity in filtered:
                if (entity['start'] < filtered_entity['end'] and 
                    entity['end'] > filtered_entity['start']):
                    # Keep the one with higher confidence
                    if entity['confidence'] > filtered_entity['confidence']:
                        filtered.remove(filtered_entity)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
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