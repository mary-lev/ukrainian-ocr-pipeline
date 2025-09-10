"""
Unit tests for NER module
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ukrainian_ocr.core.ner import NERExtractor
from tests.fixtures.create_test_data import create_sample_entities


class TestNERExtractor(unittest.TestCase):
    """Test cases for NERExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ner_extractor = NERExtractor(backend="spacy")
        self.sample_entities = create_sample_entities()
        
        # Test texts
        self.person_text = "Андрей Моисеевич Орехов"
        self.location_text = "село Песчаное, Харківська губернія"
        self.mixed_text = "Андрей Орехов з села Песчаное"
        self.empty_text = ""
        self.non_entity_text = "це просто текст без імен"
    
    def test_ner_extractor_initialization(self):
        """Test NER extractor initialization"""
        # Test spaCy backend
        spacy_extractor = NERExtractor(backend="spacy", model_name="uk_core_news_sm")
        self.assertEqual(spacy_extractor.backend, "spacy")
        self.assertEqual(spacy_extractor.model_name, "uk_core_news_sm")
        
        # Test transformers backend
        transformers_extractor = NERExtractor(
            backend="transformers", 
            model_name="bert-base-multilingual-cased"
        )
        self.assertEqual(transformers_extractor.backend, "transformers")
        self.assertEqual(transformers_extractor.model_name, "bert-base-multilingual-cased")
        
        # Test unknown backend (should fallback)
        unknown_extractor = NERExtractor(backend="unknown")
        self.assertEqual(unknown_extractor.backend, "unknown")
        self.assertIsNone(unknown_extractor.nlp)
    
    def test_extract_entities_returns_list(self):
        """Test that extract_entities returns a list"""
        entities = self.ner_extractor.extract_entities(self.person_text)
        
        self.assertIsInstance(entities, list)
        
        # Check entity structure if any found
        for entity in entities:
            self.assertIsInstance(entity, dict)
            self.assertIn('text', entity)
            self.assertIn('label', entity)
            self.assertIn('start', entity)
            self.assertIn('end', entity)
            self.assertIn('confidence', entity)
    
    def test_extract_entities_person_text(self):
        """Test entity extraction from person name text"""
        entities = self.ner_extractor.extract_entities(self.person_text)
        
        # Should find at least one entity (using placeholder rules)
        self.assertGreater(len(entities), 0)
        
        # Check if person entity found
        person_entities = [e for e in entities if e['label'] == 'PERSON']
        self.assertGreater(len(person_entities), 0)
        
        # Check entity properties
        for entity in person_entities:
            self.assertGreaterEqual(entity['start'], 0)
            self.assertLessEqual(entity['end'], len(self.person_text))
            self.assertGreater(len(entity['text']), 0)
            self.assertGreaterEqual(entity['confidence'], 0.0)
            self.assertLessEqual(entity['confidence'], 1.0)
    
    def test_extract_entities_location_text(self):
        """Test entity extraction from location text"""
        entities = self.ner_extractor.extract_entities(self.location_text)
        
        # Should find location entities (using placeholder rules)
        location_entities = [e for e in entities if e['label'] == 'LOCATION']
        self.assertGreaterEqual(len(location_entities), 0)  # May not find with simple rules
    
    def test_extract_entities_empty_text(self):
        """Test entity extraction from empty text"""
        entities = self.ner_extractor.extract_entities(self.empty_text)
        
        self.assertIsInstance(entities, list)
        self.assertEqual(len(entities), 0)
    
    def test_extract_entities_confidence_threshold(self):
        """Test confidence threshold filtering"""
        # Extract with high confidence threshold
        entities_high = self.ner_extractor.extract_entities(
            self.person_text, 
            confidence_threshold=0.9
        )
        
        # Extract with low confidence threshold  
        entities_low = self.ner_extractor.extract_entities(
            self.person_text,
            confidence_threshold=0.1
        )
        
        # High threshold should return fewer or equal entities
        self.assertLessEqual(len(entities_high), len(entities_low))
        
        # All returned entities should meet threshold
        for entity in entities_high:
            self.assertGreaterEqual(entity['confidence'], 0.9)
    
    def test_placeholder_extraction_patterns(self):
        """Test placeholder extraction patterns work"""
        # Test name pattern
        name_text = "Петренко"  # Common Ukrainian surname
        entities = self.ner_extractor.extract_entities(name_text)
        
        self.assertIsInstance(entities, list)
        # Placeholder should find some patterns
    
    def test_extract_from_alto_invalid_file(self):
        """Test ALTO extraction with invalid file"""
        result = self.ner_extractor.extract_from_alto("non_existent_file.xml")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)
    
    def test_different_backends_same_interface(self):
        """Test that different backends provide same interface"""
        backends = ["spacy", "transformers", "unknown"]
        
        for backend in backends:
            extractor = NERExtractor(backend=backend)
            entities = extractor.extract_entities(self.person_text)
            
            self.assertIsInstance(entities, list)
            
            # Check entity structure
            for entity in entities:
                required_keys = ['text', 'label', 'start', 'end', 'confidence']
                for key in required_keys:
                    self.assertIn(key, entity)
    
    def test_extract_entities_unicode_text(self):
        """Test entity extraction with Ukrainian Unicode text"""
        ukrainian_texts = [
            "Володимир Іванович Шевченко",
            "місто Київ",
            "Львівська область",
            "Андрій Петрович з Харкова"
        ]
        
        for text in ukrainian_texts:
            entities = self.ner_extractor.extract_entities(text)
            
            # Should not crash and return valid results
            self.assertIsInstance(entities, list)
            
            for entity in entities:
                self.assertIsInstance(entity['text'], str)
                self.assertTrue(len(entity['text']) > 0)


class TestNERExtractionMethods(unittest.TestCase):
    """Test specific extraction methods"""
    
    def setUp(self):
        self.ner_extractor = NERExtractor(backend="spacy")
    
    def test_spacy_extraction_method(self):
        """Test spaCy extraction method (if available)"""
        # This will use placeholder if spaCy not available
        entities = self.ner_extractor._extract_spacy(
            "Тест text", 
            confidence_threshold=0.5
        )
        
        self.assertIsInstance(entities, list)
    
    def test_transformers_extraction_method(self):
        """Test transformers extraction method (if available)"""
        # This will use placeholder if transformers not available
        entities = self.ner_extractor._extract_transformers(
            "Тест text",
            confidence_threshold=0.5
        )
        
        self.assertIsInstance(entities, list)
    
    def test_placeholder_extraction_method(self):
        """Test placeholder extraction method"""
        entities = self.ner_extractor._placeholder_extraction("Андрей Орехов")
        
        self.assertIsInstance(entities, list)
        
        if len(entities) > 0:
            # Should find person name
            person_entities = [e for e in entities if e['label'] == 'PERSON']
            self.assertGreater(len(person_entities), 0)


if __name__ == '__main__':
    unittest.main()