#!/usr/bin/env python3
"""
Surname Matching Module for Ukrainian OCR Pipeline
Handles fuzzy matching of surnames in OCR output with Cyrillic support
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import unicodedata
from collections import Counter


@dataclass
class MatchResult:
    """Result of surname matching"""
    found_text: str
    matched_surname: str
    confidence: float
    position: Optional[Tuple[int, int]] = None  # (line_index, word_index)
    context: Optional[str] = None  # Surrounding text
    line_id: Optional[str] = None  # Line identifier from OCR
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class SurnameMatcher:
    """Fuzzy surname matching with Cyrillic support and OCR error tolerance"""
    
    def __init__(
        self,
        surname_file: Optional[str] = None,
        surnames: Optional[List[str]] = None,
        threshold: float = 0.8,
        use_phonetic: bool = True,
        min_length: int = 3
    ):
        """
        Initialize surname matcher
        
        Args:
            surname_file: Path to file with surnames (one per line)
            surnames: List of surnames to match against
            threshold: Minimum similarity threshold (0.0-1.0)
            use_phonetic: Whether to use phonetic matching
            min_length: Minimum surname length to consider
        """
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.use_phonetic = use_phonetic
        self.min_length = min_length
        self.surnames = set()
        self.surname_variants = {}  # Maps surnames to their variants
        
        # Load surnames
        if surname_file:
            self.load_surnames_from_file(surname_file)
        if surnames:
            self.add_surnames(surnames)
            
        # Precompute variants for loaded surnames
        self.prepare_surname_variants()
        
    def load_surnames_from_file(self, filepath: str):
        """Load surnames from text file (one per line)"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                surnames = [line.strip() for line in f if line.strip()]
                self.add_surnames(surnames)
                self.logger.info(f"Loaded {len(surnames)} surnames from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading surnames from {filepath}: {e}")
            
    def add_surnames(self, surnames: List[str]):
        """Add surnames to the matcher"""
        for surname in surnames:
            if len(surname) >= self.min_length:
                # Store original and normalized versions
                self.surnames.add(surname)
                normalized = self.normalize_text(surname)
                if normalized != surname and len(normalized) >= self.min_length:
                    self.surnames.add(normalized)
                
    def prepare_surname_variants(self):
        """Prepare common variants for each surname"""
        for surname in self.surnames:
            variants = self.generate_variants(surname)
            self.surname_variants[surname] = variants
            
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove accents and diacritics
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        
        # Remove punctuation but keep Ukrainian letters
        text = re.sub(r'[^\w\s\u0400-\u04FF]', '', text)
        
        return text.strip()
        
    def generate_variants(self, surname: str) -> Set[str]:
        """
        Generate common OCR error variants for surname
        
        Handles common OCR mistakes in Cyrillic:
        - Character confusion (н/и, п/л, о/а, etc.)
        - Case variations
        - Diacritic variations
        """
        variants = {surname, self.normalize_text(surname)}
        
        # Common Cyrillic OCR confusions
        cyrillic_confusions = [
            ('н', 'и'), ('п', 'л'), ('о', 'а'), ('е', 'є'),
            ('і', 'ї'), ('ь', 'ъ'), ('ш', 'щ'), ('ц', 'ч'),
            ('з', 'э'), ('б', 'в'), ('д', 'л'), ('г', 'т'),
            ('м', 'н'), ('у', 'ч'), ('ю', 'о'), ('я', 'а'),
            ('с', 'з'), ('к', 'х'), ('р', 'в'), ('ф', 'в')
        ]
        
        # Generate variants with single character substitutions
        for char1, char2 in cyrillic_confusions:
            if char1 in surname.lower():
                variant = surname.lower().replace(char1, char2)
                variants.add(variant)
                variants.add(variant.capitalize())
            if char2 in surname.lower():
                variant = surname.lower().replace(char2, char1)
                variants.add(variant)
                variants.add(variant.capitalize())
                
        # Add capitalization variants
        variants.add(surname.upper())
        variants.add(surname.lower())
        variants.add(surname.capitalize())
        
        # Add variants with common Ukrainian surname endings
        base = surname.lower()
        if base.endswith('енко'):
            variants.add(base.replace('енко', 'анко'))
        elif base.endswith('анко'):
            variants.add(base.replace('анко', 'енко'))
            
        return variants
        
    def phonetic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate phonetic similarity for Cyrillic text
        Simplified phonetic matching for Ukrainian/Russian surnames
        """
        # Phonetic groups for Cyrillic (sounds that are often confused)
        phonetic_groups = [
            ['б', 'п'], ['в', 'ф'], ['г', 'к', 'х'], ['д', 'т'],
            ['ж', 'ш', 'щ'], ['з', 'с', 'ц'], ['е', 'є', 'э'],
            ['і', 'ї', 'ы', 'и'], ['о', 'а'], ['у', 'ю'], ['я', 'а']
        ]
        
        # Create phonetic representation
        def to_phonetic(text):
            result = text.lower()
            for group in phonetic_groups:
                representative = group[0]
                for char in group[1:]:
                    result = result.replace(char, representative)
            return result
            
        phon1 = to_phonetic(text1)
        phon2 = to_phonetic(text2)
        
        return SequenceMatcher(None, phon1, phon2).ratio()
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings"""
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Direct comparison
        if norm1 == norm2:
            return 1.0
            
        # Length difference penalty
        len_diff = abs(len(norm1) - len(norm2))
        max_len = max(len(norm1), len(norm2))
        if max_len > 0:
            len_penalty = len_diff / max_len
        else:
            return 0.0
            
        # Character-level similarity
        char_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Apply length penalty
        char_similarity = char_similarity * (1 - len_penalty * 0.3)
        
        # Phonetic similarity if enabled
        if self.use_phonetic:
            phon_similarity = self.phonetic_similarity(text1, text2)
            # Weighted average favoring character similarity
            return 0.7 * char_similarity + 0.3 * phon_similarity
            
        return char_similarity
        
    def find_in_text(self, text: str, line_id: Optional[str] = None) -> List[MatchResult]:
        """
        Find all surname matches in text
        
        Args:
            text: Text to search in
            line_id: Optional line identifier
            
        Returns:
            List of match results
        """
        matches = []
        
        if not text:
            return matches
        
        # Split text into words (preserve Ukrainian characters)
        words = re.findall(r'\b[\w\u0400-\u04FF]+\b', text)
        
        for word_idx, word in enumerate(words):
            if len(word) < self.min_length:
                continue
                
            best_match = None
            best_similarity = 0.0
            
            # Check against all surnames
            for surname in self.surnames:
                similarity = self.calculate_similarity(word, surname)
                
                if similarity >= self.threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = surname
                    
            if best_match:
                # Get context (surrounding words)
                context_start = max(0, word_idx - 2)
                context_end = min(len(words), word_idx + 3)
                context = ' '.join(words[context_start:context_end])
                
                match = MatchResult(
                    found_text=word,
                    matched_surname=best_match,
                    confidence=best_similarity,
                    position=(0, word_idx),
                    context=context,
                    line_id=line_id
                )
                matches.append(match)
                
        return matches
        
    def find_in_lines(self, lines: List[Dict]) -> List[MatchResult]:
        """
        Find surnames in OCR line results
        
        Args:
            lines: List of line dictionaries with 'text' field
            
        Returns:
            List of match results with line positions
        """
        all_matches = []
        
        for line_idx, line in enumerate(lines):
            text = line.get('text', '')
            line_id = line.get('id', f'line_{line_idx}')
            
            if not text:
                continue
                
            # Find matches in this line
            line_matches = self.find_in_text(text, line_id)
            
            # Update position to include line index
            for match in line_matches:
                if match.position:
                    match.position = (line_idx, match.position[1])
                all_matches.append(match)
                
        return all_matches
        
    def find_in_document(self, document_data: Union[Dict, List[Dict]]) -> Dict:
        """
        Find surnames in complete document data
        
        Args:
            document_data: Document data (either dict with pages or list of lines)
            
        Returns:
            Dictionary with matches and statistics
        """
        results = {
            'total_matches': 0,
            'unique_surnames': set(),
            'matches': []
        }
        
        # Handle different input formats
        if isinstance(document_data, list):
            # Simple list of lines
            matches = self.find_in_lines(document_data)
            results['matches'] = matches
        elif isinstance(document_data, dict):
            # Check if it's a single page or multiple pages
            if 'lines' in document_data:
                # Single page format
                matches = self.find_in_lines(document_data['lines'])
                results['matches'] = matches
            else:
                # Multi-page format
                for page_id, page_data in document_data.items():
                    if isinstance(page_data, dict) and 'lines' in page_data:
                        matches = self.find_in_lines(page_data['lines'])
                        results['matches'].extend(matches)
        
        # Calculate statistics
        results['total_matches'] = len(results['matches'])
        results['unique_surnames'] = list(set(
            m.matched_surname for m in results['matches']
        ))
        
        return results
        
    def export_matches(self, matches: List[MatchResult], output_file: str):
        """Export matches to JSON file"""
        data = [match.to_dict() for match in matches]
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Exported {len(matches)} matches to {output_file}")
        except Exception as e:
            self.logger.error(f"Error exporting matches to {output_file}: {e}")
            
    def get_statistics(self, matches: List[MatchResult]) -> Dict:
        """Get detailed statistics about matches"""
        if not matches:
            return {
                'total_matches': 0,
                'unique_surnames': 0,
                'unique_found_texts': 0,
                'average_confidence': 0,
                'confidence_distribution': {},
                'top_surnames': [],
                'top_found_texts': []
            }
            
        confidences = [m.confidence for m in matches]
        unique_surnames = set(m.matched_surname for m in matches)
        unique_found = set(m.found_text for m in matches)
        
        # Confidence distribution
        conf_dist = {
            'excellent (≥0.95)': sum(1 for c in confidences if c >= 0.95),
            'high (0.9-0.94)': sum(1 for c in confidences if 0.9 <= c < 0.95),
            'good (0.8-0.89)': sum(1 for c in confidences if 0.8 <= c < 0.9),
            'acceptable (0.7-0.79)': sum(1 for c in confidences if 0.7 <= c < 0.8),
            'low (<0.7)': sum(1 for c in confidences if c < 0.7)
        }
        
        return {
            'total_matches': len(matches),
            'unique_surnames': len(unique_surnames),
            'unique_found_texts': len(unique_found),
            'average_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'confidence_distribution': conf_dist,
            'top_surnames': self.get_top_surnames(matches, 10),
            'top_found_texts': self.get_top_found_texts(matches, 10)
        }
        
    def get_top_surnames(self, matches: List[MatchResult], n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently matched surnames"""
        surname_counts = Counter(m.matched_surname for m in matches)
        return surname_counts.most_common(n)
        
    def get_top_found_texts(self, matches: List[MatchResult], n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently found text variants"""
        found_counts = Counter(m.found_text for m in matches)
        return found_counts.most_common(n)


def create_default_ukrainian_surnames() -> List[str]:
    """Create a default list of common Ukrainian surnames for testing"""
    return [
        # Most common Ukrainian surnames
        "Шевченко", "Коваленко", "Бондаренко", "Ткаченко", "Кравченко",
        "Олійник", "Шевчук", "Поліщук", "Мельник", "Гавриленко",
        "Петренко", "Іваненко", "Михайленко", "Василенко", "Григоренко",
        "Ковальчук", "Савченко", "Левченко", "Павленко", "Марченко",
        "Жук", "Козлов", "Мороз", "Кравець", "Швець",
        "Гончар", "Коваль", "Столяр", "Рибалко", "Терещенко",
        
        # Regional variations
        "Данилко", "Федорко", "Василько", "Петрук", "Іванко",
        "Романко", "Степанко", "Максимко", "Дмитрук", "Андрійко"
    ]


def test_surname_matcher():
    """Test surname matching functionality"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize with default surnames
    surnames = create_default_ukrainian_surnames()
    matcher = SurnameMatcher(surnames=surnames, threshold=0.8)
    
    # Test text with OCR errors
    test_text = "Шевченко Іван Петрович та Коваленко Марія Василівна були присутні. Бондаренко теж прийшов."
    
    print(f"Testing with {len(surnames)} surnames...")
    print(f"Test text: {test_text}")
    
    # Find matches
    matches = matcher.find_in_text(test_text)
    
    print(f"\nFound {len(matches)} matches:")
    for match in matches:
        print(f"  '{match.found_text}' -> '{match.matched_surname}' "
              f"(confidence: {match.confidence:.3f})")
        if match.context:
            print(f"    Context: {match.context}")
        
    # Get statistics
    if matches:
        stats = matcher.get_statistics(matches)
        print(f"\nStatistics:")
        print(f"  Total matches: {stats['total_matches']}")
        print(f"  Unique surnames: {stats['unique_surnames']}")
        print(f"  Average confidence: {stats['average_confidence']:.3f}")
        print(f"  Confidence distribution: {stats['confidence_distribution']}")


if __name__ == "__main__":
    test_surname_matcher()