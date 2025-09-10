"""
ALTO XML enhancement module for Ukrainian OCR Pipeline
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

class ALTOEnhancer:
    """ALTO XML enhancement with NER annotations and person regions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def enhance_alto_with_ner(
        self, 
        alto_path: str, 
        entities_by_line: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Enhance ALTO XML file with NER annotations
        
        Args:
            alto_path: Path to input ALTO file
            entities_by_line: Dictionary mapping line IDs to entities
            output_path: Path for enhanced ALTO file
            
        Returns:
            Path to enhanced ALTO file
        """
        try:
            # Parse ALTO file
            tree = ET.parse(alto_path)
            root = tree.getroot()
            
            # Get namespace
            namespace = self._get_namespace(root)
            
            # Add NER tags to Tags section
            self._add_ner_tags(root, namespace, entities_by_line)
            
            # Annotate text lines with entity information
            self._annotate_text_lines(root, namespace, entities_by_line)
            
            # Add person-dense regions
            person_regions = self._find_person_dense_regions(root, namespace, entities_by_line)
            if person_regions:
                self._add_person_regions_to_alto(root, namespace, person_regions)
            
            # Save enhanced ALTO
            if not output_path:
                output_path = str(Path(alto_path).with_suffix('.enhanced.xml'))
                
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            self.logger.info(f"Enhanced ALTO saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error enhancing ALTO: {e}")
            return alto_path
    
    def _get_namespace(self, root: ET.Element) -> str:
        """Extract namespace from ALTO root element"""
        tag = root.tag
        if '}' in tag:
            return tag[tag.find('{')+1:tag.find('}')]
        return ''
    
    def _add_ner_tags(self, root: ET.Element, namespace: str, entities_by_line: Dict):
        """Add NER-related tags to ALTO Tags section"""
        
        # Find or create Tags element
        if namespace:
            tags_elem = root.find(f'.//{{{namespace}}}Tags')
            layout_elem = root.find(f'.//{{{namespace}}}Layout')
        else:
            tags_elem = root.find('.//Tags')
            layout_elem = root.find('.//Layout')
        
        if tags_elem is None and layout_elem is not None:
            # Create Tags element
            if namespace:
                tags_elem = ET.Element(f'{{{namespace}}}Tags')
            else:
                tags_elem = ET.Element('Tags')
            layout_elem.insert(0, tags_elem)
        
        if tags_elem is not None:
            # Collect unique entity types
            entity_types = set()
            for line_entities in entities_by_line.values():
                for entity in line_entities.get('entities', []):
                    entity_types.add(entity['label'])
            
            # Add OtherTag elements for each entity type
            tag_counter = 11  # Start from LT11
            
            for entity_type in sorted(entity_types):
                tag_id = f"LT{tag_counter}"
                
                if namespace:
                    other_tag = ET.SubElement(tags_elem, f'{{{namespace}}}OtherTag')
                else:
                    other_tag = ET.SubElement(tags_elem, 'OtherTag')
                
                other_tag.set('ID', tag_id)
                other_tag.set('LABEL', entity_type.lower())
                other_tag.set('DESCRIPTION', f'NER_{entity_type}')
                
                tag_counter += 1
            
            # Add person-dense region tag
            person_tag_id = f"LT{tag_counter}"
            if namespace:
                person_tag = ET.SubElement(tags_elem, f'{{{namespace}}}OtherTag')
            else:
                person_tag = ET.SubElement(tags_elem, 'OtherTag')
                
            person_tag.set('ID', person_tag_id)
            person_tag.set('LABEL', 'person_dense_region')
            person_tag.set('DESCRIPTION', 'Region with high concentration of person names')
    
    def _annotate_text_lines(self, root: ET.Element, namespace: str, entities_by_line: Dict):
        """Annotate text lines with entity TAGREFS"""
        
        # Create mapping from entity type to tag ID
        entity_to_tag = {}
        if namespace:
            other_tags = root.findall(f'.//{{{namespace}}}OtherTag')
        else:
            other_tags = root.findall('.//OtherTag')
            
        for tag in other_tags:
            description = tag.get('DESCRIPTION', '')
            if description.startswith('NER_'):
                entity_type = description[4:]  # Remove 'NER_' prefix
                entity_to_tag[entity_type] = tag.get('ID')
        
        # Annotate text lines
        if namespace:
            text_lines = root.findall(f'.//{{{namespace}}}TextLine')
        else:
            text_lines = root.findall('.//TextLine')
            
        for text_line in text_lines:
            line_id = text_line.get('ID')
            
            if line_id in entities_by_line:
                line_data = entities_by_line[line_id]
                entities = line_data.get('entities', [])
                
                # Collect entity types for this line
                line_entity_types = set()
                for entity in entities:
                    entity_type = entity['label']
                    if entity_type in entity_to_tag:
                        line_entity_types.add(entity_type)
                
                # Add TAGREFS and ENTITY_TYPES attributes
                if line_entity_types:
                    tag_refs = ' '.join([entity_to_tag[et] for et in sorted(line_entity_types)])
                    text_line.set('TAGREFS', tag_refs)
                    text_line.set('ENTITY_TYPES', ' '.join(sorted(line_entity_types)))
    
    def _find_person_dense_regions(
        self, 
        root: ET.Element, 
        namespace: str, 
        entities_by_line: Dict
    ) -> List[Dict]:
        """Find regions with high concentration of person names"""
        
        # Extract person lines with coordinates
        person_lines = []
        
        if namespace:
            text_lines = root.findall(f'.//{{{namespace}}}TextLine')
        else:
            text_lines = root.findall('.//TextLine')
            
        for text_line in text_lines:
            line_id = text_line.get('ID')
            
            if line_id in entities_by_line:
                entities = entities_by_line[line_id].get('entities', [])
                has_person = any(e['label'] == 'PERSON' for e in entities)
                
                if has_person:
                    # Get line coordinates from COORDS or approximate from HEIGHT/VPOS
                    coords_attr = text_line.get('COORDS')
                    if coords_attr:
                        # Parse COORDS points
                        coords = self._parse_coords(coords_attr)
                        if coords:
                            y_center = sum(point[1] for point in coords) / len(coords)
                            person_lines.append({
                                'line_id': line_id,
                                'y_center': y_center,
                                'element': text_line
                            })
                    else:
                        # Use HEIGHT/VPOS attributes
                        vpos = text_line.get('VPOS')
                        height = text_line.get('HEIGHT')
                        if vpos and height:
                            y_center = int(vpos) + int(height) // 2
                            person_lines.append({
                                'line_id': line_id,
                                'y_center': y_center,
                                'element': text_line
                            })
        
        if len(person_lines) < 3:
            return []
        
        # Perform DBSCAN clustering on y-coordinates
        coordinates = np.array([[line['y_center']] for line in person_lines])
        
        clustering = DBSCAN(eps=300, min_samples=3).fit(coordinates)
        labels = clustering.labels_
        
        # Find the largest cluster
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise points
            
        if not unique_labels:
            return []
        
        cluster_sizes = {label: sum(1 for l in labels if l == label) for label in unique_labels}
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        
        # Get lines in the largest cluster
        cluster_lines = [person_lines[i] for i, label in enumerate(labels) if label == largest_cluster]
        
        # Extend cluster with nearby lines
        cluster_y_coords = [line['y_center'] for line in cluster_lines]
        min_y, max_y = min(cluster_y_coords), max(cluster_y_coords)
        
        # Find all text lines in the extended region
        if namespace:
            all_text_lines = root.findall(f'.//{{{namespace}}}TextLine')
        else:
            all_text_lines = root.findall('.//TextLine')
            
        extended_cluster = []
        for text_line in all_text_lines:
            # Get y-coordinate
            y_coord = None
            coords_attr = text_line.get('COORDS')
            if coords_attr:
                coords = self._parse_coords(coords_attr)
                if coords:
                    y_coord = sum(point[1] for point in coords) / len(coords)
            else:
                vpos = text_line.get('VPOS')
                height = text_line.get('HEIGHT')
                if vpos and height:
                    y_coord = int(vpos) + int(height) // 2
            
            # Check if line is in extended region
            if y_coord and min_y - 100 <= y_coord <= max_y + 100:
                extended_cluster.append({
                    'line_id': text_line.get('ID'),
                    'y_center': y_coord,
                    'element': text_line
                })
        
        if len(extended_cluster) >= 5:
            return [{
                'lines': extended_cluster,
                'y_min': min(line['y_center'] for line in extended_cluster) - 50,
                'y_max': max(line['y_center'] for line in extended_cluster) + 50
            }]
        
        return []
    
    def _add_person_regions_to_alto(
        self, 
        root: ET.Element, 
        namespace: str, 
        person_regions: List[Dict]
    ):
        """Add person-dense regions as TextBlocks to ALTO"""
        
        # Find person-dense region tag ID
        person_tag_id = None
        if namespace:
            other_tags = root.findall(f'.//{{{namespace}}}OtherTag')
        else:
            other_tags = root.findall('.//OtherTag')
            
        for tag in other_tags:
            if tag.get('LABEL') == 'person_dense_region':
                person_tag_id = tag.get('ID')
                break
        
        if not person_tag_id:
            return
        
        # Find PrintSpace element
        if namespace:
            print_space = root.find(f'.//{{{namespace}}}PrintSpace')
        else:
            print_space = root.find('.//PrintSpace')
            
        if print_space is None:
            return
        
        # Add TextBlock for each person-dense region
        for i, region in enumerate(person_regions):
            if namespace:
                text_block = ET.Element(f'{{{namespace}}}TextBlock')
            else:
                text_block = ET.Element('TextBlock')
            
            text_block.set('ID', f'person_dense_region_block_{i}')
            text_block.set('TAGREFS', person_tag_id)
            text_block.set('PERSON_LINES_COUNT', str(len(region['lines'])))
            
            # Calculate block coordinates
            x_coords = []
            for line in region['lines']:
                element = line['element']
                coords_attr = element.get('COORDS')
                if coords_attr:
                    coords = self._parse_coords(coords_attr)
                    if coords:
                        x_coords.extend([point[0] for point in coords])
                else:
                    hpos = element.get('HPOS')
                    width = element.get('WIDTH')
                    if hpos and width:
                        x_coords.extend([int(hpos), int(hpos) + int(width)])
            
            if x_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                text_block.set('HPOS', str(max(0, x_min - 50)))
                text_block.set('VPOS', str(max(0, int(region['y_min']))))
                text_block.set('WIDTH', str(x_max - x_min + 100))
                text_block.set('HEIGHT', str(int(region['y_max'] - region['y_min'])))
            
            # Insert at the beginning of PrintSpace
            print_space.insert(0, text_block)
    
    def _parse_coords(self, coords_str: str) -> List[tuple]:
        """Parse COORDS attribute into list of (x, y) tuples"""
        try:
            coords = []
            parts = coords_str.split()
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    x, y = int(parts[i]), int(parts[i + 1])
                    coords.append((x, y))
            return coords
        except (ValueError, IndexError):
            return []