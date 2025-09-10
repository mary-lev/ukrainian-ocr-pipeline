"""
ALTO XML enhancement module for Ukrainian OCR Pipeline
Advanced NER-based enhancement with proper ALTO v4 compliance
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set
from pathlib import Path
import numpy as np

class ALTOEnhancer:
    """ALTO XML enhancement with sophisticated NER annotations and person regions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ner_tag_mapping = {}  # Maps entity descriptions to tag IDs
        
    def enhance_alto_with_ner(
        self, 
        alto_path: str, 
        entities_by_line: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Enhance ALTO XML file with sophisticated NER annotations
        
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
            
            # Register namespace to preserve ALTO format
            ET.register_namespace('', 'http://www.loc.gov/standards/alto/ns-v4#')
            
            # Add sophisticated NER tags
            self._add_entity_tags(root, entities_by_line)
            
            # Label text lines with proper TAGREFS and attributes
            self._label_text_lines(root, entities_by_line)
            
            # Create person-dense regions
            self._create_person_dense_blocks(root)
            
            # Save enhanced ALTO
            if not output_path:
                output_path = str(Path(alto_path).with_suffix('.enhanced.xml'))
                
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            self.logger.info(f"Enhanced ALTO saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error enhancing ALTO: {e}")
            return alto_path
    
    def _add_entity_tags(self, root: ET.Element, entities_by_line: Dict):
        """Add sophisticated Tag elements for each entity type found"""
        
        # Find or create Tags section
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        tags_elem = root.find('.//alto:Tags', ns)
        
        if tags_elem is None:
            tags_elem = root.find('.//Tags')
            
        if tags_elem is None:
            # Create Tags section if it doesn't exist
            # Find description element to insert after it
            desc_elem = root.find('.//alto:Description', ns) or root.find('.//Description')
            if desc_elem is not None:
                tags_elem = ET.Element('Tags')
                desc_elem.getparent().insert(desc_elem.getparent().index(desc_elem) + 1, tags_elem)
            else:
                # Insert at beginning
                tags_elem = ET.SubElement(root, 'Tags')
        
        # Find highest existing LT and BT numbers
        existing_lts = []
        existing_bts = []
        
        for tag in tags_elem.findall('.//alto:OtherTag', ns) + tags_elem.findall('.//OtherTag'):
            tag_id = tag.get('ID', '')
            if tag_id.startswith('LT'):
                try:
                    num = int(tag_id[2:])
                    existing_lts.append(num)
                except ValueError:
                    pass
            elif tag_id.startswith('BT'):
                try:
                    num = int(tag_id[2:])
                    existing_bts.append(num)
                except ValueError:
                    pass
                    
        next_lt_num = max(existing_lts) + 1 if existing_lts else 10
        next_bt_num = max(existing_bts) + 1 if existing_bts else 10
        
        # NER line type tags
        ner_tags = [
            ('person', 'NER_PERSON'),
            ('location', 'NER_LOCATION'), 
            ('organization', 'NER_ORGANIZATION'),
            ('mixed_entities', 'NER_MIXED'),
            ('has_entity', 'NER_ENTITY')
        ]
        
        # Block type for person-dense regions
        block_tags = [
            ('person_dense_region', 'NER_PERSON_DENSE_BLOCK')
        ]
        
        # Add line type tags
        for label, description in ner_tags:
            # Check if this NER tag already exists
            existing = None
            for tag in tags_elem.findall('.//alto:OtherTag', ns) + tags_elem.findall('.//OtherTag'):
                if tag.get('DESCRIPTION') == description:
                    existing = tag
                    break
            
            if existing is None:
                tag_id = f'LT{next_lt_num}'
                tag = ET.SubElement(tags_elem, 'OtherTag')
                tag.set('ID', tag_id)
                tag.set('LABEL', label)
                tag.set('DESCRIPTION', description)
                self.ner_tag_mapping[description] = tag_id
                next_lt_num += 1
            else:
                self.ner_tag_mapping[description] = existing.get('ID')
                
        # Add block type tags
        for label, description in block_tags:
            existing = None
            for tag in tags_elem.findall('.//alto:OtherTag', ns) + tags_elem.findall('.//OtherTag'):
                if tag.get('DESCRIPTION') == description:
                    existing = tag
                    break
                    
            if existing is None:
                tag_id = f'BT{next_bt_num}'
                tag = ET.SubElement(tags_elem, 'OtherTag')
                tag.set('ID', tag_id)
                tag.set('LABEL', label)
                tag.set('DESCRIPTION', description)
                self.ner_tag_mapping[description] = tag_id
                next_bt_num += 1
            else:
                self.ner_tag_mapping[description] = existing.get('ID')
    
    def _label_text_lines(self, root: ET.Element, entities_by_line: Dict):
        """Label TextLine elements with sophisticated NER annotations"""
        
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Process each TextLine in ALTO
        for text_line in root.findall('.//alto:TextLine', ns) + root.findall('.//TextLine'):
            line_id = text_line.get('ID', '')
            
            # Check if this line has entities
            if line_id in entities_by_line:
                line_data = entities_by_line[line_id]
                entities = line_data.get('entities', [])
                
                if entities:
                    # Determine entity types in this line
                    entity_types = set()
                    entity_texts = []
                    confidences = []
                    
                    for entity in entities:
                        label = entity.get('label', '').upper()
                        text = entity.get('text', '')
                        confidence = entity.get('confidence', 0.0)
                        
                        # Map labels to our categories
                        if label in ['PER', 'PERSON']:
                            entity_types.add('PERSON')
                        elif label in ['LOC', 'LOCATION']:
                            entity_types.add('LOCATION') 
                        elif label in ['ORG', 'ORGANIZATION']:
                            entity_types.add('ORGANIZATION')
                        else:
                            entity_types.add(label)
                            
                        entity_texts.append(text)
                        confidences.append(confidence)
                    
                    if entity_types:
                        # Determine which NER tag to use
                        if len(entity_types) > 1:
                            ner_description = 'NER_MIXED'
                        elif 'PERSON' in entity_types:
                            ner_description = 'NER_PERSON'
                        elif 'LOCATION' in entity_types:
                            ner_description = 'NER_LOCATION'
                        elif 'ORGANIZATION' in entity_types:
                            ner_description = 'NER_ORGANIZATION'
                        else:
                            ner_description = 'NER_ENTITY'
                        
                        # Get the LT ID for this NER type
                        lt_id = self.ner_tag_mapping.get(ner_description, 'LT1')
                        
                        # Set TAGREFS to the appropriate LT ID
                        existing_tagrefs = text_line.get('TAGREFS', '').split()
                        if lt_id not in existing_tagrefs:
                            existing_tagrefs.append(lt_id)
                        text_line.set('TAGREFS', ' '.join(existing_tagrefs))
                        
                        # Add comprehensive custom attributes
                        text_line.set('ENTITY_TYPES', '|'.join(sorted(entity_types)))
                        text_line.set('ENTITY_TEXTS', '|'.join(entity_texts[:5]))  # Limit to 5
                        text_line.set('ENTITY_COUNT', str(len(entities)))
                        
                        # Add processing confidence
                        if confidences:
                            avg_confidence = sum(confidences) / len(confidences)
                            text_line.set('NER_CONFIDENCE', f"{avg_confidence:.2f}")
                        
                        # Add STYLEREFS for visual distinction
                        text_line.set('STYLEREFS', 'ENTITY_LINE_STYLE')
    
    def _create_person_dense_blocks(self, root: ET.Element):
        """Create TextBlock elements for person-dense regions using clustering"""
        
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Find person tag ID
        person_tag_id = self.ner_tag_mapping.get('NER_PERSON')
        if not person_tag_id:
            return
            
        # Find block tag ID for person-dense regions
        block_tag_id = self.ner_tag_mapping.get('NER_PERSON_DENSE_BLOCK')
        if not block_tag_id:
            return
            
        # Get all person lines
        person_lines = []
        for line in root.findall('.//alto:TextLine', ns) + root.findall('.//TextLine'):
            tagrefs = line.get('TAGREFS', '')
            if person_tag_id in tagrefs.split():
                try:
                    person_lines.append({
                        'element': line,
                        'x': int(line.get('HPOS', 0)),
                        'y': int(line.get('VPOS', 0)),
                        'width': int(line.get('WIDTH', 0)),
                        'height': int(line.get('HEIGHT', 0))
                    })
                except (ValueError, TypeError):
                    continue
                
        if len(person_lines) < 3:  # Need at least 3 lines for a dense region
            return
            
        # Use clustering to find dense regions
        try:
            from sklearn.cluster import DBSCAN
            
            # Get line centers
            centers = np.array([[line['x'] + line['width']//2, line['y'] + line['height']//2] 
                               for line in person_lines])
            
            # Cluster lines with larger eps to capture more distributed person names
            clustering = DBSCAN(eps=300, min_samples=3).fit(centers)
            labels = clustering.labels_
            
            # Find largest cluster
            unique_labels = set(labels) - {-1}
            if not unique_labels:
                return
                
            largest_cluster = None
            max_size = 0
            for label in unique_labels:
                cluster_size = np.sum(labels == label)
                if cluster_size > max_size:
                    max_size = cluster_size
                    largest_cluster = label
                    
            # Get lines in largest cluster
            cluster_indices = np.where(labels == largest_cluster)[0]
            cluster_lines = [person_lines[i] for i in cluster_indices]
            
            if len(cluster_lines) < 3:
                return
                
            # Also include nearby person lines that might have been classified as noise
            nearby_lines = []
            cluster_center_y = np.mean([line['y'] + line['height']//2 for line in cluster_lines])
            
            for person_line in person_lines:
                line_center_y = person_line['y'] + person_line['height']//2
                # Include lines within 400px of cluster center
                if abs(line_center_y - cluster_center_y) <= 400:
                    if person_line not in cluster_lines:
                        nearby_lines.append(person_line)
                        
            # Combine cluster lines with nearby lines
            extended_cluster = cluster_lines + nearby_lines
            
            self.logger.info(f"Extended cluster from {len(cluster_lines)} to {len(extended_cluster)} lines")
                
            # Calculate bounding box for extended cluster
            min_x = max(0, min(line['x'] for line in extended_cluster) - 50)
            min_y = max(0, min(line['y'] for line in extended_cluster) - 50)
            max_x = max(line['x'] + line['width'] for line in extended_cluster) + 50
            max_y = max(line['y'] + line['height'] for line in extended_cluster) + 50
            
            # Find PrintSpace as the proper parent for TextBlock
            print_space = root.find('.//alto:PrintSpace', ns) or root.find('.//PrintSpace')
            if print_space is None:
                self.logger.warning("No PrintSpace found - cannot create person-dense TextBlock")
                return
                
            # Create TextBlock element
            text_block = ET.Element('TextBlock')
            text_block.set('ID', 'person_dense_region_block')
            text_block.set('TAGREFS', block_tag_id)
            text_block.set('HPOS', str(min_x))
            text_block.set('VPOS', str(min_y))
            text_block.set('WIDTH', str(max_x - min_x))
            text_block.set('HEIGHT', str(max_y - min_y))
            text_block.set('PERSON_LINES_COUNT', str(len(extended_cluster)))
            text_block.set('DESCRIPTION', f'Dense region containing {len(extended_cluster)} person name lines')
            
            # Add Shape element with polygon
            shape = ET.SubElement(text_block, 'Shape')
            polygon = ET.SubElement(shape, 'Polygon')
            
            # Create rectangle polygon
            points = f"{min_x} {min_y} {max_x} {min_y} {max_x} {max_y} {min_x} {max_y}"
            polygon.set('POINTS', points)
            
            # Insert TextBlock at the beginning of PrintSpace
            print_space.insert(0, text_block)
            
            self.logger.info(f"Created person-dense TextBlock with {len(extended_cluster)} lines at ({min_x}, {min_y})")
            
            return text_block
            
        except ImportError:
            self.logger.warning("sklearn not available - skipping person-dense region clustering")
            return None
        except Exception as e:
            self.logger.warning(f"Error creating person-dense regions: {e}")
            return None
    
    # Legacy methods for backward compatibility
    def _get_namespace(self, root: ET.Element) -> str:
        """Extract namespace from ALTO root element"""
        tag = root.tag
        if '}' in tag:
            return tag[tag.find('{')+1:tag.find('}')]
        return ''