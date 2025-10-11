"""
Subject-Object Constraint Module

This module enforces semantic consistency between subjects and objects in cocktail images.
It analyzes detected regions and relationships to ensure logical coherence.

Key functions:
- check_subject_object: Main API for constraint validation
- extract_triples: Parse relationships from detection data
- validate_consistency: Check semantic coherence
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import json

logger = logging.getLogger(__name__)

# Semantic relationship rules
VALID_RELATIONSHIPS = {
    'glass': {
        'contains': ['liquid', 'ice', 'foam', 'garnish'],
        'supports': ['rim_garnish', 'salt_rim', 'sugar_rim'],
        'invalid_with': ['no_liquid', 'broken_glass']
    },
    'liquid': {
        'color_harmony': {
            'pink': ['rose', 'strawberry', 'raspberry', 'floral'],
            'golden': ['whiskey', 'bourbon', 'citrus', 'amber'],
            'blue': ['tropical', 'ocean', 'blueberry'],
            'green': ['mint', 'lime', 'herbs', 'absinthe'],
            'red': ['strawberry', 'cherry', 'wine', 'berry'],
            'clear': ['vodka', 'gin', 'water', 'rum']
        },
        'incompatible': ['solid_food', 'non_edible']
    },
    'garnish': {
        'fruit': ['citrus', 'berry', 'tropical', 'stone_fruit'],
        'floral': ['rose', 'lavender', 'hibiscus', 'edible_flower'],
        'herb': ['mint', 'basil', 'thyme', 'rosemary'],
        'placement': ['rim', 'float', 'skewer', 'muddle']
    }
}

def extract_triples(regions: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """
    Extract subject-relation-object triples from detected regions.
    
    Args:
        regions: List of detected regions with labels and relationships
        
    Returns:
        List of (subject, relation, object) triples
    """
    triples = []
    
    for region in regions:
        if 'relationships' not in region:
            continue
            
        subject = region.get('label', 'unknown')
        for relation, objects in region['relationships'].items():
            if isinstance(objects, list):
                for obj in objects:
                    triples.append((subject, relation, obj))
            else:
                triples.append((subject, relation, objects))
    
    # Infer common relationships from co-occurrence
    labels = [r.get('label') for r in regions if 'label' in r]
    
    # Glass-liquid relationship
    if 'glass' in labels and any('liquid' in l for l in labels):
        liquid_types = [l for l in labels if 'liquid' in l]
        for liquid in liquid_types:
            triples.append(('glass', 'contains', liquid))
    
    # Garnish-placement relationships
    garnish_items = [l for l in labels if 'garnish' in l or l in ['rose', 'mint', 'citrus', 'berry']]
    for garnish in garnish_items:
        if 'glass' in labels:
            triples.append((garnish, 'placed_on', 'glass'))
    
    return triples

def validate_consistency(triples: List[Tuple[str, str, str]]) -> Tuple[float, Dict[str, Any]]:
    """
    Validate semantic consistency of extracted triples.
    
    Args:
        triples: List of (subject, relation, object) triples
        
    Returns:
        Tuple of (consistency_score, validation_details)
    """
    details = {
        'valid_relationships': [],
        'invalid_relationships': [],
        'warnings': [],
        'total_relationships': len(triples)
    }
    
    valid_count = 0
    
    for subject, relation, obj in triples:
        relationship_key = f"{subject}-{relation}-{obj}"
        
        # Check against known valid relationships
        is_valid = False
        
        if subject in VALID_RELATIONSHIPS:
            subject_rules = VALID_RELATIONSHIPS[subject]
            
            # Check direct relationship validity
            if relation in subject_rules:
                if isinstance(subject_rules[relation], list):
                    is_valid = any(valid_obj in obj for valid_obj in subject_rules[relation])
                elif isinstance(subject_rules[relation], dict):
                    # Color harmony rules
                    for color, compatible in subject_rules[relation].items():
                        if color in subject and any(comp in obj for comp in compatible):
                            is_valid = True
                            break
            
            # Check for explicitly invalid combinations
            if 'invalid_with' in subject_rules:
                if any(invalid in obj for invalid in subject_rules['invalid_with']):
                    is_valid = False
                    details['warnings'].append(f"Invalid combination: {relationship_key}")
        
        if is_valid:
            details['valid_relationships'].append(relationship_key)
            valid_count += 1
        else:
            details['invalid_relationships'].append(relationship_key)
    
    # Calculate consistency score
    if len(triples) == 0:
        consistency_score = 1.0  # No relationships to validate
    else:
        consistency_score = valid_count / len(triples)
    
    return consistency_score, details

def check_subject_object(triples: Optional[List[Tuple[str, str, str]]] = None, 
                        regions: Optional[List[Dict[str, Any]]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Main API for subject-object constraint checking.
    
    Args:
        triples: Pre-extracted relationship triples (optional)
        regions: Detected regions to extract relationships from (optional)
        
    Returns:
        Tuple of (compliance_score, details_dict)
        
    Example:
        >>> regions = [
        ...     {'label': 'glass', 'relationships': {'contains': ['pink_liquid']}},
        ...     {'label': 'rose_garnish', 'relationships': {'placed_on': ['glass']}}
        ... ]
        >>> compliance, details = check_subject_object(regions=regions)
        >>> print(f"Compliance: {compliance:.2f}")
    """
    
    if triples is None and regions is None:
        raise ValueError("Either triples or regions must be provided")
    
    # Extract triples if not provided
    if triples is None:
        triples = extract_triples(regions)
    
    # Validate consistency
    compliance_score, validation_details = validate_consistency(triples)
    
    # Compile final details
    details = {
        'compliance_score': compliance_score,
        'extracted_triples': triples,
        'validation': validation_details,
        'module': 'subject_object_constraints',
        'version': '1.0.0'
    }
    
    logger.info(f"Subject-object constraint check: {compliance_score:.3f} compliance "
               f"({len(validation_details['valid_relationships'])}/{len(triples)} valid)")
    
    return compliance_score, details

# Convenience functions for common use cases
def check_glass_liquid_harmony(glass_type: str, liquid_color: str, liquid_type: str = '') -> float:
    """Check if glass and liquid combination is harmonious."""
    mock_regions = [
        {
            'label': 'glass',
            'type': glass_type,
            'relationships': {'contains': [f'{liquid_color}_liquid']}
        },
        {
            'label': f'{liquid_color}_liquid',
            'type': liquid_type,
            'relationships': {}
        }
    ]
    
    compliance, _ = check_subject_object(regions=mock_regions)
    return compliance

def check_garnish_placement(garnish_type: str, placement: str, liquid_color: str = '') -> float:
    """Check if garnish placement is appropriate."""
    mock_regions = [
        {
            'label': 'glass',
            'relationships': {'contains': [f'{liquid_color}_liquid'] if liquid_color else []}
        },
        {
            'label': garnish_type,
            'relationships': {'placed_on': [placement]}
        }
    ]
    
    compliance, _ = check_subject_object(regions=mock_regions)
    return compliance

if __name__ == "__main__":
    # Example usage and testing
    print("=== Subject-Object Constraint Module Test ===")
    
    # Test case 1: Valid pink floral cocktail
    regions1 = [
        {
            'label': 'crystal_glass',
            'relationships': {'contains': ['pink_liquid', 'rose_garnish']}
        },
        {
            'label': 'rose_garnish',
            'relationships': {'placed_on': ['glass']}
        }
    ]
    
    compliance1, details1 = check_subject_object(regions=regions1)
    print(f"Test 1 - Pink floral: {compliance1:.2f} compliance")
    print(f"Valid relationships: {len(details1['validation']['valid_relationships'])}")
    
    # Test case 2: Invalid combination
    regions2 = [
        {
            'label': 'glass',
            'relationships': {'contains': ['orange_liquid'], 'invalid_with': ['broken_glass']}
        }
    ]
    
    compliance2, details2 = check_subject_object(regions=regions2)
    print(f"Test 2 - Invalid combo: {compliance2:.2f} compliance")
    print(f"Warnings: {details2['validation']['warnings']}")
    
    # Test convenience functions
    glass_harmony = check_glass_liquid_harmony('crystal', 'golden', 'whiskey')
    garnish_placement = check_garnish_placement('mint', 'rim', 'green')
    
    print(f"Glass-liquid harmony: {glass_harmony:.2f}")
    print(f"Garnish placement: {garnish_placement:.2f}")