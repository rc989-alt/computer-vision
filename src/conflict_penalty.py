"""
Conflict Penalty Module

This module detects and penalizes semantic conflicts between visual elements
in cocktail images, such as color-ingredient mismatches and inappropriate garnishes.

Key functions:
- conflict_penalty: Main API for conflict detection and penalty calculation
- build_conflict_graph: Create knowledge graph of conflicts
- detect_conflicts: Identify conflicts in detected regions
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Set
import json
import math

logger = logging.getLogger(__name__)

# Predefined conflict rules
CONFLICT_RULES = {
    'color_ingredient_conflicts': {
        'pink': {
            'forbidden_ingredients': ['orange_juice', 'orange_liqueur', 'orange_bitters'],
            'forbidden_garnishes': ['orange_peel', 'orange_slice', 'orange_wheel'],
            'penalty_weight': 0.8
        },
        'orange': {
            'forbidden_ingredients': ['grape_juice', 'blueberry', 'blackberry'],
            'forbidden_garnishes': ['grape_garnish', 'dark_berry'],
            'penalty_weight': 0.7
        },
        'blue': {
            'forbidden_ingredients': ['tomato_juice', 'red_wine', 'cranberry'],
            'forbidden_garnishes': ['red_cherry', 'strawberry'],
            'penalty_weight': 0.9
        },
        'green': {
            'forbidden_ingredients': ['red_wine', 'cherry_juice'],
            'forbidden_garnishes': ['red_cherry', 'red_strawberry'],
            'penalty_weight': 0.6
        }
    },
    'garnish_season_conflicts': {
        'winter': {
            'forbidden_garnishes': ['fresh_berries', 'tropical_fruit'],
            'penalty_weight': 0.4
        },
        'summer': {
            'forbidden_garnishes': ['cinnamon_stick', 'mulling_spices'],
            'penalty_weight': 0.3
        }
    },
    'glass_content_conflicts': {
        'martini_glass': {
            'forbidden_contents': ['thick_cream', 'heavy_foam', 'chunky_fruit'],
            'penalty_weight': 0.5
        },
        'rocks_glass': {
            'forbidden_contents': ['layered_shots', 'delicate_foam'],
            'penalty_weight': 0.4
        }
    },
    'temperature_conflicts': {
        'hot_drinks': {
            'forbidden_garnishes': ['ice_cubes', 'frozen_fruit'],
            'penalty_weight': 0.9
        },
        'frozen_drinks': {
            'forbidden_garnishes': ['hot_spices', 'warm_nuts'],
            'penalty_weight': 0.8
        }
    }
}

def build_conflict_graph(regions: List[Dict[str, Any]], 
                        custom_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a knowledge graph of potential conflicts from detected regions.
    
    Args:
        regions: List of detected regions with labels and properties
        custom_rules: Optional custom conflict rules to merge with defaults
        
    Returns:
        Conflict graph dictionary with nodes and edges
    """
    
    # Merge custom rules with defaults
    rules = CONFLICT_RULES.copy()
    if custom_rules:
        for category, category_rules in custom_rules.items():
            if category in rules:
                rules[category].update(category_rules)
            else:
                rules[category] = category_rules
    
    graph = {
        'nodes': {},
        'edges': [],
        'rules': rules,
        'metadata': {
            'total_regions': len(regions),
            'rules_applied': len(rules)
        }
    }
    
    # Create nodes from regions
    for i, region in enumerate(regions):
        node_id = f"region_{i}"
        graph['nodes'][node_id] = {
            'label': region.get('label', 'unknown'),
            'properties': region.get('properties', {}),
            'color': region.get('color', ''),
            'type': region.get('type', ''),
            'confidence': region.get('confidence', 1.0)
        }
    
    # Create conflict edges based on rules
    node_ids = list(graph['nodes'].keys())
    for i, node1_id in enumerate(node_ids):
        for j, node2_id in enumerate(node_ids[i+1:], i+1):
            node1 = graph['nodes'][node1_id]
            node2 = graph['nodes'][node2_id]
            
            # Check for conflicts between these nodes
            conflicts = _check_node_conflicts(node1, node2, rules)
            
            if conflicts:
                graph['edges'].append({
                    'source': node1_id,
                    'target': node2_id,
                    'conflicts': conflicts,
                    'total_penalty': sum(c['penalty'] for c in conflicts)
                })
    
    return graph

def _check_node_conflicts(node1: Dict[str, Any], node2: Dict[str, Any], 
                         rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for conflicts between two nodes based on rules."""
    conflicts = []
    
    # Color-ingredient conflicts
    if 'color_ingredient_conflicts' in rules:
        color_rules = rules['color_ingredient_conflicts']
        
        # Check if node1 has color and node2 has forbidden ingredient/garnish
        node1_color = node1.get('color', '').lower()
        node2_label = node2.get('label', '').lower()
        node2_type = node2.get('type', '').lower()
        
        if node1_color in color_rules:
            rule = color_rules[node1_color]
            forbidden_items = (rule.get('forbidden_ingredients', []) + 
                             rule.get('forbidden_garnishes', []))
            
            for forbidden in forbidden_items:
                if forbidden.lower() in node2_label or forbidden.lower() in node2_type:
                    conflicts.append({
                        'type': 'color_ingredient_conflict',
                        'description': f"{node1_color} color conflicts with {forbidden}",
                        'penalty': rule.get('penalty_weight', 0.5),
                        'rule_source': 'color_ingredient_conflicts'
                    })
    
    # Glass-content conflicts
    if 'glass_content_conflicts' in rules:
        glass_rules = rules['glass_content_conflicts']
        
        # Check if one node is glass and other is forbidden content
        for node, other_node in [(node1, node2), (node2, node1)]:
            node_type = node.get('type', '').lower()
            other_label = other_node.get('label', '').lower()
            
            if node_type in glass_rules:
                rule = glass_rules[node_type]
                forbidden_contents = rule.get('forbidden_contents', [])
                
                for forbidden in forbidden_contents:
                    if forbidden.lower() in other_label:
                        conflicts.append({
                            'type': 'glass_content_conflict',
                            'description': f"{node_type} glass conflicts with {forbidden}",
                            'penalty': rule.get('penalty_weight', 0.5),
                            'rule_source': 'glass_content_conflicts'
                        })
    
    return conflicts

def detect_conflicts(regions: List[Dict[str, Any]], 
                    graph: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], float]:
    """
    Detect conflicts in regions using conflict graph.
    
    Args:
        regions: List of detected regions
        graph: Pre-built conflict graph (optional)
        
    Returns:
        Tuple of (conflict_list, total_penalty_score)
    """
    
    if graph is None:
        graph = build_conflict_graph(regions)
    
    conflicts = []
    total_penalty = 0.0
    
    # Extract conflicts from graph edges
    for edge in graph['edges']:
        source_node = graph['nodes'][edge['source']]
        target_node = graph['nodes'][edge['target']]
        
        for conflict in edge['conflicts']:
            conflict_detail = {
                'source_region': source_node['label'],
                'target_region': target_node['label'],
                'conflict_type': conflict['type'],
                'description': conflict['description'],
                'penalty': conflict['penalty'],
                'confidence': min(source_node.get('confidence', 1.0), 
                                target_node.get('confidence', 1.0))
            }
            conflicts.append(conflict_detail)
            total_penalty += conflict['penalty'] * conflict_detail['confidence']
    
    return conflicts, total_penalty

def conflict_penalty(regions: List[Dict[str, Any]], 
                    graph: Optional[Dict[str, Any]] = None,
                    alpha: float = 0.3) -> Tuple[float, Dict[str, Any]]:
    """
    Main API for conflict detection and penalty calculation.
    
    Args:
        regions: List of detected regions with labels and properties
        graph: Optional pre-built conflict graph
        alpha: Penalty scaling factor (0.0 = no penalty, 1.0 = full penalty)
        
    Returns:
        Tuple of (penalty_score, details_dict)
        
    Example:
        >>> regions = [
        ...     {'label': 'pink_liquid', 'color': 'pink', 'type': 'cocktail'},
        ...     {'label': 'orange_garnish', 'type': 'orange_peel'}
        ... ]
        >>> penalty, details = conflict_penalty(regions, alpha=0.3)
        >>> print(f"Penalty: {penalty:.3f}")
    """
    
    if not regions:
        return 0.0, {'conflicts': [], 'total_penalty': 0.0, 'alpha': alpha}
    
    # Build or use provided conflict graph
    if graph is None:
        graph = build_conflict_graph(regions)
    
    # Detect conflicts
    conflicts, raw_penalty = detect_conflicts(regions, graph)
    
    # Apply alpha scaling and normalization
    scaled_penalty = raw_penalty * alpha
    
    # Normalize penalty to [0, 1] range using sigmoid-like function
    normalized_penalty = 1.0 - math.exp(-scaled_penalty)
    
    # Compile details
    details = {
        'conflicts': conflicts,
        'conflict_count': len(conflicts),
        'raw_penalty': raw_penalty,
        'scaled_penalty': scaled_penalty,
        'final_penalty': normalized_penalty,
        'alpha': alpha,
        'graph_stats': {
            'nodes': len(graph['nodes']),
            'edges': len(graph['edges']),
            'rules_applied': len(graph['rules'])
        },
        'module': 'conflict_penalty',
        'version': '1.0.0'
    }
    
    logger.info(f"Conflict penalty calculation: {len(conflicts)} conflicts detected, "
               f"penalty = {normalized_penalty:.3f} (α={alpha})")
    
    return normalized_penalty, details

# Convenience functions for specific conflict types
def check_color_ingredient_conflict(color: str, ingredients: List[str]) -> float:
    """Check for color-ingredient conflicts specifically."""
    mock_regions = [
        {'label': f'{color}_liquid', 'color': color, 'type': 'cocktail'}
    ]
    
    for ingredient in ingredients:
        mock_regions.append({
            'label': ingredient,
            'type': 'ingredient'
        })
    
    penalty, _ = conflict_penalty(mock_regions)
    return penalty

def check_seasonal_conflict(season: str, garnishes: List[str]) -> float:
    """Check for seasonal garnish conflicts."""
    mock_regions = [
        {'label': 'cocktail', 'properties': {'season': season}}
    ]
    
    for garnish in garnishes:
        mock_regions.append({
            'label': garnish,
            'type': 'garnish'
        })
    
    penalty, _ = conflict_penalty(mock_regions)
    return penalty

if __name__ == "__main__":
    # Example usage and testing
    print("=== Conflict Penalty Module Test ===")
    
    # Test case 1: Pink cocktail with orange garnish (should conflict)
    regions1 = [
        {
            'label': 'pink_cocktail',
            'color': 'pink',
            'type': 'cocktail',
            'confidence': 0.9
        },
        {
            'label': 'orange_peel',
            'type': 'orange_peel',
            'confidence': 0.8
        }
    ]
    
    penalty1, details1 = conflict_penalty(regions1, alpha=0.3)
    print(f"Test 1 - Pink + Orange: penalty = {penalty1:.3f}")
    print(f"Conflicts detected: {details1['conflict_count']}")
    
    # Test case 2: Harmonious combination
    regions2 = [
        {
            'label': 'golden_whiskey',
            'color': 'golden',
            'type': 'whiskey',
            'confidence': 0.95
        },
        {
            'label': 'orange_peel',
            'type': 'citrus_garnish',
            'confidence': 0.85
        }
    ]
    
    penalty2, details2 = conflict_penalty(regions2, alpha=0.3)
    print(f"Test 2 - Golden + Orange: penalty = {penalty2:.3f}")
    print(f"Conflicts detected: {details2['conflict_count']}")
    
    # Test convenience functions
    color_conflict = check_color_ingredient_conflict('pink', ['orange_juice'])
    seasonal_conflict = check_seasonal_conflict('winter', ['fresh_berries'])
    
    print(f"Color-ingredient conflict: {color_conflict:.3f}")
    print(f"Seasonal conflict: {seasonal_conflict:.3f}")