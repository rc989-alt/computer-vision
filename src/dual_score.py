"""
Dual Score Fusion Module

This module combines compliance and conflict scores into a unified dual scoring system
for cocktail image evaluation, with configurable weighting and normalization options.

Key functions:
- fuse_dual_score: Main API for score fusion
- normalize_scores: Score normalization utilities
- weighted_harmonic_mean: Alternative fusion method
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import math

logger = logging.getLogger(__name__)

def normalize_scores(scores: Union[List[float], Dict[str, float]], 
                    method: str = 'minmax') -> Union[List[float], Dict[str, float]]:
    """
    Normalize scores using various methods.
    
    Args:
        scores: List or dict of scores to normalize
        method: Normalization method ('minmax', 'zscore', 'sigmoid', 'none')
        
    Returns:
        Normalized scores in same format as input
    """
    
    if isinstance(scores, dict):
        values = list(scores.values())
        keys = list(scores.keys())
        is_dict = True
    else:
        values = scores
        keys = None
        is_dict = False
    
    if not values or method == 'none':
        return scores
    
    if method == 'minmax':
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            normalized = [1.0] * len(values)
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    elif method == 'zscore':
        mean_val = sum(values) / len(values)
        var_val = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = math.sqrt(var_val) if var_val > 0 else 1.0
        normalized = [(v - mean_val) / std_val for v in values]
        # Convert to [0, 1] range using sigmoid
        normalized = [1 / (1 + math.exp(-v)) for v in normalized]
    
    elif method == 'sigmoid':
        normalized = [1 / (1 + math.exp(-v)) for v in values]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_dict:
        return dict(zip(keys, normalized))
    return normalized

def weighted_harmonic_mean(scores: List[float], weights: List[float]) -> float:
    """
    Calculate weighted harmonic mean of scores.
    
    Args:
        scores: List of scores
        weights: List of weights (must sum to 1.0)
        
    Returns:
        Weighted harmonic mean
    """
    if len(scores) != len(weights):
        raise ValueError("Scores and weights must have same length")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    
    # Handle zero scores to avoid division by zero
    safe_scores = [max(s, 1e-10) for s in scores]
    
    denominator = sum(w / s for w, s in zip(weights, safe_scores))
    
    if denominator == 0:
        return 0.0
    
    return 1.0 / denominator

def fuse_dual_score(compliance: float, 
                   conflict: float,
                   w_c: float = 0.5,
                   w_n: float = 0.5, 
                   normalize: bool = True,
                   fusion_method: str = 'weighted_sum') -> float:
    """
    Main API for dual score fusion combining compliance and conflict scores.
    
    Args:
        compliance: Compliance score (higher is better, [0, 1])
        conflict: Conflict penalty score (higher is worse, [0, 1])  
        w_c: Weight for compliance score
        w_n: Weight for conflict penalty (negative contribution)
        normalize: Whether to normalize weights to sum to 1.0
        fusion_method: Method for score fusion ('weighted_sum', 'harmonic_mean', 'geometric_mean')
        
    Returns:
        Fused dual score in [0, 1] range
        
    Example:
        >>> compliance = 0.85
        >>> conflict = 0.12
        >>> fused = fuse_dual_score(compliance, conflict, w_c=0.7, w_n=0.3)
        >>> print(f"Fused score: {fused:.3f}")
    """
    
    # Input validation
    if not (0 <= compliance <= 1):
        logger.warning(f"Compliance score {compliance} outside [0,1] range")
        compliance = max(0, min(1, compliance))
    
    if not (0 <= conflict <= 1):
        logger.warning(f"Conflict score {conflict} outside [0,1] range")
        conflict = max(0, min(1, conflict))
    
    # Normalize weights if requested
    if normalize and (w_c + w_n) > 0:
        total_weight = w_c + w_n
        w_c = w_c / total_weight
        w_n = w_n / total_weight
    
    # Convert conflict penalty to positive contribution (1 - conflict)
    conflict_contribution = 1.0 - conflict
    
    # Apply fusion method
    if fusion_method == 'weighted_sum':
        fused_score = w_c * compliance + w_n * conflict_contribution
    
    elif fusion_method == 'harmonic_mean':
        scores = [compliance, conflict_contribution]
        weights = [w_c, w_n]
        fused_score = weighted_harmonic_mean(scores, weights)
    
    elif fusion_method == 'geometric_mean':
        # Weighted geometric mean
        if compliance > 0 and conflict_contribution > 0:
            fused_score = (compliance ** w_c) * (conflict_contribution ** w_n)
        else:
            fused_score = 0.0
    
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Ensure result is in [0, 1] range
    fused_score = max(0.0, min(1.0, fused_score))
    
    logger.debug(f"Dual score fusion: compliance={compliance:.3f}, conflict={conflict:.3f}, "
                f"w_c={w_c:.3f}, w_n={w_n:.3f}, method={fusion_method}, result={fused_score:.3f}")
    
    return fused_score

def fuse_multiple_scores(scores: Dict[str, float], 
                        weights: Dict[str, float],
                        normalize_weights: bool = True,
                        fusion_method: str = 'weighted_sum') -> Tuple[float, Dict[str, Any]]:
    """
    Fuse multiple named scores with associated weights.
    
    Args:
        scores: Dictionary of score_name -> score_value
        weights: Dictionary of score_name -> weight_value
        normalize_weights: Whether to normalize weights to sum to 1.0
        fusion_method: Method for score fusion
        
    Returns:
        Tuple of (fused_score, details_dict)
    """
    
    # Validate inputs
    if not scores:
        return 0.0, {'error': 'No scores provided'}
    
    missing_weights = set(scores.keys()) - set(weights.keys())
    if missing_weights:
        logger.warning(f"Missing weights for scores: {missing_weights}")
        for key in missing_weights:
            weights[key] = 1.0 / len(scores)  # Equal weight for missing
    
    # Normalize weights if requested
    if normalize_weights:
        total_weight = sum(weights[k] for k in scores.keys())
        if total_weight > 0:
            weights = {k: weights[k] / total_weight for k in scores.keys()}
    
    # Apply fusion
    if fusion_method == 'weighted_sum':
        fused_score = sum(scores[k] * weights[k] for k in scores.keys())
    
    elif fusion_method == 'harmonic_mean':
        score_list = [scores[k] for k in scores.keys()]
        weight_list = [weights[k] for k in scores.keys()]
        fused_score = weighted_harmonic_mean(score_list, weight_list)
    
    elif fusion_method == 'geometric_mean':
        fused_score = 1.0
        for k in scores.keys():
            if scores[k] > 0:
                fused_score *= scores[k] ** weights[k]
            else:
                fused_score = 0.0
                break
    
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Compile details
    details = {
        'input_scores': scores.copy(),
        'weights_used': {k: weights[k] for k in scores.keys()},
        'fusion_method': fusion_method,
        'normalized_weights': normalize_weights,
        'fused_score': fused_score,
        'module': 'dual_score_fusion',
        'version': '1.0.0'
    }
    
    return fused_score, details

def calculate_confidence_weighted_score(base_score: float, 
                                      confidence: float,
                                      confidence_threshold: float = 0.5) -> float:
    """
    Apply confidence weighting to a base score.
    
    Args:
        base_score: Base score to weight
        confidence: Confidence level [0, 1]
        confidence_threshold: Minimum confidence for full score
        
    Returns:
        Confidence-weighted score
    """
    if confidence >= confidence_threshold:
        return base_score
    else:
        # Linear scaling based on confidence
        scaling_factor = confidence / confidence_threshold
        return base_score * scaling_factor

# Preset fusion configurations
FUSION_PRESETS = {
    'balanced': {
        'w_c': 0.5,
        'w_n': 0.5,
        'fusion_method': 'weighted_sum',
        'normalize': True
    },
    'compliance_focused': {
        'w_c': 0.7,
        'w_n': 0.3,
        'fusion_method': 'weighted_sum',
        'normalize': True
    },
    'conflict_sensitive': {
        'w_c': 0.4,
        'w_n': 0.6,
        'fusion_method': 'harmonic_mean',
        'normalize': True
    },
    'conservative': {
        'w_c': 0.3,
        'w_n': 0.7,
        'fusion_method': 'geometric_mean',
        'normalize': True
    }
}

def fuse_with_preset(compliance: float, conflict: float, preset: str = 'balanced') -> float:
    """
    Fuse scores using a predefined preset configuration.
    
    Args:
        compliance: Compliance score
        conflict: Conflict penalty score
        preset: Preset name from FUSION_PRESETS
        
    Returns:
        Fused score using preset configuration
    """
    if preset not in FUSION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(FUSION_PRESETS.keys())}")
    
    config = FUSION_PRESETS[preset]
    return fuse_dual_score(compliance, conflict, **config)

if __name__ == "__main__":
    # Example usage and testing
    print("=== Dual Score Fusion Module Test ===")
    
    # Test case 1: Balanced fusion
    compliance1 = 0.85
    conflict1 = 0.12
    
    fused1 = fuse_dual_score(compliance1, conflict1, w_c=0.6, w_n=0.4)
    print(f"Test 1 - Balanced: compliance={compliance1}, conflict={conflict1}, fused={fused1:.3f}")
    
    # Test case 2: High conflict penalty
    compliance2 = 0.90
    conflict2 = 0.45
    
    fused2 = fuse_dual_score(compliance2, conflict2, w_c=0.6, w_n=0.4)
    print(f"Test 2 - High conflict: compliance={compliance2}, conflict={conflict2}, fused={fused2:.3f}")
    
    # Test different fusion methods
    methods = ['weighted_sum', 'harmonic_mean', 'geometric_mean']
    for method in methods:
        score = fuse_dual_score(0.8, 0.2, fusion_method=method)
        print(f"Method {method}: {score:.3f}")
    
    # Test presets
    for preset_name in FUSION_PRESETS:
        preset_score = fuse_with_preset(0.75, 0.25, preset_name)
        print(f"Preset {preset_name}: {preset_score:.3f}")
    
    # Test multiple score fusion
    multi_scores = {
        'visual_appeal': 0.8,
        'ingredient_match': 0.7,
        'garnish_quality': 0.9,
        'glass_suitability': 0.85
    }
    multi_weights = {
        'visual_appeal': 0.3,
        'ingredient_match': 0.3,
        'garnish_quality': 0.2,
        'glass_suitability': 0.2
    }
    
    multi_fused, multi_details = fuse_multiple_scores(multi_scores, multi_weights)
    print(f"Multi-score fusion: {multi_fused:.3f}")
    print(f"Components: {multi_details['input_scores']}")