"""
Computer Vision Pipeline Core Modules

This package contains the three core modules for the cocktail computer vision pipeline:

1. subject_object: Subject-object semantic constraint validation
2. conflict_penalty: Conflict detection and penalty calculation  
3. dual_score: Dual score fusion and normalization

Example usage:
    from src.subject_object import check_subject_object
    from src.conflict_penalty import conflict_penalty
    from src.dual_score import fuse_dual_score
    
    # Check semantic constraints
    compliance, details = check_subject_object(regions=detected_regions)
    
    # Calculate conflict penalties
    penalty, penalty_details = conflict_penalty(detected_regions, alpha=0.3)
    
    # Fuse scores
    final_score = fuse_dual_score(compliance, penalty, w_c=0.6, w_n=0.4)
"""

from .subject_object import check_subject_object
from .conflict_penalty import conflict_penalty
from .dual_score import fuse_dual_score

__version__ = "1.0.0"
__author__ = "Computer Vision Pipeline Team"

__all__ = [
    'check_subject_object',
    'conflict_penalty', 
    'fuse_dual_score'
]