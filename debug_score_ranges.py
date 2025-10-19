#!/usr/bin/env python3
"""
Debug score ranges to understand baseline vs RA-Guard scoring differences
"""

import json
import numpy as np
import sys
sys.path.append('.')

from scripts.demo_candidate_library import CandidateLibraryDemo

def analyze_ra_guard_scores():
    """Analyze actual RA-Guard score distribution"""
    
    print("🔍 ANALYZING RA-GUARD SCORE DISTRIBUTION")
    print("=" * 60)
    
    ra_guard = CandidateLibraryDemo(gallery_dir="pilot_gallery")
    
    # Test queries
    test_queries = [
        "colorful drink",
        "tropical cocktail", 
        "martini glass",
        "refreshing drink",
        "elegant evening cocktail"
    ]
    
    all_scores = []
    
    for query in test_queries:
        print(f"\n🔄 Query: '{query}'")
        result = ra_guard.process_query(
            query=query,
            domain="cocktails", 
            num_candidates=100
        )
        
        scores = result.reranking_scores
        all_scores.extend(scores)
        
        print(f"   Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   Mean: {np.mean(scores):.3f}")
        print(f"   Std: {np.std(scores):.3f}")
        
    print("\n📊 OVERALL RA-GUARD STATISTICS:")
    print(f"Total scores analyzed: {len(all_scores)}")
    print(f"Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
    print(f"Mean score: {np.mean(all_scores):.3f}")
    print(f"Std dev: {np.std(all_scores):.3f}")
    print(f"Median: {np.median(all_scores):.3f}")
    print(f"25th percentile: {np.percentile(all_scores, 25):.3f}")
    print(f"75th percentile: {np.percentile(all_scores, 75):.3f}")
    
    # Analyze top scores
    top_scores = []
    for i in range(0, len(all_scores), 100):  # Top score per query
        query_scores = all_scores[i:i+100]
        if query_scores:
            top_scores.append(max(query_scores))
    
    print(f"\n🎯 TOP SCORE ANALYSIS:")
    print(f"Top scores: {len(top_scores)}")
    print(f"Top score range: {min(top_scores):.3f} - {max(top_scores):.3f}")
    print(f"Top score mean: {np.mean(top_scores):.3f}")
    
    return {
        'all_scores': all_scores,
        'mean': np.mean(all_scores),
        'std': np.std(all_scores),
        'min': min(all_scores),
        'max': max(all_scores),
        'top_mean': np.mean(top_scores)
    }

if __name__ == "__main__":
    stats = analyze_ra_guard_scores()
    
    print(f"\n💡 BASELINE CALIBRATION RECOMMENDATION:")
    print(f"Suggested baseline mean: {stats['mean'] * 0.85:.3f}")  # Slightly lower than RA-Guard
    print(f"Suggested baseline std: {stats['std']:.3f}")
    print(f"Expected improvement: ~{(stats['mean'] - stats['mean'] * 0.85):.3f} points")