#!/usr/bin/env python3
"""
Debug nDCG Calculation - Let's see what's happening with the scores
"""

import json
import random
import math

def debug_ndcg_calculation():
    """Debug what's going wrong with nDCG"""
    
    print("🔍 DEBUGGING nDCG CALCULATION")
    print("=" * 50)
    
    # Test with realistic baseline scores
    baseline_scores = [random.uniform(0.1, 0.8) for _ in range(10)]
    baseline_scores.sort(reverse=True)
    
    print(f"Sample baseline scores (top 10): {[f'{s:.3f}' for s in baseline_scores]}")
    
    # Convert to relevance
    relevance = []
    for score in baseline_scores:
        if score >= 0.7:
            relevance.append(3)
        elif score >= 0.5:
            relevance.append(2)
        elif score >= 0.3:
            relevance.append(1)
        else:
            relevance.append(0)
    
    print(f"Converted relevance:              {relevance}")
    
    # DCG calculation
    dcg = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            gain = 2**rel - 1
            discount = math.log2(i + 2)
            contribution = gain / discount
            print(f"Position {i+1}: rel={rel}, gain={gain}, discount={discount:.3f}, contrib={contribution:.3f}")
            dcg += contribution
    
    print(f"DCG: {dcg:.3f}")
    
    # IDCG calculation
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        if rel > 0:
            gain = 2**rel - 1
            discount = math.log2(i + 2)
            idcg += gain / discount
    
    print(f"IDCG: {idcg:.3f}")
    print(f"nDCG: {dcg/idcg:.3f}")
    
    # Let's also test RA-Guard style scores
    print(f"\n" + "="*50)
    ra_guard_scores = [0.543, 0.521, 0.498, 0.467, 0.445, 0.423, 0.401, 0.378, 0.356, 0.334]
    print(f"RA-Guard scores (top 10):        {[f'{s:.3f}' for s in ra_guard_scores]}")
    
    # Convert to relevance
    ra_relevance = []
    for score in ra_guard_scores:
        if score >= 0.7:
            ra_relevance.append(3)
        elif score >= 0.5:
            ra_relevance.append(2)
        elif score >= 0.3:
            ra_relevance.append(1)
        else:
            ra_relevance.append(0)
    
    print(f"RA-Guard relevance:               {ra_relevance}")
    
    # DCG
    ra_dcg = 0.0
    for i, rel in enumerate(ra_relevance):
        if rel > 0:
            gain = 2**rel - 1
            discount = math.log2(i + 2)
            ra_dcg += gain / discount
    
    # IDCG (same as before)
    ra_ndcg = ra_dcg / idcg if idcg > 0 else 0.0
    
    print(f"RA-Guard DCG: {ra_dcg:.3f}")
    print(f"RA-Guard nDCG: {ra_ndcg:.3f}")
    
    print(f"\nΔnDCG: {ra_ndcg - dcg/idcg:.3f}")

def test_with_low_baseline():
    """Test with intentionally low baseline scores"""
    
    print(f"\n🔍 TESTING WITH LOW BASELINE")
    print("=" * 50)
    
    # Really bad baseline
    baseline_scores = [0.2, 0.15, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06]
    # Good RA-Guard
    ra_guard_scores = [0.75, 0.72, 0.68, 0.65, 0.62, 0.58, 0.55, 0.52, 0.48, 0.45]
    
    def compute_ndcg(scores):
        relevance = []
        for score in scores:
            if score >= 0.7:
                relevance.append(3)
            elif score >= 0.5:
                relevance.append(2)
            elif score >= 0.3:
                relevance.append(1)
            else:
                relevance.append(0)
        
        # DCG
        dcg = 0.0
        for i, rel in enumerate(relevance):
            if rel > 0:
                gain = 2**rel - 1
                discount = math.log2(i + 2)
                dcg += gain / discount
        
        # IDCG - use the best possible relevance for this set
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            if rel > 0:
                gain = 2**rel - 1
                discount = math.log2(i + 2)
                idcg += gain / discount
        
        return dcg / idcg if idcg > 0 else 0.0
    
    baseline_ndcg = compute_ndcg(baseline_scores)
    ra_guard_ndcg = compute_ndcg(ra_guard_scores)
    
    print(f"Baseline scores: {[f'{s:.2f}' for s in baseline_scores]}")
    print(f"RA-Guard scores: {[f'{s:.2f}' for s in ra_guard_scores]}")
    print(f"Baseline nDCG: {baseline_ndcg:.3f}")
    print(f"RA-Guard nDCG: {ra_guard_ndcg:.3f}")
    print(f"Lift: {ra_guard_ndcg - baseline_ndcg:+.3f}")

if __name__ == "__main__":
    debug_ndcg_calculation()
    test_with_low_baseline()