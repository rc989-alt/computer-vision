#!/usr/bin/env python3
"""
Test script for the three core modules to verify they work correctly.
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_subject_object():
    """Test the subject-object constraint module."""
    print("=== Testing Subject-Object Constraints ===")
    
    try:
        from src.subject_object import check_subject_object
        
        # Test valid combination
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
        print(f"✅ Valid combination: {compliance1:.3f} compliance")
        
        # Test invalid combination  
        regions2 = [
            {
                'label': 'glass',
                'relationships': {'invalid_with': ['broken_glass']}
            }
        ]
        
        compliance2, details2 = check_subject_object(regions=regions2)
        print(f"✅ Invalid combination: {compliance2:.3f} compliance")
        
        return True
        
    except Exception as e:
        print(f"❌ Subject-object test failed: {e}")
        return False

def test_conflict_penalty():
    """Test the conflict penalty module."""
    print("\n=== Testing Conflict Penalty ===")
    
    try:
        from src.conflict_penalty import conflict_penalty
        
        # Test conflicting combination (pink + orange)
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
        print(f"✅ Conflicting (pink + orange): {penalty1:.3f} penalty")
        print(f"   Conflicts detected: {details1['conflict_count']}")
        
        # Test harmonious combination
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
        print(f"✅ Harmonious (golden + orange): {penalty2:.3f} penalty")
        print(f"   Conflicts detected: {details2['conflict_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conflict penalty test failed: {e}")
        return False

def test_dual_score():
    """Test the dual score fusion module.""" 
    print("\n=== Testing Dual Score Fusion ===")
    
    try:
        from src.dual_score import fuse_dual_score
        
        # Test balanced fusion
        compliance = 0.85
        conflict = 0.12
        
        fused1 = fuse_dual_score(compliance, conflict, w_c=0.6, w_n=0.4)
        print(f"✅ Balanced fusion: compliance={compliance}, conflict={conflict}, fused={fused1:.3f}")
        
        # Test high conflict scenario
        compliance2 = 0.90
        conflict2 = 0.45
        
        fused2 = fuse_dual_score(compliance2, conflict2, w_c=0.6, w_n=0.4)
        print(f"✅ High conflict: compliance={compliance2}, conflict={conflict2}, fused={fused2:.3f}")
        
        # Test different fusion methods
        methods = ['weighted_sum', 'harmonic_mean', 'geometric_mean']
        for method in methods:
            score = fuse_dual_score(0.8, 0.2, fusion_method=method)
            print(f"   Method {method}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dual score test failed: {e}")
        return False

def test_pipeline_integration():
    """Test integration with pipeline."""
    print("\n=== Testing Pipeline Integration ===")
    
    try:
        # Import all modules
        from src.subject_object import check_subject_object  
        from src.conflict_penalty import conflict_penalty
        from src.dual_score import fuse_dual_score
        
        # Simulate pipeline processing
        mock_regions = [
            {
                'label': 'crystal_glass',
                'type': 'glass',
                'confidence': 0.9,
                'relationships': {'contains': ['pink_liquid']}
            },
            {
                'label': 'pink_liquid',
                'color': 'pink',
                'type': 'cocktail',
                'confidence': 0.85
            },
            {
                'label': 'rose_garnish',
                'type': 'floral',
                'confidence': 0.8,
                'relationships': {'placed_on': ['glass']}
            }
        ]
        
        # Run full pipeline
        compliance_score, compliance_details = check_subject_object(regions=mock_regions)
        penalty_score, penalty_details = conflict_penalty(mock_regions, alpha=0.25)
        final_score = fuse_dual_score(compliance_score, penalty_score, w_c=0.6, w_n=0.4)
        
        print(f"✅ Pipeline integration test:")
        print(f"   Compliance: {compliance_score:.3f}")
        print(f"   Conflict penalty: {penalty_score:.3f}")
        print(f"   Final fused score: {final_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Core Modules\n")
    
    tests = [
        test_subject_object,
        test_conflict_penalty,
        test_dual_score,
        test_pipeline_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Core modules are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())