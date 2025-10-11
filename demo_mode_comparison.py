#!/usr/bin/env python3
"""
Demo script showing baseline vs region_control comparison using core modules.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.subject_object import check_subject_object
from src.conflict_penalty import conflict_penalty
from src.dual_score import fuse_dual_score

def extract_regions_from_candidate(candidate):
    """Extract mock regions from candidate description."""
    regions = []
    description = candidate.get('alt_description', '').lower()
    
    # Glass detection
    if any(glass in description for glass in ['glass', 'coupe', 'martini', 'rocks']):
        glass_type = 'crystal_glass' if 'crystal' in description else 'glass'
        regions.append({
            'label': glass_type,
            'type': 'glass',
            'confidence': 0.9,
            'relationships': {'contains': ['liquid']}
        })
    
    # Color and liquid detection
    colors = ['pink', 'golden', 'blue', 'green', 'red', 'clear', 'amber', 'purple']
    for color in colors:
        if color in description:
            regions.append({
                'label': f'{color}_liquid',
                'color': color,
                'type': 'cocktail',
                'confidence': 0.8
            })
    
    # Garnish detection
    garnish_map = {
        'rose': 'floral', 'petal': 'floral', 'lavender': 'floral',
        'orange': 'citrus', 'lime': 'citrus', 'lemon': 'citrus',
        'mint': 'herb', 'basil': 'herb',
        'berry': 'fruit', 'strawberry': 'fruit', 'cherry': 'fruit'
    }
    
    for garnish, category in garnish_map.items():
        if garnish in description:
            regions.append({
                'label': f'{garnish}_garnish',
                'type': category,
                'confidence': 0.7,
                'relationships': {'placed_on': ['glass']}
            })
    
    return regions

def process_baseline(candidate):
    """Process candidate with baseline (CLIP-only) scoring."""
    return {
        'id': candidate['id'],
        'score': candidate['score'],
        'mode': 'baseline',
        'processing': 'clip_only'
    }

def process_region_control(candidate):
    """Process candidate with region control enhancements."""
    # Extract regions
    regions = extract_regions_from_candidate(candidate)
    
    # Apply core modules
    compliance_score, compliance_details = check_subject_object(regions=regions)
    penalty_score, penalty_details = conflict_penalty(regions, alpha=0.25)
    
    # Fuse scores
    base_score = candidate.get('score', 0.5)
    enhanced_score = fuse_dual_score(compliance_score, penalty_score, w_c=0.6, w_n=0.4)
    
    # Combine with base score (70% enhanced, 30% base)
    final_score = 0.7 * enhanced_score + 0.3 * base_score
    
    return {
        'id': candidate['id'],
        'original_score': base_score,
        'compliance_score': compliance_score,
        'conflict_penalty': penalty_score,
        'enhanced_score': enhanced_score,  
        'final_score': final_score,
        'score': final_score,
        'mode': 'region_control',
        'processing': 'clip_yolo_region_control',
        'regions_detected': len(regions),
        'compliance_details': compliance_details,
        'penalty_details': penalty_details
    }

def compare_modes(input_file, output_file):
    """Compare baseline vs region control modes."""
    
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    comparison_results = []
    
    for inspiration in data.get('inspirations', []):
        query = inspiration['query']
        candidates = inspiration.get('candidates', [])
        
        baseline_candidates = []
        region_control_candidates = []
        
        for candidate in candidates:
            # Process with both modes
            baseline_result = process_baseline(candidate)
            region_control_result = process_region_control(candidate)
            
            baseline_candidates.append(baseline_result)
            region_control_candidates.append(region_control_result)
        
        # Sort candidates by score
        baseline_candidates.sort(key=lambda x: x['score'], reverse=True)
        region_control_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        comparison_results.append({
            'query': query,
            'baseline': {
                'top_candidate': baseline_candidates[0] if baseline_candidates else None,
                'all_candidates': baseline_candidates
            },
            'region_control': {
                'top_candidate': region_control_candidates[0] if region_control_candidates else None,
                'all_candidates': region_control_candidates
            },
            'improvement': (
                region_control_candidates[0]['score'] - baseline_candidates[0]['score']
                if baseline_candidates and region_control_candidates else 0
            )
        })
    
    # Calculate summary statistics
    improvements = [r['improvement'] for r in comparison_results]
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    
    baseline_avg = sum(r['baseline']['top_candidate']['score'] for r in comparison_results) / len(comparison_results)
    region_control_avg = sum(r['region_control']['top_candidate']['score'] for r in comparison_results) / len(comparison_results)
    
    # Create output
    output_data = {
        'comparison_results': comparison_results,
        'summary': {
            'queries_processed': len(comparison_results),
            'average_baseline_score': baseline_avg,
            'average_region_control_score': region_control_avg,
            'average_improvement': avg_improvement,
            'improvement_percentage': (avg_improvement / baseline_avg * 100) if baseline_avg > 0 else 0,
            'positive_improvements': sum(1 for imp in improvements if imp > 0),
            'negative_improvements': sum(1 for imp in improvements if imp < 0),
            'neutral_improvements': sum(1 for imp in improvements if imp == 0)
        },
        'metadata': {
            'generated_by': 'mode_comparison_demo',
            'timestamp': '2025-01-11T00:00:00Z',
            'input_file': input_file,
            'core_modules_used': ['subject_object', 'conflict_penalty', 'dual_score']
        }
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_data

def main():
    """Run the mode comparison demo."""
    print("🔄 Running Baseline vs Region Control Comparison\n")
    
    input_file = 'demo/samples.json'
    output_file = 'runs/mode_comparison.json'
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    try:
        results = compare_modes(input_file, output_file)
        
        # Print summary
        summary = results['summary']
        print("📊 Comparison Results:")
        print(f"   Queries processed: {summary['queries_processed']}")
        print(f"   Average baseline score: {summary['average_baseline_score']:.3f}")
        print(f"   Average region control score: {summary['average_region_control_score']:.3f}")
        print(f"   Average improvement: +{summary['average_improvement']:.3f} ({summary['improvement_percentage']:.1f}%)")
        print(f"   Positive improvements: {summary['positive_improvements']}/{summary['queries_processed']}")
        
        print(f"\n📁 Detailed results saved to: {output_file}")
        
        # Show top improvements
        sorted_results = sorted(results['comparison_results'], key=lambda x: x['improvement'], reverse=True)
        print(f"\n🎯 Top Improvements:")
        for i, result in enumerate(sorted_results[:3]):
            print(f"   {i+1}. \"{result['query'][:50]}...\" (+{result['improvement']:.3f})")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())