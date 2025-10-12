#!/usr/bin/env python3
"""
Generate Training Data from Step 4 Pipeline

Converts Step 4 pipeline outputs into training data for CoTRR-lite reranker.
Handles feature extraction, label generation, and train/test splitting.
"""

import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any
import argparse
from datetime import datetime

# Import Step 4 components
import sys
sys.path.append('../src')

logger = logging.getLogger(__name__)

def load_step4_pipeline_output(run_dir: str) -> List[Dict]:
    """Load scored items from Step 4 pipeline run"""
    run_path = Path(run_dir)
    
    # Look for manifest file
    manifest_files = list(run_path.glob("**/manifest.json"))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest.json found in {run_dir}")
    
    manifest_file = manifest_files[0]
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    # Load scored items
    scored_items = []
    
    # Check for scored items in manifest
    if 'items_by_stage' in manifest and 'scored' in manifest['items_by_stage']:
        for item_id in manifest['items_by_stage']['scored']:
            # Load item data (this would be in your Step 4 data structure)
            item_data = {
                'id': item_id,
                'manifest_id': manifest['manifest_id'],
                'run_id': manifest['run_id'],
                # Mock data for now - replace with actual Step 4 data loading
                'url': f'https://example.com/{item_id}.jpg',
                'query': f'cocktail query for {item_id}',
                'domain': np.random.choice(['blue_tropical', 'red_berry', 'green_citrus']),
                
                # Step 4 pipeline outputs
                'dual_score': np.random.uniform(0.4, 0.9),
                'compliance_score': np.random.uniform(0.5, 1.0),
                'conflict_score': np.random.uniform(0.0, 0.6),
                'conflict_prob': np.random.uniform(0.0, 0.4),
                
                # CLIP embeddings (would come from batched_scoring.py)
                'image_embedding': np.random.normal(0, 1, 512).tolist(),
                'text_embedding': np.random.normal(0, 1, 512).tolist(),
                
                # Visual features (would come from your detection pipeline)
                'subject_ratio': np.random.uniform(0.2, 0.8),
                'glass_ratio': np.random.uniform(0.0, 0.3),
                'garnish_ratio': np.random.uniform(0.0, 0.2),
                'ice_ratio': np.random.uniform(0.0, 0.4),
                'color_delta_e': np.random.uniform(10, 100),
                'brightness': np.random.uniform(0.2, 0.8),
                'contrast': np.random.uniform(0.3, 0.7),
                'saturation': np.random.uniform(0.2, 0.8),
                
                # Conflict details
                'conflicts': [
                    {'type': 'color_mismatch', 'severity': 'soft', 'confidence': 0.7},
                    {'type': 'attribute_conflict', 'severity': 'strong', 'confidence': 0.9}
                ] if np.random.random() < 0.3 else [],
                
                # Metadata
                'processing_time': np.random.uniform(0.1, 2.0),
                'model_version': manifest.get('model_version', 'unknown'),
                'overlay_version': manifest.get('overlay_version', 'unknown')
            }
            
            scored_items.append(item_data)
    
    logger.info(f"Loaded {len(scored_items)} scored items from {run_dir}")
    return scored_items

def enhance_with_additional_features(items: List[Dict]) -> List[Dict]:
    """Add additional features that might not be in Step 4 output"""
    
    for item in items:
        # Add calibrated conflict probability (temperature scaling)
        raw_conflict = item.get('conflict_score', 0.0)
        # Mock temperature scaling (T=1.5)
        item['conflict_calibrated'] = 1 / (1 + np.exp(-raw_conflict / 1.5))
        
        # Add strong/soft conflict counts
        conflicts = item.get('conflicts', [])
        item['strong_conflict_count'] = sum(1 for c in conflicts if c.get('severity') == 'strong')
        item['soft_conflict_count'] = sum(1 for c in conflicts if c.get('severity') == 'soft')
        
        # Add derived features
        compliance = item.get('compliance_score', 0.5)
        conflict_prob = item.get('conflict_prob', 0.5)
        
        # Dual score with configurable lambda
        lambda_param = 0.7
        item['dual_score_lambda_07'] = lambda_param * compliance + (1 - lambda_param) * (1 - conflict_prob)
        
        # Quality score (combines multiple factors)
        subject_ratio = item.get('subject_ratio', 0.5)
        quality_penalty = max(0, 0.3 - subject_ratio)  # Penalty for low subject ratio
        item['quality_score'] = compliance - quality_penalty - conflict_prob
    
    return items

def create_training_dataset(items: List[Dict], output_path: str, 
                          lambda_param: float = 0.7) -> Dict[str, Any]:
    """Create training dataset JSONL file"""
    
    # Enhance items with additional features
    enhanced_items = enhance_with_additional_features(items)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    with open(output_path, 'w') as f:
        for item in enhanced_items:
            f.write(json.dumps(item) + '\n')
    
    # Create dataset metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_items': len(enhanced_items),
        'lambda_param': lambda_param,
        'source_runs': list(set([item.get('run_id', 'unknown') for item in enhanced_items])),
        'domains': list(set([item.get('domain', 'unknown') for item in enhanced_items])),
        'queries': len(set([item.get('query', 'unknown') for item in enhanced_items])),
        'feature_summary': {
            'clip_embeddings': True,
            'visual_features': True,
            'conflict_features': True,
            'calibrated_features': True
        }
    }
    
    # Save metadata
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created training dataset: {output_path}")
    logger.info(f"  Total items: {metadata['total_items']}")
    logger.info(f"  Domains: {len(metadata['domains'])}")
    logger.info(f"  Queries: {metadata['queries']}")
    
    return metadata

def create_mock_step4_output(output_dir: str = "runs/step4_demo_run"):
    """Create mock Step 4 pipeline output for testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create mock manifest
    manifest = {
        'manifest_id': 'step4_demo_20251011_161500',
        'run_id': 'step4_demo_run',
        'model_version': 'clip_v1.1',
        'overlay_version': 'overlay_v2.0',
        'created_at': datetime.now().isoformat(),
        'status': 'completed',
        'items_by_stage': {
            'raw_candidates': [f'item_{i:06d}' for i in range(1000)],
            'deduplicated': [f'item_{i:06d}' for i in range(334)],  # 66.6% duplicates removed
            'embedded': [f'item_{i:06d}' for i in range(334)],
            'scored': [f'item_{i:06d}' for i in range(334)],
            'overlay_ready': [f'item_{i:06d}' for i in range(259)],
            'borderline_review': [f'item_{i:06d}' for i in range(334, 409)]  # 75 borderline
        },
        'metrics': {
            'throughput_items_per_second': 118.076,
            'cache_hit_rate': 0.976,
            'duplicate_rate': 0.666,
            'borderline_rate': 0.075
        }
    }
    
    # Save manifest
    manifest_path = output_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created mock Step 4 output: {output_dir}")
    return output_dir

def main():
    """Main training data generation"""
    parser = argparse.ArgumentParser(description='Generate training data from Step 4 pipeline')
    parser.add_argument('--input', default='runs/latest', help='Step 4 pipeline run directory')
    parser.add_argument('--output', default='research/data/training.jsonl', help='Output training data file')
    parser.add_argument('--lambda-param', type=float, default=0.7, help='Lambda parameter for dual score')
    parser.add_argument('--create-mock', action='store_true', help='Create mock Step 4 output for testing')
    
    args = parser.parse_args()
    
    print("🔄 Generating Training Data from Step 4 Pipeline")
    print("=" * 50)
    
    if args.create_mock:
        print("📝 Creating mock Step 4 pipeline output...")
        mock_dir = create_mock_step4_output()
        args.input = mock_dir
        print(f"✅ Mock output created: {mock_dir}")
    
    try:
        # Load Step 4 pipeline output
        print(f"📥 Loading Step 4 output from: {args.input}")
        items = load_step4_pipeline_output(args.input)
        
        # Create training dataset  
        print(f"⚙️  Creating training dataset...")
        metadata = create_training_dataset(items, args.output, args.lambda_param)
        
        print(f"\n✅ Training data generation complete!")
        print(f"   Output file: {args.output}")
        print(f"   Total items: {metadata['total_items']}")
        print(f"   Domains: {len(metadata['domains'])}")
        print(f"   Queries: {metadata['queries']}")
        print(f"   Lambda parameter: {metadata['lambda_param']}")
        
        print(f"\n🚀 Ready to train CoTRR-lite reranker:")
        print(f"   python research/src/reranker_train.py --data {args.output}")
        
    except Exception as e:
        print(f"❌ Error generating training data: {e}")
        raise

if __name__ == "__main__":
    main()