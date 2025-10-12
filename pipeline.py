#!/usr/bin/env python3
"""
Complete computer vision pipeline for cocktail image analysis.

This script provides a unified interface to run the entire pipeline:
1. Image preprocessing and feature extraction
2. CLIP embedding generation
3. YOLO object detection
4. Dual scoring evaluation
5. Constraint application
6. Reranking with CoTRR-lite

Usage:
    python pipeline.py --config config/default.json --input data/input/images.json --output data/output/results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
import subprocess
import logging

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core modules
REGION_CONTROL_AVAILABLE = True
try:
    from src.subject_object import check_subject_object
    from src.conflict_penalty import conflict_penalty  
    from src.dual_score import fuse_dual_score
except ImportError as e:
    REGION_CONTROL_AVAILABLE = False
    logger.warning(f"Could not import core modules: {e}")
    logger.warning("Region control features will be disabled")

def run_command(cmd, description):
    """Run a shell command with error handling."""
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        sys.exit(1)

def apply_region_controls(input_file, output_file, config):
    """Apply region control enhancements using core modules."""
    if not REGION_CONTROL_AVAILABLE:
        logger.warning("Region control modules not available, skipping")
        return
    
    logger.info("Applying region control enhancements...")
    
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    enhanced_results = []
    
    # Process each query result  
    for item in data.get('results', data.get('inspirations', [])):
        if 'candidates' in item:
            # Process candidates for this query
            enhanced_candidates = []
            
            for candidate in item['candidates']:
                # Extract mock regions from candidate data
                regions = extract_mock_regions(candidate)
                
                # Apply region control modules
                compliance_score, compliance_details = check_subject_object(regions=regions)
                penalty_score, penalty_details = conflict_penalty(
                    regions, alpha=config.get('conflict_penalty', 0.25)
                )
                
                # Fuse scores
                base_score = candidate.get('score', 0.5)
                enhanced_score = fuse_dual_score(
                    compliance_score, penalty_score,
                    w_c=config.get('compliance_weight', 0.6),
                    w_n=0.4
                )
                
                # Combine with base score
                final_score = 0.7 * enhanced_score + 0.3 * base_score
                
                # Create enhanced candidate
                enhanced_candidate = candidate.copy()
                enhanced_candidate.update({
                    'original_score': base_score,
                    'compliance_score': compliance_score,
                    'conflict_penalty': penalty_score,
                    'enhanced_score': enhanced_score,
                    'final_score': final_score,
                    'score': final_score,  # Update main score
                    'region_controls': {
                        'compliance_details': compliance_details,
                        'penalty_details': penalty_details,
                        'applied': True
                    }
                })
                
                enhanced_candidates.append(enhanced_candidate)
            
            # Sort candidates by enhanced score
            enhanced_candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            # Update item with enhanced candidates
            enhanced_item = item.copy()
            enhanced_item['candidates'] = enhanced_candidates
            enhanced_results.append(enhanced_item)
        else:
            # Item without candidates, pass through
            enhanced_results.append(item)
    
    # Save enhanced results
    output_data = {
        'results' if 'results' in data else 'inspirations': enhanced_results,
        'metadata': {
            'enhanced_with_region_controls': True,
            'processing_timestamp': '2025-01-11T00:00:00Z',
            'config_used': config
        }
    }
    
    # Preserve any other top-level data
    for key, value in data.items():
        if key not in ['results', 'inspirations']:
            output_data[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✅ Enhanced {len(enhanced_results)} items with region controls")

def extract_mock_regions(candidate):
    """Extract mock region data from candidate for processing."""
    regions = []
    
    # Extract from alt_description
    description = candidate.get('alt_description', '').lower()
    
    # Basic glass detection
    if any(glass in description for glass in ['glass', 'coupe', 'martini', 'rocks']):
        regions.append({
            'label': 'glass',
            'type': 'crystal_glass' if 'crystal' in description else 'glass',
            'confidence': 0.9
        })
    
    # Color detection
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
    garnishes = ['rose', 'orange', 'mint', 'lime', 'berry', 'fruit', 'petal', 'herb']
    for garnish in garnishes:
        if garnish in description:
            garnish_type = 'floral' if garnish in ['rose', 'petal'] else 'fruit'
            regions.append({
                'label': f'{garnish}_garnish',
                'type': garnish_type,
                'confidence': 0.7
            })
    
    return regions

def main():
    parser = argparse.ArgumentParser(description='Computer Vision Pipeline for Cocktail Images')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--input', required=True, help='Input JSON file with image data')
    parser.add_argument('--output', required=True, help='Output JSON file for results')
    parser.add_argument('--steps', default='all', help='Pipeline steps to run (all,extract,score,rerank)')
    parser.add_argument('--mode', default='region_control', choices=['baseline', 'region_control'], 
                       help='Pipeline mode: baseline (CLIP-only) or region_control (full pipeline)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    logger.info("🚀 Starting Computer Vision Pipeline")
    logger.info(f"📁 Input: {args.input}")
    logger.info(f"📁 Output: {args.output}")
    logger.info(f"⚙️ Config: {args.config}")
    logger.info(f"🎛️ Mode: {args.mode}")
    
    if args.mode == 'region_control' and not REGION_CONTROL_AVAILABLE:
        logger.error("Region control mode requested but core modules not available")
        logger.error("Please install dependencies or use --mode baseline")
        sys.exit(1)
    
    # Create intermediate file paths
    base_name = Path(args.output).stem
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True)
    
    scored_file = temp_dir / f"{base_name}_scored.json"
    detected_file = temp_dir / f"{base_name}_detected.json"
    compliant_file = temp_dir / f"{base_name}_compliant.json"
    region_enhanced_file = temp_dir / f"{base_name}_region_enhanced.json"
    
    steps = args.steps.lower().split(',')
    
    if 'all' in steps or 'extract' in steps:
        # Step 1: Extract features with CLIP and YOLO
        logger.info("🔍 Step 1: Feature Extraction")
        
        # YOLO detection
        run_command([
            sys.executable, 'scripts/image_model.py',
            '--in', args.input,
            '--out', str(detected_file),
            '--threshold', str(config.get('detection_threshold', 0.5))
        ], "YOLO object detection")
        
        # CLIP scoring (if probe model exists)
        probe_model = config.get('clip_probe_model')
        if probe_model and os.path.exists(probe_model):
            run_command([
                sys.executable, 'scripts/clip_probe/run_clip_probe_inference.py',
                '--model', probe_model,
                '--test-data', str(detected_file),
                '--output', str(scored_file)
            ], "CLIP probe inference")
        else:
            scored_file = detected_file
    
    if 'all' in steps or 'score' in steps:
        # Step 2: Apply compliance scoring and constraints
        logger.info("🎯 Step 2: Compliance Scoring")
        
        if args.mode == 'baseline':
            # Baseline mode: skip region control processing
            logger.info("🔧 Baseline mode: skipping region control processing")
            compliant_file = scored_file
        else:
            # Full compliance scoring with region controls
            run_command([
                'node', 'scripts/rerank_with_compliance.mjs',
                '--in', str(scored_file),
                '--out', str(compliant_file),
                '--family', config.get('family', 'cocktail_detection'),
                '--positive', config.get('positive_class', 'glass'),
                '--negative', config.get('negative_class', 'background'),
                '--weight', str(config.get('compliance_weight', 1.0)),
                '--require-glass', str(config.get('require_glass', True)).lower(),
                '--penalty', str(config.get('conflict_penalty', 0.25))
            ], "Compliance scoring and constraint application")
            
            # Apply region control enhancements
            if REGION_CONTROL_AVAILABLE:
                logger.info("🎯 Step 2.5: Region Control Enhancement")
                apply_region_controls(str(compliant_file), str(region_enhanced_file), config)
                compliant_file = region_enhanced_file
    
    if 'all' in steps or 'rerank' in steps:
        # Step 3: Advanced reranking with CoTRR-lite
        logger.info("🎲 Step 3: Advanced Reranking")
        
        llm_proxy = config.get('llm_proxy_url')
        cmd = [
            'node', 'scripts/rerank_listwise_llm.mjs',
            '--in', str(compliant_file),
            '--out', args.output,
            '--family', config.get('family', 'cocktail_detection'),
            '--positive', config.get('positive_class', 'glass'),
            '--negative', config.get('negative_class', 'background'),
            '--llm-weight', str(config.get('llm_weight', 0.6)),
            '--top-k', str(config.get('top_k', 10))
        ]
        
        if llm_proxy:
            cmd.extend(['--proxy-url', llm_proxy])
        
        run_command(cmd, "CoTRR-lite listwise reranking")
    
    logger.info("✅ Pipeline completed successfully!")
    logger.info(f"📁 Final results: {args.output}")

if __name__ == "__main__":
    main()