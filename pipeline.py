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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def main():
    parser = argparse.ArgumentParser(description='Computer Vision Pipeline for Cocktail Images')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--input', required=True, help='Input JSON file with image data')
    parser.add_argument('--output', required=True, help='Output JSON file for results')
    parser.add_argument('--steps', default='all', help='Pipeline steps to run (all,extract,score,rerank)')
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
    
    # Create intermediate file paths
    base_name = Path(args.output).stem
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True)
    
    scored_file = temp_dir / f"{base_name}_scored.json"
    detected_file = temp_dir / f"{base_name}_detected.json"
    compliant_file = temp_dir / f"{base_name}_compliant.json"
    
    steps = args.steps.lower().split(',')
    
    if 'all' in steps or 'extract' in steps:
        # Step 1: Extract features with CLIP and YOLO
        logger.info("🔍 Step 1: Feature Extraction")
        
        # YOLO detection
        run_command([
            'python', 'scripts/image_model.py',
            '--in', args.input,
            '--out', str(detected_file),
            '--threshold', str(config.get('detection_threshold', 0.5))
        ], "YOLO object detection")
        
        # CLIP scoring (if probe model exists)
        probe_model = config.get('clip_probe_model')
        if probe_model and os.path.exists(probe_model):
            run_command([
                'python', 'scripts/clip_probe/run_clip_probe_inference.py',
                '--model', probe_model,
                '--test-data', str(detected_file),
                '--output', str(scored_file)
            ], "CLIP probe inference")
        else:
            scored_file = detected_file
    
    if 'all' in steps or 'score' in steps:
        # Step 2: Apply compliance scoring and constraints
        logger.info("🎯 Step 2: Compliance Scoring")
        
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