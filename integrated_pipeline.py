#!/usr/bin/env python3
"""
Pipeline Integration with Canary Monitoring

Example of how to integrate canary checks into the main processing pipeline.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_canary_check() -> bool:
    """
    Run canary check before pipeline execution.
    
    Returns:
        True if pipeline should proceed, False if blocked
    """
    logger.info("🕯️  Running canary pre-flight check...")
    
    try:
        result = subprocess.run([
            sys.executable, "canary_automation.py", "--check"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Canary check passed - pipeline cleared to proceed")
            return True
        else:
            logger.error("🚨 Canary check failed - pipeline blocked")
            logger.error(f"   Reason: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ Canary check timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"💥 Canary check error: {e}")
        return False

def run_main_pipeline(mode: str = "region_control") -> bool:
    """
    Run the main computer vision pipeline.
    
    Args:
        mode: Pipeline mode (baseline or region_control)
        
    Returns:
        True if successful, False if failed
    """
    logger.info(f"🚀 Starting main pipeline in {mode} mode...")
    
    try:
        result = subprocess.run([
            sys.executable, "pipeline.py", "--mode", mode
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            logger.info("✅ Main pipeline completed successfully")
            logger.info(f"   Output: {result.stdout.strip().split(chr(10))[-1]}")  # Last line
            return True
        else:
            logger.error("❌ Main pipeline failed")
            logger.error(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ Main pipeline timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"💥 Main pipeline error: {e}")
        return False

def post_pipeline_validation() -> bool:
    """
    Run post-pipeline validation checks.
    
    Returns:
        True if validation passed, False if issues detected
    """
    logger.info("🔍 Running post-pipeline validation...")
    
    # Check if output files exist
    expected_outputs = [
        "results/latest_scores.json",
        "results/latest_analysis.json"
    ]
    
    missing_files = []
    for output_file in expected_outputs:
        if not Path(output_file).exists():
            missing_files.append(output_file)
    
    if missing_files:
        logger.warning(f"⚠️  Missing expected outputs: {missing_files}")
        return False
    
    # Additional validation could include:
    # - Score range validation
    # - Output format validation  
    # - Performance metric checks
    
    logger.info("✅ Post-pipeline validation passed")
    return True

def main():
    """Main pipeline execution with canary integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Pipeline with Canary Monitoring')
    parser.add_argument('--mode', choices=['baseline', 'region_control'], 
                        default='region_control', help='Pipeline execution mode')
    parser.add_argument('--skip-canary', action='store_true', 
                        help='Skip canary check (use with caution)')
    parser.add_argument('--force', action='store_true',
                        help='Force pipeline execution even if canary fails')
    
    args = parser.parse_args()
    
    logger.info("🎯 INTEGRATED PIPELINE EXECUTION")
    logger.info("=" * 50)
    
    success = True
    
    # Step 1: Pre-flight canary check
    if not args.skip_canary:
        canary_passed = run_canary_check()
        
        if not canary_passed:
            if args.force:
                logger.warning("⚠️  Proceeding despite canary failure (--force used)")
            else:
                logger.error("🛑 Pipeline execution blocked by canary check")
                logger.error("   Use --force to override (not recommended)")
                sys.exit(1)
    else:
        logger.warning("⚠️  Canary check skipped (--skip-canary used)")
    
    # Step 2: Main pipeline execution
    pipeline_success = run_main_pipeline(args.mode)
    
    if not pipeline_success:
        logger.error("💥 Pipeline execution failed")
        success = False
    
    # Step 3: Post-pipeline validation
    if pipeline_success:
        validation_success = post_pipeline_validation()
        
        if not validation_success:
            logger.warning("⚠️  Post-pipeline validation detected issues")
            success = False
    
    # Step 4: Summary and cleanup
    if success:
        logger.info("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("   All checks passed, outputs validated")
    else:
        logger.error("💥 PIPELINE EXECUTION FAILED")
        logger.error("   Check logs above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()