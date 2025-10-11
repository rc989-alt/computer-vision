#!/usr/bin/env python3
"""
Borderline Review Integration

Complete integration script for human-in-the-loop borderline review workflow.
"""

import subprocess
import sys
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BorderlineReviewWorkflow:
    """Complete borderline review workflow."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.review_dir = self.project_root / "review"
        self.review_dir.mkdir(exist_ok=True)
    
    def step1_extract_borderline_items(self, low_margin: float = 0.10, 
                                      high_margin: float = 0.25, 
                                      limit: int = 50) -> bool:
        """Step 1: Extract borderline items for review."""
        
        logger.info(f"🔍 Extracting borderline items (margin {low_margin}-{high_margin})")
        
        try:
            result = subprocess.run([
                sys.executable, "make_borderline_items.py",
                "--low", str(low_margin),
                "--high", str(high_margin), 
                "--out", str(self.review_dir / "items.json"),
                "--limit", str(limit)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Failed to extract borderline items: {result.stderr}")
                return False
            
            # Count extracted items
            items_file = self.review_dir / "items.json"
            if items_file.exists():
                with open(items_file, 'r') as f:
                    items = json.load(f)
                
                logger.info(f"✅ Extracted {len(items)} borderline items")
                return len(items) > 0
            
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("Borderline extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Borderline extraction error: {e}")
            return False
    
    def step2_start_review_ui(self, port: int = 8000) -> str:
        """Step 2: Start review UI server."""
        
        logger.info(f"🌐 Starting review UI server on port {port}")
        
        # Check if items exist
        items_file = self.review_dir / "items.json"
        if not items_file.exists():
            logger.error("No items.json found. Run step 1 first.")
            return ""
        
        try:
            # Start server in background
            import threading
            import http.server
            import socketserver
            import os
            
            # Change to review directory for server
            original_cwd = os.getcwd()
            os.chdir(self.review_dir)
            
            handler = http.server.SimpleHTTPRequestHandler
            httpd = socketserver.TCPServer(("", port), handler)
            
            # Start server in background thread
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            
            # Restore working directory
            os.chdir(original_cwd)
            
            url = f"http://localhost:{port}"
            logger.info(f"✅ Review UI available at: {url}")
            
            # Try to open browser
            try:
                import webbrowser
                webbrowser.open(url)
            except:
                pass
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to start review UI: {e}")
            return ""
    
    def step3_apply_patch(self, patch_file: str = "overlay_patch.json", 
                         dry_run: bool = False) -> bool:
        """Step 3: Apply overlay patch from human review."""
        
        patch_path = self.review_dir / patch_file
        if not patch_path.exists():
            logger.error(f"Patch file not found: {patch_path}")
            logger.info("Complete human review and export overlay patch first")
            return False
        
        logger.info(f"⚙️ Applying overlay patch (dry_run={dry_run})")
        
        try:
            cmd = [
                sys.executable, "apply_overlay_patch.py",
                "--patch", str(patch_path)
            ]
            
            if dry_run:
                cmd.append("--dry-run")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Patch application failed: {result.stderr}")
                return False
            
            logger.info("✅ Overlay patch applied successfully")
            print(result.stdout)
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Patch application timed out")
            return False
        except Exception as e:
            logger.error(f"Patch application error: {e}")
            return False
    
    def step4_validate_canary(self) -> bool:
        """Step 4: Run canary check after overlay changes."""
        
        logger.info("🕯️ Running canary validation after overlay changes")
        
        try:
            result = subprocess.run([
                sys.executable, "canary_automation.py", "--check"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error("🚨 Canary check failed after overlay changes")
                logger.error("Pipeline blocked - investigate drift")
                return False
            
            logger.info("✅ Canary check passed - overlay changes validated")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Canary check timed out")
            return False
        except Exception as e:
            logger.error(f"Canary check error: {e}")
            return False
    
    def run_complete_workflow(self, interactive: bool = True) -> bool:
        """Run complete borderline review workflow."""
        
        logger.info("🎯 STARTING BORDERLINE REVIEW WORKFLOW")
        logger.info("=" * 50)
        
        # Step 1: Extract borderline items
        if not self.step1_extract_borderline_items():
            logger.error("❌ Step 1 failed: Could not extract borderline items")
            return False
        
        # Step 2: Start review UI
        review_url = self.step2_start_review_ui()
        if not review_url:
            logger.error("❌ Step 2 failed: Could not start review UI")
            return False
        
        if interactive:
            print(f"\n🔍 HUMAN REVIEW REQUIRED")
            print(f"   1. Open: {review_url}")
            print(f"   2. Review borderline items (≤5 minutes)")
            print(f"   3. Export 'overlay_patch.json'")
            print(f"   4. Press Enter when complete...")
            input()
        
        # Step 3: Apply patch (dry run first)
        logger.info("Validating patch...")
        if not self.step3_apply_patch(dry_run=True):
            logger.error("❌ Step 3 failed: Patch validation failed")
            return False
        
        # Apply patch for real
        if not self.step3_apply_patch(dry_run=False):
            logger.error("❌ Step 3 failed: Patch application failed")
            return False
        
        # Step 4: Validate with canary
        if not self.step4_validate_canary():
            logger.error("❌ Step 4 failed: Canary validation failed")
            return False
        
        logger.info("🎉 BORDERLINE REVIEW WORKFLOW COMPLETED SUCCESSFULLY")
        return True

def main():
    """CLI interface for borderline review workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Borderline Review Workflow')
    parser.add_argument('--step', choices=['1', '2', '3', '4', 'all'], 
                        help='Run specific step or all steps')
    parser.add_argument('--low-margin', type=float, default=0.10,
                        help='Low margin threshold for borderline items')
    parser.add_argument('--high-margin', type=float, default=0.25,
                        help='High margin threshold for borderline items')
    parser.add_argument('--limit', type=int, default=50,
                        help='Maximum borderline items to extract')
    parser.add_argument('--port', type=int, default=8000,
                        help='Review UI server port')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run without interactive prompts')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate patch without applying')
    
    args = parser.parse_args()
    
    workflow = BorderlineReviewWorkflow()
    
    if args.step == '1':
        success = workflow.step1_extract_borderline_items(
            args.low_margin, args.high_margin, args.limit)
    elif args.step == '2':
        url = workflow.step2_start_review_ui(args.port)
        success = bool(url)
        if success:
            print(f"Review UI started at: {url}")
            print("Press Ctrl+C to stop server")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n✅ Review server stopped")
    elif args.step == '3':
        success = workflow.step3_apply_patch(dry_run=args.dry_run)
    elif args.step == '4':
        success = workflow.step4_validate_canary()
    else:  # all steps
        success = workflow.run_complete_workflow(
            interactive=not args.non_interactive)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()