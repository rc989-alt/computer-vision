#!/usr/bin/env python3
"""
Apply Overlay Patch

Apply human review decisions to the overlay system without modifying frozen snapshots.  
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import hashlib
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OverlayPatchApplier:
    """Apply overlay patches from human review decisions."""
    
    def __init__(self, overlay_db_path: str = "data/dataset/metadata"):
        self.overlay_db_path = Path(overlay_db_path)
        self.overlay_db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized overlay patch applier: {overlay_db_path}")
    
    def load_patch(self, patch_path: str) -> Dict[str, Any]:
        """Load overlay patch from JSON file."""
        
        with open(patch_path, 'r') as f:
            patch = json.load(f)
        
        # Validate patch structure
        required_fields = ['run_id', 'overlay_id', 'actions']
        for field in required_fields:
            if field not in patch:
                raise ValueError(f"Patch missing required field: {field}")
        
        logger.info(f"Loaded patch with {len(patch['actions'])} actions")
        return patch
    
    def validate_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patch for conflicts and consistency."""
        
        issues = []
        warnings = []
        stats = {}
        
        actions = patch['actions']
        
        # Count action types
        action_counts = {}
        affected_ids = set()
        
        for action in actions:
            op = action.get('op', 'unknown')
            action_counts[op] = action_counts.get(op, 0) + 1
            
            item_id = action.get('id')
            if item_id:
                if item_id in affected_ids:
                    issues.append(f"Duplicate action for item {item_id}")
                affected_ids.add(item_id)
        
        stats['action_counts'] = action_counts
        stats['affected_items'] = len(affected_ids)
        
        # Validate action structure
        for i, action in enumerate(actions):
            op = action.get('op')
            if op not in ['approve', 'remove', 'fix']:
                issues.append(f"Action {i}: Invalid operation '{op}'")
            
            if not action.get('id'):
                issues.append(f"Action {i}: Missing item ID")
            
            if op in ['remove', 'fix'] and not action.get('reason'):
                warnings.append(f"Action {i}: Missing reason for {op} operation")
        
        # Check for potential conflicts
        fix_actions = [a for a in actions if a['op'] == 'fix']
        if fix_actions:
            warnings.append(f"{len(fix_actions)} items need manual fixes - review required")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'stats': stats
        }
    
    def find_existing_overlays(self) -> List[Path]:
        """Find existing overlay files in the database."""
        
        overlay_files = list(self.overlay_db_path.glob("*-overlay.json"))
        overlay_files.sort()  # Lexicographic order
        return overlay_files
    
    def get_next_overlay_version(self) -> str:
        """Determine the next overlay version number."""
        
        existing_overlays = self.find_existing_overlays()
        
        if not existing_overlays:
            return "v1.0-overlay"
        
        # Extract version numbers
        versions = []
        for overlay_file in existing_overlays:
            name = overlay_file.stem
            if '-overlay' in name:
                version_part = name.replace('-overlay', '')
                try:
                    # Handle versions like v1.1, v1.2, etc.
                    if version_part.startswith('v'):
                        major, minor = version_part[1:].split('.')
                        versions.append((int(major), int(minor)))
                except:
                    continue
        
        if not versions:
            return "v1.0-overlay"
        
        # Get next version
        max_major, max_minor = max(versions)
        next_version = f"v{max_major}.{max_minor + 1}-overlay"
        
        return next_version
    
    def load_frozen_snapshot_hash(self) -> str:
        """Load frozen snapshot hash for validation."""
        
        frozen_path = self.overlay_db_path / "frozen_snapshot.json"
        if not frozen_path.exists():
            logger.warning("Frozen snapshot not found for hash validation")
            return ""
        
        with open(frozen_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def create_overlay_from_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Create overlay structure from patch."""
        
        # Convert patch actions to overlay format
        exclusions = []
        replacements = []
        approvals = []
        
        for action in patch['actions']:
            item_id = action['id']
            reason = action.get('reason', 'human_review')
            
            if action['op'] == 'remove':
                exclusions.append({
                    'id': item_id,
                    'reason': reason,
                    'source': 'human_review',
                    'excluded_at': datetime.now().isoformat()
                })
            
            elif action['op'] == 'fix':
                # For now, fixes are treated as exclusions pending manual correction
                exclusions.append({
                    'id': item_id,
                    'reason': f"needs_fix: {reason}",
                    'source': 'human_review',
                    'excluded_at': datetime.now().isoformat()
                })
            
            elif action['op'] == 'approve':
                approvals.append({
                    'id': item_id,
                    'approved_at': datetime.now().isoformat(),
                    'source': 'human_review'
                })
        
        # Create overlay structure
        overlay = {
            'version': self.get_next_overlay_version(),
            'created_at': datetime.now().isoformat(),
            'parent_snapshot_sha256': self.load_frozen_snapshot_hash(),
            'source': 'human_review',
            'patch_metadata': {
                'original_run_id': patch['run_id'],
                'original_overlay_id': patch['overlay_id'],
                'reviewer': patch.get('reviewer', 'unknown'),
                'review_metadata': patch.get('metadata', {})
            },
            'exclusions': exclusions,
            'replacements': replacements,
            'approvals': approvals,
            'stats': {
                'total_actions': len(patch['actions']),
                'exclusions_count': len(exclusions),
                'replacements_count': len(replacements),
                'approvals_count': len(approvals)
            }
        }
        
        return overlay
    
    def save_overlay(self, overlay: Dict[str, Any]) -> str:
        """Save overlay to database."""
        
        version = overlay['version']
        overlay_file = self.overlay_db_path / f"{version}.json"
        
        with open(overlay_file, 'w') as f:
            json.dump(overlay, f, indent=2)
        
        logger.info(f"Saved overlay to {overlay_file}")
        return str(overlay_file)
    
    def apply_patch(self, patch_path: str, dry_run: bool = False) -> Dict[str, Any]:
        """Apply overlay patch with validation."""
        
        # Load patch
        patch = self.load_patch(patch_path)
        
        # Validate patch
        validation = self.validate_patch(patch)
        
        if not validation['is_valid']:
            logger.error("Patch validation failed")
            return {
                'success': False,
                'validation': validation,
                'error': 'Patch validation failed'
            }
        
        if validation['warnings']:
            logger.warning(f"Patch has {len(validation['warnings'])} warnings")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")
        
        # Create overlay
        overlay = self.create_overlay_from_patch(patch)
        
        result = {
            'success': True,
            'validation': validation,
            'overlay_version': overlay['version'],
            'actions_applied': len(patch['actions']),
            'dry_run': dry_run
        }
        
        if dry_run:
            logger.info("DRY RUN: Patch would be applied successfully")
            result['overlay_preview'] = overlay
        else:
            # Save overlay
            overlay_file = self.save_overlay(overlay)
            result['overlay_file'] = overlay_file
            logger.info(f"Applied patch successfully: {overlay_file}")
        
        return result

def main():
    """CLI interface for overlay patch application."""
    parser = argparse.ArgumentParser(description='Apply overlay patch from human review')
    parser.add_argument('--patch', required=True, help='Path to overlay patch JSON file')
    parser.add_argument('--overlay-db', default='data/dataset/metadata', help='Overlay database directory')
    parser.add_argument('--dry-run', action='store_true', help='Validate patch without applying')
    parser.add_argument('--run-id', help='Expected run ID for validation')
    
    args = parser.parse_args()
    
    # Apply patch
    applier = OverlayPatchApplier(args.overlay_db)
    result = applier.apply_patch(args.patch, dry_run=args.dry_run)
    
    # Print results
    if result['success']:
        print("✅ PATCH APPLICATION SUCCESSFUL")
        print(f"   Overlay version: {result['overlay_version']}")
        print(f"   Actions applied: {result['actions_applied']}")
        
        if result['dry_run']:
            print("   Mode: DRY RUN (no changes made)")
        else:
            print(f"   Overlay saved: {result['overlay_file']}")
        
        # Print validation details
        validation = result['validation']
        if validation['warnings']:
            print(f"\n⚠️  Warnings ({len(validation['warnings'])}):")
            for warning in validation['warnings']:
                print(f"   - {warning}")
        
        # Print statistics
        stats = validation['stats']
        print(f"\n📊 Patch Statistics:")
        print(f"   Total items affected: {stats['affected_items']}")
        for action, count in stats['action_counts'].items():
            print(f"   {action}: {count}")
    
    else:
        print("❌ PATCH APPLICATION FAILED")
        print(f"   Error: {result['error']}")
        
        validation = result.get('validation', {})
        if validation.get('issues'):
            print(f"\n🚨 Issues ({len(validation['issues'])}):")
            for issue in validation['issues']:
                print(f"   - {issue}")
        
        exit(1)

if __name__ == "__main__":
    main()