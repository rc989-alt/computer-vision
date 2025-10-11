#!/usr/bin/env python3
"""
Emergency Quarantine - Apply overlay patch to remove keyboard item

Creates overlay patch to quarantine the obvious off-topic item (demo_017 - keyboard)
that slipped through to borderline review.
"""

import json
from datetime import datetime
from apply_overlay_patch import OverlayPatchApplier

def quarantine_keyboard_item():
    """Quarantine the keyboard item (demo_017) via overlay patch"""
    
    # Create patch to remove the keyboard item
    patch = {
        "metadata": {
            "patch_type": "emergency_quarantine", 
            "created_at": datetime.now().isoformat(),
            "creator": "system",
            "reason": "Pre-off-topic gate would have caught this - keyboard not cocktail"
        },
        "operations": [
            {
                "op": "remove",
                "id": "demo_017",
                "reason": "off_topic_precheck",
                "details": {
                    "original_query": "black charcoal cocktail with activated carbon",
                    "actual_content": "keyboard/computer equipment", 
                    "sim_cocktail": 0.5,
                    "sim_not_cocktail": 0.275,
                    "gate_reason": "combined_clip_objects_fail"
                }
            }
        ]
    }
    
    # Save patch file
    patch_file = f"overlays/emergency_quarantine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(patch_file, 'w') as f:
        json.dump(patch, f, indent=2)
    
    print(f"Created emergency quarantine patch: {patch_file}")
    
    # Apply the patch using overlay system
    applier = OverlayPatchApplier()
    
    # Dry run first
    print("\n=== DRY RUN ===")
    result = applier.apply_patch(patch_file, dry_run=True)
    print(f"Dry run result: {result}")
    
    if result["success"]:
        print("\n=== APPLYING PATCH ===")
        result = applier.apply_patch(patch_file, dry_run=False)
        print(f"Application result: {result}")
        
        if result["success"]:
            print(f"\n✅ Successfully quarantined keyboard item demo_017")
            print(f"   New overlay: {result['overlay_path']}")
            print(f"   Items removed: {len(result['stats']['removed'])}")
        else:
            print(f"\n❌ Failed to apply patch: {result['error']}")
    else:
        print(f"\n❌ Patch validation failed: {result['error']}")

if __name__ == "__main__":
    quarantine_keyboard_item()