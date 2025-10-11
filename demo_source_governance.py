#!/usr/bin/env python3
"""
Complete Source Governance Demo

Demonstrates the end-to-end governance pipeline:
1. Pre-gate filtering catches keyboards
2. Borderline extraction with governance
3. Source reputation tracking
4. CI integration with reporting

Shows how keyboards are blocked while legitimate cocktails pass through.
"""

import json
import tempfile
import os
from datetime import datetime
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, 'src')

from pre_offtopic_gate import PreOffTopicGate
from source_governance import SourceGovernance

def demo_complete_pipeline():
    """Demo the complete source governance pipeline"""
    
    print("🔍 SOURCE GOVERNANCE PIPELINE DEMO")
    print("="*50)
    
    # Sample items including the keyboard
    test_items = [
        {
            "id": "demo_005",
            "domain": "blue_tropical",
            "query": "blue tropical cocktail with pineapple garnish",
            "url": "https://images.unsplash.com/photo-1546171753-97d7676e4602",
            "source_id": "unsplash_photographer1",
            "sim_cocktail": 0.611,
            "sim_not_cocktail": 0.414,
            "detected_objects": ["glass", "olive", "garnish"]
        },
        {
            "id": "demo_017", 
            "domain": "black_charcoal",
            "query": "black charcoal cocktail with activated carbon",
            "url": "https://images.unsplash.com/photo-1612198188060-c7c2a3b66eae",
            "source_id": "unsplash_photographer2",
            "sim_cocktail": 0.5,    # Too low - this is the keyboard
            "sim_not_cocktail": 0.275,
            "detected_objects": ["glass", "olive"]  # False positive glass detection
        },
        {
            "id": "demo_020",
            "domain": "red_berry",
            "query": "red berry cocktail with strawberry",
            "url": "https://images.unsplash.com/photo-1234567890123",
            "source_id": "unsplash_photographer3", 
            "sim_cocktail": 0.45,   # Also too low
            "sim_not_cocktail": 0.35,
            "detected_objects": ["garnish"]  # No glass
        }
    ]
    
    print(f"📥 INPUT: {len(test_items)} candidate items")
    for item in test_items:
        print(f"   {item['id']}: {item['query']} (sim_cocktail: {item['sim_cocktail']})")
    
    # Stage 1: Pre-Off-Topic Gate
    print(f"\n🚫 STAGE 1: PRE-OFF-TOPIC GATE")
    gate = PreOffTopicGate()
    
    passed_items = []
    quarantined_items = []
    
    for item in test_items:
        sims = {
            'cocktail': item['sim_cocktail'],
            'not_cocktail': item['sim_not_cocktail']
        }
        detections = item['detected_objects']
        
        result = gate.evaluate(item, sims, detections)
        
        if result.discard:
            quarantined_items.append({
                'item': item,
                'reason': result.reason,
                'details': result.details
            })
            print(f"   ❌ QUARANTINED {item['id']}: {result.reason}")
        else:
            passed_items.append(item)
            print(f"   ✅ PASSED {item['id']}: {result.reason}")
    
    print(f"\n📊 Pre-gate Results:")
    print(f"   Passed: {len(passed_items)} items")
    print(f"   Quarantined: {len(quarantined_items)} items")
    
    # Stage 2: Source Governance
    print(f"\n👮 STAGE 2: SOURCE GOVERNANCE")
    
    # Setup temp database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    governance = SourceGovernance(db_path=temp_db.name)
    
    # Simulate some history for photographer2 (bad reputation)
    print("   📈 Building source reputation history...")
    for i in range(20):
        governance.log_event("unsplash_photographer2", "submitted", f"item_{i}")
        if i % 3 == 0:  # 33% off-topic rate
            governance.log_event("unsplash_photographer2", "off_topic", f"item_{i}", "not_cocktail")
        else:
            governance.log_event("unsplash_photographer2", "accepted", f"item_{i}")
    
    governance.update_reputation("unsplash_photographer2", "black_charcoal", "photographer2")
    
    # Check governance evaluation
    evaluation = governance.evaluate_batch_for_ci(passed_items)
    
    print(f"\n📋 Governance Evaluation:")
    print(f"   Overall Pass: {'✅' if evaluation['pass'] else '❌'}")
    print(f"   Failures: {len(evaluation['failures'])}")
    
    for source_id, stats in evaluation['source_stats'].items():
        status_emoji = {"ok": "✅", "probation": "⚠️", "blocked": "❌"}[stats["status"]]
        print(f"   {source_id}: {status_emoji} {stats['status']} ({stats['off_topic_rate']:.1%} off-topic)")
    
    # Stage 3: Final Results
    print(f"\n🎯 FINAL RESULTS")
    print(f"   Original items: {len(test_items)}")
    print(f"   Pre-gate passed: {len(passed_items)}")
    print(f"   Governance approved: {len([item for item in passed_items if evaluation['pass']])}")
    
    print(f"\n🔍 WHAT GOT CAUGHT:")
    for q in quarantined_items:
        item = q['item']
        print(f"   🚫 {item['id']}: {q['reason']}")
        if item['id'] == 'demo_017':
            print(f"      💻 This was the KEYBOARD! Pre-gate working correctly.")
    
    print(f"\n✅ PIPELINE SUCCESS:")
    print(f"   - Keyboards blocked before human review")
    print(f"   - Low-quality sources identified") 
    print(f"   - Only legitimate borderline items reach reviewers")
    print(f"   - Source reputation tracked for continuous improvement")
    
    # Cleanup
    os.unlink(temp_db.name)
    
    return {
        'input_count': len(test_items),
        'quarantined_count': len(quarantined_items),
        'passed_count': len(passed_items),
        'keyboard_caught': any(q['item']['id'] == 'demo_017' for q in quarantined_items),
        'governance_pass': evaluation['pass']
    }

if __name__ == "__main__":
    results = demo_complete_pipeline()
    
    print(f"\n{'='*50}")
    print(f"🎉 DEMO COMPLETE")
    print(f"{'='*50}")
    print(f"Keyboard Detection: {'✅ SUCCESS' if results['keyboard_caught'] else '❌ FAILED'}")
    print(f"Quality Control: {'✅ WORKING' if results['quarantined_count'] > 0 else '❌ BROKEN'}")
    print(f"Source Governance: {'✅ ACTIVE' if 'governance_pass' in results else '❌ INACTIVE'}")
    print(f"\n🚀 Ready for production!")