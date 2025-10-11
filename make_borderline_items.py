#!/usr/bin/env python3
"""
Make Borderline Items

Extract borderline items from scored data for human review.
Filters items in the uncertain margin band for manual curation.

Updated with Pre-Off-Topic Gate integration to prevent keyboards and
other obvious non-cocktails from reaching human reviewers.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import hashlib
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pre_offtopic_gate import PreOffTopicGate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BorderlineExtractor:
    """Extract borderline items from scored dataset for review."""
    
    def __init__(self):
        self.domain_map = {
            # Color domains from our clean dataset
            'color_golden': 'gold_old_fashioned',
            'color_blue': 'blue_tropical', 
            'color_red': 'red_strawberry',
            'color_green': 'green_martini',
            'color_orange': 'orange_creamsicle',
            'color_black': 'black_charcoal',
            'color_pink': 'pink_cosmopolitan',
            'color_purple': 'purple_cocktail',
            'color_clear': 'clear_martini',
            'color_white': 'white_cream',
            'color_silver': 'silver_cocktail'
        }
    
    def load_clean_dataset(self) -> List[Dict[str, Any]]:
        """Load clean dataset with overlay corrections."""
        try:
            from overlay_loader import OverlayDatasetLoader
            loader = OverlayDatasetLoader()
            clean_dataset = loader.load_clean_dataset()
            
            # Convert to flat list
            items = []
            for inspiration in clean_dataset['inspirations']:
                query = inspiration['query']
                for candidate in inspiration['candidates']:
                    item = candidate.copy()
                    item['query'] = query
                    item['description'] = inspiration.get('description', '')
                    items.append(item)
            
            return items
            
        except ImportError:
            logger.error("Overlay loader not available")
            return []
    
    def simulate_scoring(self, items: List[Dict[str, Any]], 
                        low_margin: float = 0.10, 
                        high_margin: float = 0.25) -> List[Dict[str, Any]]:
        """
        Simulate CLIP scoring to generate borderline items.
        In production, this would use real CLIP scores.
        """
        import random
        import numpy as np
        
        # Set seed for reproducible simulation
        random.seed(42)
        np.random.seed(42)
        
        scored_items = []
        
        for item in items:
            # Simulate CLIP scores based on item characteristics
            query_lower = item['query'].lower()
            desc_lower = item.get('alt_description', '').lower()
            
            # Base scores - items in clean dataset should generally score well
            base_cocktail_sim = 0.75 + np.random.normal(0, 0.08)
            base_not_cocktail_sim = 0.25 + np.random.normal(0, 0.06)
            
            # Add some items to borderline range by adjusting scores
            if random.random() < 0.4:  # 40% chance to be borderline
                # Make it more uncertain
                base_cocktail_sim -= random.uniform(0.15, 0.30)
                base_not_cocktail_sim += random.uniform(0.05, 0.15)
            
            # Clip to valid range
            sim_cocktail = np.clip(base_cocktail_sim, 0.0, 1.0)
            sim_not_cocktail = np.clip(base_not_cocktail_sim, 0.0, 1.0)
            clip_margin = sim_cocktail - sim_not_cocktail
            
            # Only include items in the target margin range
            if low_margin <= clip_margin < high_margin:
                # Classify domain
                domain = self._classify_domain(query_lower, desc_lower)
                
                # Generate thumbnail URL
                thumb_url = item['regular'].replace('?', '?w=256&q=60&')
                
                # Simulate detected objects
                detected_objects = self._simulate_detection(query_lower, desc_lower)
                
                scored_item = {
                    'id': item['id'],
                    'domain': domain,
                    'query': item['query'],
                    'url': item['regular'],
                    'thumb_url': thumb_url,
                    'clip_margin': round(clip_margin, 3),
                    'sim_cocktail': round(sim_cocktail, 3),
                    'sim_not_cocktail': round(sim_not_cocktail, 3),
                    'detected_objects': detected_objects,
                    'flags': {
                        'require_glass': True,
                        'garnish_in_glass': 'garnish' in query_lower or 'peel' in query_lower,
                        'conflict_pairs': []
                    },
                    'run_id': datetime.now().isoformat(),
                    'overlay_id': f"ovl_{datetime.now().strftime('%Y_%m_%d_%H')}"
                }
                scored_items.append(scored_item)
        
        return scored_items
    
    def _classify_domain(self, query: str, description: str) -> str:
        """Classify item into domain based on query and description."""
        text = f"{query} {description}".lower()
        
        # Color-based classification
        if any(word in text for word in ['golden', 'yellow', 'amber', 'gold']):
            return 'gold_old_fashioned'
        elif any(word in text for word in ['blue', 'azure', 'tropical']):
            return 'blue_tropical'
        elif any(word in text for word in ['red', 'ruby', 'strawberry', 'berry']):
            return 'red_strawberry'
        elif any(word in text for word in ['green', 'lime', 'mint', 'olive']):
            return 'green_martini'
        elif any(word in text for word in ['orange', 'tangerine', 'peach', 'creamsicle']):
            return 'orange_creamsicle'
        elif any(word in text for word in ['black', 'charcoal', 'dark']):
            return 'black_charcoal'
        elif any(word in text for word in ['pink', 'rose', 'blush', 'cosmopolitan']):
            return 'pink_cosmopolitan'
        elif any(word in text for word in ['clear', 'crystal', 'transparent', 'martini']):
            return 'clear_martini'
        elif any(word in text for word in ['white', 'cream', 'ivory', 'vanilla']):
            return 'white_cream'
        else:
            return 'mixed_cocktail'
    
    def _simulate_detection(self, query: str, description: str) -> List[str]:
        """Simulate object detection results."""
        text = f"{query} {description}".lower()
        objects = []
        
        # Always detect glass for cocktails
        objects.append('glass')
        
        # Detect common cocktail components
        if any(word in text for word in ['ice', 'cube', 'sphere']):
            objects.append('ice')
        if any(word in text for word in ['orange', 'peel', 'twist', 'zest']):
            objects.append('orange_peel')
        if any(word in text for word in ['olive', 'garnish']):
            objects.append('olive')
        if any(word in text for word in ['rim', 'salt', 'sugar']):
            objects.append('rim')
        if any(word in text for word in ['umbrella', 'cherry', 'fruit']):
            objects.append('garnish')
        if any(word in text for word in ['mint', 'herb', 'basil']):
            objects.append('herbs')
        if any(word in text for word in ['foam', 'froth', 'cream']):
            objects.append('foam')
        
        return objects[:4]  # Limit to 4 detected objects
    
    def extract_borderline_items(self, 
                                input_path: str = None,
                                low_margin: float = 0.10,
                                high_margin: float = 0.25,
                                limit: int = 50) -> List[Dict[str, Any]]:
        """Extract borderline items for review with pre-off-topic gate filtering."""
        
        if input_path:
            # Load from existing scored data (production mode)
            with open(input_path, 'r') as f:
                items = json.load(f)
            
            # Filter by margin
            candidate_items = [
                item for item in items 
                if low_margin <= item.get('clip_margin', 0) < high_margin
            ]
        else:
            # Simulate from clean dataset (demo mode)
            logger.info("Simulating borderline items from clean dataset")
            clean_items = self.load_clean_dataset()
            candidate_items = self.simulate_scoring(clean_items, low_margin, high_margin)
        
        # Apply pre-off-topic gate to filter obvious non-cocktails
        logger.info(f"Applying pre-off-topic gate to {len(candidate_items)} candidate items")
        gate = PreOffTopicGate()
        borderline = []
        quarantined = []
        
        for item in candidate_items:
            # Extract CLIP similarities and detections
            sims = {
                'cocktail': item.get('sim_cocktail', 0.0),
                'not_cocktail': item.get('sim_not_cocktail', 0.0)
            }
            detections = item.get('detected_objects', [])
            
            # Evaluate with pre-gate
            gate_result = gate.evaluate(item, sims, detections)
            
            if gate_result.discard:
                logger.info(f"Quarantined {item['id']}: {gate_result.reason}")
                quarantined.append({
                    'item': item,
                    'reason': gate_result.reason,
                    'details': gate_result.details
                })
            else:
                borderline.append(item)
        
        logger.info(f"Pre-gate results: {len(borderline)} items for review, {len(quarantined)} quarantined")
        
        # Sort by margin (most suspicious first)
        borderline.sort(key=lambda x: x.get('clip_margin', 0))
        
        # Limit to requested number
        limited = borderline[:limit]
        
        logger.info(f"Extracted {len(limited)} borderline items (margin {low_margin}-{high_margin})")
        return limited
    
    def export_items(self, items: List[Dict[str, Any]], output_path: str):
        """Export borderline items to JSON file."""
        
        with open(output_path, 'w') as f:
            json.dump(items, f, indent=2)
        
        logger.info(f"Exported {len(items)} borderline items to {output_path}")
        
        # Print summary by domain
        domain_counts = {}
        for item in items:
            domain = item['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print(f"\n📊 Borderline Items Summary:")
        print(f"   Total items: {len(items)}")
        print(f"   Margin range: {min(item['clip_margin'] for item in items):.3f} - {max(item['clip_margin'] for item in items):.3f}")
        print(f"   Domain distribution:")
        for domain, count in sorted(domain_counts.items()):
            print(f"     {domain}: {count}")

def main():
    """CLI interface for borderline item extraction."""
    parser = argparse.ArgumentParser(description='Extract borderline items for human review')
    parser.add_argument('--input', help='Input scored JSON file (optional - will simulate if not provided)')
    parser.add_argument('--low', type=float, default=0.10, help='Low margin threshold')
    parser.add_argument('--high', type=float, default=0.25, help='High margin threshold') 
    parser.add_argument('--out', default='review/items.json', help='Output JSON file')
    parser.add_argument('--limit', type=int, default=50, help='Maximum items to extract')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract borderline items
    extractor = BorderlineExtractor()
    items = extractor.extract_borderline_items(
        input_path=args.input,
        low_margin=args.low,
        high_margin=args.high,
        limit=args.limit
    )
    
    # Export to file
    extractor.export_items(items, str(output_path))
    
    print(f"\n✅ Ready for review!")
    print(f"   1. Open {output_path.parent}/index.html in browser")
    print(f"   2. Review {len(items)} borderline items")
    print(f"   3. Export decisions and overlay patch")

if __name__ == "__main__":
    main()