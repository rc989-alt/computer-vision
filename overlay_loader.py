#!/usr/bin/env python3
"""
Overlay Dataset Loader

Loads datasets with overlay corrections applied on top of frozen snapshots.
Implements the layered correction approach without modifying immutable data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OverlayDatasetLoader:
    """Loads datasets with overlay corrections applied."""
    
    def __init__(self, dataset_path: str = "data/dataset"):
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata"
        self.frozen_snapshot_path = self.metadata_path / "frozen_snapshot.json"
        
        # Load frozen snapshot
        with open(self.frozen_snapshot_path, 'r') as f:
            self.frozen_snapshot = json.load(f)
        
        # Calculate snapshot hash for validation
        with open(self.frozen_snapshot_path, 'rb') as f:
            self.snapshot_hash = hashlib.sha256(f.read()).hexdigest()
        
        logger.info(f"Loaded frozen snapshot: {self.snapshot_hash[:12]}...")
    
    def find_overlay_files(self) -> List[Path]:
        """Find all overlay files in lexicographic order."""
        overlay_files = list(self.metadata_path.glob("*-overlay.json"))
        overlay_files.sort()  # Lexicographic order
        return overlay_files
    
    def load_overlay(self, overlay_path: Path) -> Dict[str, Any]:
        """Load and validate an overlay file."""
        with open(overlay_path, 'r') as f:
            overlay = json.load(f)
        
        # Validate parent snapshot hash
        expected_hash = overlay.get('parent_snapshot_sha256', '')
        if expected_hash and expected_hash != self.snapshot_hash:
            logger.warning(f"Overlay {overlay_path.name} has mismatched parent hash")
            logger.warning(f"Expected: {expected_hash[:12]}..., Got: {self.snapshot_hash[:12]}...")
        
        logger.info(f"Loaded overlay: {overlay_path.name} (v{overlay.get('version', 'unknown')})")
        return overlay
    
    def apply_overlays(self, overlays: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply overlays to frozen snapshot data."""
        
        # Start with original data
        dataset = {
            'inspirations': [],
            'metadata': self.frozen_snapshot.get('metadata', {}),
            'applied_overlays': []
        }
        
        # Create lookup for easy access
        original_items = {}
        for inspiration in self.frozen_snapshot['data']['inspirations']:
            for candidate in inspiration.get('candidates', []):
                original_items[candidate['id']] = {
                    'candidate': candidate,
                    'query': inspiration['query'],  
                    'description': inspiration.get('description', '')
                }
        
        # Collect all exclusions and replacements
        all_exclusions = set()
        all_replacements = {}
        
        for overlay in overlays:
            # Track applied overlay
            dataset['applied_overlays'].append({
                'version': overlay.get('version', 'unknown'),
                'created_at': overlay.get('created_at', ''),
                'exclusions_count': len(overlay.get('exclusions', [])),
                'replacements_count': len(overlay.get('replacements', []))
            })
            
            # Add exclusions
            for exclusion in overlay.get('exclusions', []):
                if 'id' in exclusion:
                    all_exclusions.add(exclusion['id'])
                    logger.debug(f"Excluding {exclusion['id']}: {exclusion['reason']}")
            
            # Add replacements
            for replacement in overlay.get('replacements', []):
                old_id = replacement.get('old_id')
                new_data = replacement.get('new', {})
                if old_id and new_data:
                    all_replacements[old_id] = replacement
                    logger.debug(f"Replacing {old_id} with {new_data.get('id', 'new_item')}")
        
        # Rebuild dataset with overlays applied
        query_groups = {}
        
        # Group original items by query (preserving structure)
        for inspiration in self.frozen_snapshot['data']['inspirations']:
            query = inspiration['query']
            if query not in query_groups:
                query_groups[query] = {
                    'query': query,
                    'description': inspiration.get('description', ''),
                    'candidates': []
                }
        
        # Process each original item
        for item_id, item_data in original_items.items():
            query = item_data['query']
            candidate = item_data['candidate']
            
            # Skip if excluded
            if item_id in all_exclusions:
                continue
            
            # Replace if needed
            if item_id in all_replacements:
                replacement = all_replacements[item_id]
                new_candidate = replacement['new'].copy()
                # Preserve original scores if not provided
                if 'score' not in new_candidate:
                    new_candidate['score'] = candidate.get('score', 0.0)
                if 'baseline_score' not in new_candidate:
                    new_candidate['baseline_score'] = candidate.get('baseline_score', 0.0)
                
                query_groups[query]['candidates'].append(new_candidate)
            else:
                # Keep original
                query_groups[query]['candidates'].append(candidate)
        
        # Convert back to inspirations format, filtering empty queries
        for query, group in query_groups.items():
            if group['candidates']:  # Only include queries with remaining candidates
                dataset['inspirations'].append(group)
        
        # Update metadata
        dataset['metadata'].update({
            'overlay_applied': True,
            'original_total_images': len(original_items),
            'final_total_images': sum(len(insp['candidates']) for insp in dataset['inspirations']),
            'total_exclusions': len(all_exclusions),
            'total_replacements': len(all_replacements)
        })
        
        return dataset
    
    def load_clean_dataset(self) -> Dict[str, Any]:
        """Load dataset with all available overlays applied."""
        
        # Find and load overlays
        overlay_files = self.find_overlay_files()
        overlays = []
        
        for overlay_file in overlay_files:
            try:
                overlay = self.load_overlay(overlay_file)
                overlays.append(overlay)
            except Exception as e:
                logger.error(f"Failed to load overlay {overlay_file}: {e}")
        
        # Apply overlays
        if overlays:
            dataset = self.apply_overlays(overlays)
            logger.info(f"Applied {len(overlays)} overlays")
        else:
            logger.warning("No overlays found, using original frozen snapshot")
            dataset = self.frozen_snapshot['data'].copy()
            dataset['applied_overlays'] = []
        
        return dataset
    
    def export_clean_manifest(self, output_path: str = None) -> str:
        """Export clean dataset as CSV manifest."""
        if output_path is None:  
            output_path = str(self.metadata_path / "clean_manifest.csv")
        
        clean_dataset = self.load_clean_dataset()
        
        import csv
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'id', 'url', 'query', 'alt_description', 'score_baseline', 'score_enhanced',
                'kept_excluded_replaced', 'domain_classification', 'validation_status'
            ])
            
            for inspiration in clean_dataset['inspirations']:
                query = inspiration['query'] 
                for candidate in inspiration['candidates']:
                    writer.writerow([
                        candidate['id'],
                        candidate['regular'],
                        query,
                        candidate.get('alt_description', ''),
                        candidate.get('baseline_score', 0.0),
                        candidate.get('score', 0.0),
                        'kept',  # All items in clean dataset are kept
                        self._classify_domain_simple(query, candidate.get('alt_description', '')),
                        'validated'
                    ])
        
        logger.info(f"Exported clean manifest to {output_path}")
        return output_path
    
    def _classify_domain_simple(self, query: str, description: str) -> str:
        """Simple domain classification for manifest."""
        text = f"{query} {description}".lower()
        
        colors = ['pink', 'golden', 'blue', 'green', 'red', 'clear', 'purple', 'orange', 'black', 'white', 'silver']
        for color in colors:
            if color in text:
                return f"color_{color}"
        
        return "unclassified"
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics for the clean dataset."""
        clean_dataset = self.load_clean_dataset()
        
        total_images = sum(len(insp['candidates']) for insp in clean_dataset['inspirations'])
        total_queries = len(clean_dataset['inspirations'])
        
        # Domain distribution
        domain_counts = {}
        for inspiration in clean_dataset['inspirations']:
            for candidate in inspiration['candidates']:
                domain = self._classify_domain_simple(inspiration['query'], candidate.get('alt_description', ''))
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            'total_queries': total_queries,
            'total_images': total_images,
            'avg_candidates_per_query': total_images / total_queries if total_queries > 0 else 0,
            'domain_distribution': domain_counts,
            'overlay_metadata': clean_dataset.get('metadata', {}),
            'applied_overlays': clean_dataset.get('applied_overlays', [])
        }

def main():
    """CLI interface for overlay dataset loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Overlay Dataset Loader')
    parser.add_argument('--stats', action='store_true', help='Show clean dataset statistics')
    parser.add_argument('--export', help='Export clean manifest to CSV file')
    parser.add_argument('--inspect', help='Inspect specific image ID in clean dataset')
    
    args = parser.parse_args()
    
    loader = OverlayDatasetLoader()
    
    if args.stats:
        stats = loader.get_dataset_stats()
        print("📊 Clean Dataset Statistics:")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Avg candidates per query: {stats['avg_candidates_per_query']:.1f}")
        print(f"   Applied overlays: {len(stats['applied_overlays'])}")
        
        print("\n🎯 Domain Distribution:")
        for domain, count in sorted(stats['domain_distribution'].items()):
            print(f"   {domain}: {count}")
        
        if stats['applied_overlays']:
            print("\n🔧 Applied Overlays:")
            for overlay in stats['applied_overlays']:
                print(f"   {overlay['version']}: -{overlay['exclusions_count']} +{overlay['replacements_count']}")
    
    elif args.export:
        manifest_path = loader.export_clean_manifest(args.export)
        print(f"✅ Exported clean manifest to {manifest_path}")
    
    elif args.inspect:
        clean_dataset = loader.load_clean_dataset()
        found = False
        
        for inspiration in clean_dataset['inspirations']:
            for candidate in inspiration['candidates']:
                if candidate['id'] == args.inspect:
                    print(f"🔍 Found {args.inspect} in clean dataset:")
                    print(f"   Query: {inspiration['query']}")
                    print(f"   URL: {candidate['regular']}")
                    print(f"   Description: {candidate.get('alt_description', 'N/A')}")
                    print(f"   Score: {candidate.get('score', 'N/A')}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"❌ Image {args.inspect} not found in clean dataset (may be excluded)")
    
    else:
        # Default: show summary
        stats = loader.get_dataset_stats()
        print(f"🗄️ Clean Dataset Loaded")
        print(f"   Original → Clean: {stats['overlay_metadata'].get('original_total_images', '?')} → {stats['total_images']} images")
        print(f"   Exclusions: {stats['overlay_metadata'].get('total_exclusions', 0)}")
        print(f"   Overlays applied: {len(stats['applied_overlays'])}")

if __name__ == "__main__":
    main()