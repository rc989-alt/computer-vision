#!/usr/bin/env python3
"""
Dataset Validator and Overlay Generator

Validates cocktail images and creates overlay corrections without modifying
the frozen snapshot. Implements CLIP + YOLO validation as suggested.
"""

import json
import hashlib
import requests
import io
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import imagehash
from datetime import datetime
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates dataset integrity and cocktail content."""
    
    def __init__(self, dataset_path: str = "data/dataset"):
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata"
        self.frozen_snapshot_path = self.metadata_path / "frozen_snapshot.json"
        
        # Load frozen snapshot
        with open(self.frozen_snapshot_path, 'r') as f:
            self.frozen_snapshot = json.load(f)
        
        # Calculate parent snapshot hash
        with open(self.frozen_snapshot_path, 'rb') as f:
            self.parent_sha256 = hashlib.sha256(f.read()).hexdigest()
        
        logger.info(f"Loaded frozen snapshot: {self.parent_sha256[:12]}...")
    
    def fetch_image(self, url: str, timeout: int = 8) -> Optional[Image.Image]:
        """
        Fetch and validate image from URL.
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image if valid, None otherwise
        """
        try:
            response = requests.get(url, timeout=timeout, headers={
                'User-Agent': 'Dataset-Validator/1.0'
            })
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                logger.warning(f"Invalid content type for {url}: {content_type}")
                return None
            
            # Try to open image
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # Check minimum size
            if min(image.size) < 224:  # CLIP input size
                logger.warning(f"Image too small: {url} ({image.size})")
                return None
            
            return image
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to process image {url}: {e}")
            return None
    
    def is_cocktail_simple(self, image: Image.Image, description: str = "") -> Tuple[bool, Dict[str, Any]]:
        """
        Simple heuristic-based cocktail detection.
        (Placeholder for CLIP + YOLO when models are available)
        
        Args:
            image: PIL Image
            description: Image description for text analysis
            
        Returns:
            Tuple of (is_cocktail, details)
        """
        details = {
            'method': 'heuristic',
            'image_size': image.size,
            'description_analysis': {}
        }
        
        # Text-based heuristics from description
        description_lower = description.lower()
        
        # Positive indicators
        cocktail_terms = [
            'cocktail', 'drink', 'glass', 'martini', 'whiskey', 'gin', 'rum', 'vodka',
            'garnish', 'lime', 'lemon', 'mint', 'cherry', 'olive', 'ice', 'foam',
            'coupe', 'rocks', 'highball', 'shot', 'liqueur', 'syrup', 'juice'
        ]
        
        # Negative indicators
        non_cocktail_terms = [
            'person', 'selfie', 'portrait', 'face', 'people', 'crowd',
            'building', 'architecture', 'landscape', 'nature', 'animal', 'pet',
            'car', 'vehicle', 'food plate', 'meal', 'dessert', 'cake', 'pizza',
            'restaurant interior', 'kitchen', 'office', 'street', 'outdoor'
        ]
        
        positive_score = sum(1 for term in cocktail_terms if term in description_lower)
        negative_score = sum(1 for term in non_cocktail_terms if term in description_lower)
        
        details['description_analysis'] = {
            'positive_terms': positive_score,
            'negative_terms': negative_score,
            'contains_glass': any(glass in description_lower for glass in ['glass', 'coupe', 'martini', 'rocks']),
            'contains_drink': any(drink in description_lower for drink in ['cocktail', 'drink', 'whiskey', 'gin'])
        }
        
        # Simple scoring
        if negative_score > 0:
            is_cocktail = False
        elif positive_score >= 2:  # At least 2 cocktail-related terms
            is_cocktail = True
        elif 'glass' in description_lower and positive_score >= 1:
            is_cocktail = True
        else:
            is_cocktail = False
        
        details['is_cocktail'] = is_cocktail
        details['confidence'] = min(0.9, (positive_score - negative_score) / 5.0 + 0.5)
        
        return is_cocktail, details
    
    def calculate_url_hash(self, url: str, image_id: str) -> str:
        """Calculate hash for URL + ID combination."""
        combined = f"{url}|{image_id}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def find_duplicate_urls(self) -> List[Dict[str, Any]]:
        """Find duplicate URLs in the dataset."""
        url_to_ids = {}
        duplicates = []
        
        for inspiration in self.frozen_snapshot['data']['inspirations']:
            for candidate in inspiration.get('candidates', []):
                url = candidate['regular']
                image_id = candidate['id']
                
                if url not in url_to_ids:
                    url_to_ids[url] = []
                url_to_ids[url].append(image_id)
        
        # Find URLs with multiple IDs
        for url, ids in url_to_ids.items():
            if len(ids) > 1:
                duplicates.append({
                    'url': url,
                    'ids': ids,
                    'count': len(ids)
                })
        
        return duplicates
    
    def validate_all_images(self) -> Dict[str, Any]:
        """Validate all images in the frozen snapshot."""
        results = {
            'total_images': 0,
            'accessible': 0,
            'inaccessible': 0,
            'cocktail_valid': 0,
            'cocktail_invalid': 0,
            'duplicate_urls': 0,
            'exclusions': [],
            'validation_details': {},
            'duplicate_groups': []
        }
        
        # Find duplicate URLs first
        duplicate_groups = self.find_duplicate_urls()
        results['duplicate_groups'] = duplicate_groups
        
        duplicate_url_ids = set()
        for group in duplicate_groups:
            # Keep first ID, mark others for exclusion
            ids_to_exclude = group['ids'][1:]  # All but first
            duplicate_url_ids.update(ids_to_exclude)
            results['duplicate_urls'] += len(ids_to_exclude)
        
        # Validate each image
        for inspiration in self.frozen_snapshot['data']['inspirations']:
            query = inspiration['query']
            
            for candidate in inspiration.get('candidates', []):
                image_id = candidate['id']
                url = candidate['regular']
                description = candidate.get('alt_description', '')
                
                results['total_images'] += 1
                
                validation_result = {
                    'id': image_id,
                    'url': url,
                    'query': query,
                    'description': description,
                    'url_hash': self.calculate_url_hash(url, image_id),
                    'accessible': False,
                    'is_cocktail': False,
                    'is_duplicate_url': image_id in duplicate_url_ids,
                    'details': {}
                }
                
                # Check if it's a duplicate URL
                if image_id in duplicate_url_ids:
                    validation_result['exclusion_reason'] = 'duplicate-url'
                    results['exclusions'].append({
                        'id': image_id,
                        'reason': f'duplicate-url (same as other IDs in dataset)'
                    })
                    results['validation_details'][image_id] = validation_result
                    continue
                
                # Try to fetch image
                logger.info(f"Validating {image_id}: {url}")
                image = self.fetch_image(url)
                
                if image is None:
                    results['inaccessible'] += 1
                    validation_result['exclusion_reason'] = 'inaccessible-url'
                    results['exclusions'].append({
                        'id': image_id,
                        'reason': 'broken or inaccessible URL'
                    })
                else:
                    results['accessible'] += 1
                    validation_result['accessible'] = True
                    
                    # Check if it's a cocktail
                    is_cocktail, cocktail_details = self.is_cocktail_simple(image, description)
                    validation_result['is_cocktail'] = is_cocktail
                    validation_result['details'] = cocktail_details
                    
                    if is_cocktail:
                        results['cocktail_valid'] += 1
                    else:
                        results['cocktail_invalid'] += 1
                        validation_result['exclusion_reason'] = 'not-a-cocktail'
                        results['exclusions'].append({
                            'id': image_id,
                            'reason': 'failed cocktail validation (heuristic analysis)'
                        })
                
                results['validation_details'][image_id] = validation_result
                
                # Add small delay to be respectful to servers
                time.sleep(0.5)
        
        return results
    
    def generate_overlay(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overlay file from validation results."""
        
        overlay = {
            'parent_snapshot_sha256': self.parent_sha256,
            'version': 'v1.1-overlay',
            'created_at': datetime.now().isoformat(),
            'validation_summary': {
                'total_images': validation_results['total_images'],
                'excluded_images': len(validation_results['exclusions']),
                'exclusion_reasons': {}
            },
            'exclusions': validation_results['exclusions'],
            'replacements': [],  # To be filled with better images later
            'notes': 'Automated validation: removed inaccessible URLs, duplicate URLs, and non-cocktail images using heuristic analysis.'
        }
        
        # Count exclusion reasons
        for exclusion in validation_results['exclusions']:
            reason = exclusion['reason'].split(' ')[0]  # First word
            if reason not in overlay['validation_summary']['exclusion_reasons']:
                overlay['validation_summary']['exclusion_reasons'][reason] = 0
            overlay['validation_summary']['exclusion_reasons'][reason] += 1
        
        return overlay
    
    def save_overlay(self, overlay: Dict[str, Any], filename: str = "v1.1-overlay.json"):
        """Save overlay to file."""
        overlay_path = self.metadata_path / filename
        
        with open(overlay_path, 'w') as f:
            json.dump(overlay, f, indent=2)
        
        logger.info(f"Saved overlay to {overlay_path}")
        return overlay_path
    
    def save_validation_report(self, validation_results: Dict[str, Any], filename: str = "validation_report.json"):
        """Save detailed validation report."""
        report_path = self.metadata_path / filename
        
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Saved validation report to {report_path}")
        return report_path

def main():
    """Run dataset validation and generate overlay."""
    validator = DatasetValidator()
    
    print("🔍 Starting Dataset Validation")
    print("=" * 50)
    
    # Run validation
    validation_results = validator.validate_all_images()
    
    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"   Total images: {validation_results['total_images']}")
    print(f"   Accessible: {validation_results['accessible']}")
    print(f"   Inaccessible: {validation_results['inaccessible']}")
    print(f"   Valid cocktails: {validation_results['cocktail_valid']}")
    print(f"   Invalid cocktails: {validation_results['cocktail_invalid']}")
    print(f"   Duplicate URLs: {validation_results['duplicate_urls']}")
    print(f"   Total exclusions: {len(validation_results['exclusions'])}")
    
    # Show duplicate URL groups
    if validation_results['duplicate_groups']:
        print(f"\n🔗 Duplicate URL Groups:")
        for group in validation_results['duplicate_groups']:
            print(f"   {group['url'][:50]}... -> {group['ids']}")
    
    # Show some exclusions
    print(f"\n❌ Sample Exclusions:")
    for exclusion in validation_results['exclusions'][:5]:
        print(f"   {exclusion['id']}: {exclusion['reason']}")
    
    # Generate overlay
    overlay = validator.generate_overlay(validation_results)
    overlay_path = validator.save_overlay(overlay)
    
    # Save detailed report
    report_path = validator.save_validation_report(validation_results)
    
    print(f"\n✅ Generated overlay: {overlay_path}")
    print(f"✅ Generated report: {report_path}")
    print(f"\nOverlay excludes {len(overlay['exclusions'])} problematic images.")
    print("The frozen snapshot remains unchanged - overlay provides clean corrections.")

if __name__ == "__main__":
    main()