#!/usr/bin/env python3
"""
Dataset Management System for Computer Vision Pipeline

This module provides reproducible dataset management with:
- Frozen snapshots of JSON + image URIs
- Domain granularity classification
- Comprehensive data schema with image metadata
- Deduplication using perceptual hashing
- Train/val/test splitting with domain balance

Schema:
- image_id: Unique identifier
- url: Original image URL
- local_path: Downloaded image path
- width, height: Image dimensions
- sha256: Content hash for exact duplicates
- phash, dhash: Perceptual hashes for near-duplicates
- clip_vec: CLIP embedding vector
- domain: Domain classification (color-based, cocktail-type, etc.)
- split: train/val/test assignment
"""

import json
import csv
import hashlib
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from urllib.parse import urlparse
import requests
from PIL import Image
import imagehash
import numpy as np
from datetime import datetime
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages dataset creation, validation, and reproducibility."""
    
    def __init__(self, base_path: str = "data/dataset"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.images_path = self.base_path / "images"
        self.images_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        self.db_path = self.metadata_path / "dataset.db"
        self.csv_path = self.metadata_path / "dataset.csv"
        self.snapshot_path = self.metadata_path / "frozen_snapshot.json"
        
        # Initialize database
        self._init_database()
        
        # Domain classification rules
        self.domain_rules = self._define_domain_rules()
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                local_path TEXT,
                width INTEGER,
                height INTEGER,
                sha256 TEXT,
                phash TEXT,
                dhash TEXT,
                clip_vec BLOB,
                domain TEXT,
                sub_domain TEXT,
                split TEXT,
                query_source TEXT,
                alt_description TEXT,
                score_baseline REAL,
                score_enhanced REAL,
                download_timestamp TEXT,
                processing_timestamp TEXT,
                metadata_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_domain ON images (domain)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_split ON images (split)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sha256 ON images (sha256)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_phash ON images (phash)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _define_domain_rules(self) -> Dict[str, Any]:
        """Define domain classification rules."""
        return {
            'color_based': {
                'pink': ['pink', 'rose', 'blush', 'salmon'],
                'golden': ['golden', 'amber', 'yellow', 'honey'],
                'blue': ['blue', 'turquoise', 'navy', 'azure'],
                'green': ['green', 'mint', 'lime', 'emerald'],
                'red': ['red', 'crimson', 'ruby', 'burgundy'],
                'clear': ['clear', 'transparent', 'crystal'],
                'purple': ['purple', 'lavender', 'violet', 'plum'],
                'orange': ['orange', 'tangerine', 'coral'],
                'black': ['black', 'dark', 'charcoal'],
                'white': ['white', 'cream', 'ivory'],
                'silver': ['silver', 'metallic', 'grey'],
                'rainbow': ['rainbow', 'layered', 'gradient', 'multi']
            },
            'cocktail_type': {
                'classic': ['martini', 'old_fashioned', 'manhattan', 'negroni'],
                'tropical': ['pina_colada', 'mai_tai', 'hurricane', 'blue_hawaii'],
                'floral': ['rose', 'lavender', 'hibiscus', 'elderflower'],
                'citrus': ['margarita', 'lemon_drop', 'gimlet', 'daiquiri'],
                'creamy': ['white_russian', 'brandy_alexander', 'mudslide'],
                'modern': ['molecular', 'foam', 'smoke', 'nitrogen'],
                'wine_based': ['sangria', 'spritz', 'wine_cocktail'],
                'seasonal': ['mulled', 'hot_toddy', 'eggnog', 'punch']
            },
            'garnish_type': {
                'fruit': ['citrus', 'berry', 'tropical', 'stone_fruit'],
                'floral': ['rose', 'lavender', 'hibiscus', 'edible_flower'],
                'herb': ['mint', 'basil', 'thyme', 'rosemary'],
                'spice': ['cinnamon', 'star_anise', 'cardamom'],
                'rim': ['salt', 'sugar', 'spice_rim'],
                'decorative': ['umbrella', 'pick', 'sparkler', 'dry_ice']
            },
            'glass_type': {
                'coupe': ['coupe', 'champagne_coupe'],
                'martini': ['martini_glass', 'cocktail_glass'],
                'rocks': ['old_fashioned', 'rocks_glass', 'lowball'],
                'highball': ['collins', 'highball', 'tumbler'],
                'wine': ['wine_glass', 'burgundy', 'bordeaux'],
                'specialty': ['hurricane', 'margarita', 'irish_coffee'],
                'shot': ['shot_glass', 'cordial']
            }
        }
    
    def classify_domain(self, query: str, description: str = "", candidate_data: Dict = None) -> Tuple[str, str]:
        """
        Classify image into domain and sub-domain.
        
        Args:
            query: Query string used to find image
            description: Alt description of image
            candidate_data: Additional candidate metadata
            
        Returns:
            Tuple of (domain, sub_domain)
        """
        text = f"{query} {description}".lower()
        
        # Primary domain classification (color-based is most reliable)
        for color, keywords in self.domain_rules['color_based'].items():
            if any(keyword in text for keyword in keywords):
                # Find sub-domain within cocktail type
                for cocktail_type, type_keywords in self.domain_rules['cocktail_type'].items():
                    if any(keyword in text for keyword in type_keywords):
                        return f"color_{color}", cocktail_type
                
                return f"color_{color}", "unspecified"
        
        # Fallback to cocktail type classification
        for cocktail_type, keywords in self.domain_rules['cocktail_type'].items():
            if any(keyword in text for keyword in keywords):
                return "cocktail_type", cocktail_type
        
        # Final fallback
        return "unclassified", "unknown"
    
    def calculate_hashes(self, image_path: str) -> Tuple[str, str, str]:
        """
        Calculate SHA256, pHash, and dHash for image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (sha256, phash, dhash)
        """
        try:
            # SHA256 for exact duplicates
            with open(image_path, 'rb') as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            
            # Perceptual hashes for near-duplicates
            image = Image.open(image_path)
            phash = str(imagehash.phash(image))
            dhash = str(imagehash.dhash(image))
            
            return sha256, phash, dhash
            
        except Exception as e:
            logger.error(f"Error calculating hashes for {image_path}: {e}")
            return "", "", ""
    
    def download_image(self, url: str, image_id: str) -> Optional[str]:
        """
        Download image from URL.
        
        Args:
            url: Image URL
            image_id: Unique image identifier
            
        Returns:
            Local file path if successful, None otherwise
        """
        try:
            # Determine file extension
            parsed_url = urlparse(url)
            extension = Path(parsed_url.path).suffix or '.jpg'
            
            local_path = self.images_path / f"{image_id}{extension}"
            
            # Skip if already exists
            if local_path.exists():
                return str(local_path)
            
            # Download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {url} -> {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract image metadata.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image metadata
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
            sha256, phash, dhash = self.calculate_hashes(image_path)
            
            return {
                'width': width,
                'height': height,
                'sha256': sha256,
                'phash': phash,
                'dhash': dhash
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'width': 0,
                'height': 0,
                'sha256': '',
                'phash': '',
                'dhash': ''
            }
    
    def create_frozen_snapshot(self, source_json: str) -> str:
        """
        Create a frozen snapshot of the dataset.
        
        Args:
            source_json: Path to source JSON file
            
        Returns:
            Path to frozen snapshot
        """
        with open(source_json, 'r') as f:
            data = json.load(f)
        
        # Add timestamp and version info
        snapshot = {
            'created_at': datetime.now().isoformat(),
            'source_file': source_json,
            'version': '1.0.0',
            'total_queries': 0,
            'total_images': 0,
            'data': data
        }
        
        # Count queries and images
        if 'inspirations' in data:
            snapshot['total_queries'] = len(data['inspirations'])
            snapshot['total_images'] = sum(
                len(insp.get('candidates', [])) for insp in data['inspirations']
            )
        
        # Save frozen snapshot
        with open(self.snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        logger.info(f"Created frozen snapshot: {self.snapshot_path}")
        logger.info(f"Queries: {snapshot['total_queries']}, Images: {snapshot['total_images']}")
        
        return str(self.snapshot_path)
    
    def process_dataset(self, source_json: str, download_images: bool = True) -> Dict[str, Any]:
        """
        Process entire dataset from JSON file.
        
        Args:
            source_json: Path to source JSON file
            download_images: Whether to download images
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing dataset from {source_json}")
        
        # Create frozen snapshot first
        snapshot_path = self.create_frozen_snapshot(source_json)
        
        # Load data
        with open(source_json, 'r') as f:
            data = json.load(f)
        
        stats = {
            'processed': 0,
            'downloaded': 0,
            'failed_downloads': 0,
            'duplicates_sha256': 0,
            'duplicates_phash': 0,
            'domains': {},
            'splits': {'train': 0, 'val': 0, 'test': 0}
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Track existing hashes for deduplication
        cursor.execute("SELECT sha256, phash FROM images WHERE sha256 != '' OR phash != ''")
        existing_hashes = cursor.fetchall()
        existing_sha256 = {row[0] for row in existing_hashes if row[0]}
        existing_phash = {row[1] for row in existing_hashes if row[1]}
        
        for inspiration in data.get('inspirations', []):
            query = inspiration['query']
            description = inspiration.get('description', '')
            
            for candidate in inspiration.get('candidates', []):
                image_id = candidate['id']
                url = candidate['regular']
                alt_description = candidate.get('alt_description', '')
                
                # Classify domain
                domain, sub_domain = self.classify_domain(query, alt_description, candidate)
                
                # Download image if requested
                local_path = None
                if download_images:
                    local_path = self.download_image(url, image_id)
                    if local_path:
                        stats['downloaded'] += 1
                    else:
                        stats['failed_downloads'] += 1
                        continue
                
                # Process image metadata
                image_metadata = {}
                if local_path:
                    image_metadata = self.process_image(local_path)
                    
                    # Check for duplicates
                    if image_metadata['sha256'] in existing_sha256:
                        stats['duplicates_sha256'] += 1
                        logger.warning(f"SHA256 duplicate detected: {image_id}")
                        continue
                    
                    if image_metadata['phash'] in existing_phash:
                        stats['duplicates_phash'] += 1
                        logger.warning(f"Perceptual hash duplicate detected: {image_id}")
                        continue
                    
                    # Add to existing hash sets
                    existing_sha256.add(image_metadata['sha256'])
                    existing_phash.add(image_metadata['phash'])
                
                # Assign split (stratified by domain)
                split = self._assign_split(domain, stats['splits'])
                stats['splits'][split] += 1
                
                # Insert into database
                cursor.execute('''
                    INSERT OR REPLACE INTO images (
                        image_id, url, local_path, width, height, sha256, phash, dhash,
                        domain, sub_domain, split, query_source, alt_description,
                        score_baseline, score_enhanced, download_timestamp, 
                        processing_timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id, url, local_path,
                    image_metadata.get('width', 0),
                    image_metadata.get('height', 0),
                    image_metadata.get('sha256', ''),
                    image_metadata.get('phash', ''),
                    image_metadata.get('dhash', ''),
                    domain, sub_domain, split, query, alt_description,
                    candidate.get('baseline_score', 0.0),
                    candidate.get('score', 0.0),
                    datetime.now().isoformat() if local_path else None,
                    datetime.now().isoformat(),
                    json.dumps(candidate)
                ))
                
                stats['processed'] += 1
                
                # Track domain stats
                if domain not in stats['domains']:
                    stats['domains'][domain] = 0
                stats['domains'][domain] += 1
        
        conn.commit()
        conn.close()
        
        # Export to CSV
        self._export_to_csv()
        
        logger.info(f"Dataset processing complete: {stats}")
        return stats
    
    def _assign_split(self, domain: str, current_splits: Dict[str, int]) -> str:
        """Assign train/val/test split with domain stratification."""
        total = sum(current_splits.values())
        
        # Target ratios
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        
        # Current ratios
        if total == 0:
            return 'train'
        
        current_train_ratio = current_splits['train'] / total
        current_val_ratio = current_splits['val'] / total
        current_test_ratio = current_splits['test'] / total
        
        # Assign to most under-represented split
        if current_train_ratio < train_ratio:
            return 'train'
        elif current_val_ratio < val_ratio:
            return 'val'
        else:
            return 'test'
    
    def _export_to_csv(self):
        """Export database to CSV."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_id, url, local_path, width, height, sha256, phash, dhash,
                   domain, sub_domain, split, query_source, alt_description,
                   score_baseline, score_enhanced, download_timestamp, processing_timestamp
            FROM images
            ORDER BY domain, image_id
        """)
        
        rows = cursor.fetchall()
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'image_id', 'url', 'local_path', 'width', 'height', 'sha256', 'phash', 'dhash',
                'domain', 'sub_domain', 'split', 'query_source', 'alt_description',
                'score_baseline', 'score_enhanced', 'download_timestamp', 'processing_timestamp'
            ])
            writer.writerows(rows)
        
        conn.close()
        logger.info(f"Exported dataset to CSV: {self.csv_path}")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]
        
        # Domain distribution
        cursor.execute("SELECT domain, COUNT(*) FROM images GROUP BY domain")
        domain_dist = dict(cursor.fetchall())
        
        # Split distribution
        cursor.execute("SELECT split, COUNT(*) FROM images GROUP BY split")
        split_dist = dict(cursor.fetchall())
        
        # Download status
        cursor.execute("SELECT COUNT(*) FROM images WHERE local_path IS NOT NULL")
        downloaded_count = cursor.fetchone()[0]
        
        # Duplicates
        cursor.execute("SELECT COUNT(DISTINCT sha256) FROM images WHERE sha256 != ''")
        unique_sha256 = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT phash) FROM images WHERE phash != ''")
        unique_phash = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_images': total_images,
            'downloaded_images': downloaded_count,
            'download_rate': downloaded_count / total_images if total_images > 0 else 0,
            'domain_distribution': domain_dist,
            'split_distribution': split_dist,
            'unique_sha256_hashes': unique_sha256,
            'unique_phash_hashes': unique_phash,
            'potential_exact_duplicates': total_images - unique_sha256 if unique_sha256 > 0 else 0,
            'potential_near_duplicates': total_images - unique_phash if unique_phash > 0 else 0
        }
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset integrity and consistency."""
        issues = []
        warnings = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for missing local files
        cursor.execute("SELECT image_id, local_path FROM images WHERE local_path IS NOT NULL")
        for image_id, local_path in cursor.fetchall():
            if not os.path.exists(local_path):
                issues.append(f"Missing local file: {image_id} -> {local_path}")
        
        # Check for orphaned files
        cursor.execute("SELECT local_path FROM images WHERE local_path IS NOT NULL")
        db_paths = {row[0] for row in cursor.fetchall()}
        
        for img_file in self.images_path.glob("*"):
            if str(img_file) not in db_paths:
                warnings.append(f"Orphaned file: {img_file}")
        
        # Check split balance
        cursor.execute("SELECT split, COUNT(*) FROM images GROUP BY split")
        split_counts = dict(cursor.fetchall())
        total = sum(split_counts.values())
        
        for split, count in split_counts.items():
            ratio = count / total
            if split == 'train' and (ratio < 0.6 or ratio > 0.8):
                warnings.append(f"Train split imbalanced: {ratio:.2%} (expected ~70%)")
            elif split in ['val', 'test'] and (ratio < 0.1 or ratio > 0.25):
                warnings.append(f"{split.capitalize()} split imbalanced: {ratio:.2%} (expected ~15%)")
        
        conn.close()
        
        return {
            'issues': issues,
            'warnings': warnings,
            'is_valid': len(issues) == 0
        }
    
    def load_dataset(self, use_overlay: bool = True) -> List[Dict[str, Any]]:
        """
        Load dataset with optional overlay corrections.
        
        Args:
            use_overlay: If True, apply overlay corrections on top of frozen snapshot
            
        Returns:
            List of dataset items with metadata
        """
        if use_overlay:
            # Use overlay loader for clean dataset
            try:
                from overlay_loader import OverlayDatasetLoader
                overlay_loader = OverlayDatasetLoader(str(self.base_path))
                clean_dataset = overlay_loader.load_clean_dataset()
                
                # Convert to flat list of items
                items = []
                for inspiration in clean_dataset['inspirations']:
                    query = inspiration['query']
                    for candidate in inspiration['candidates']:
                        item = candidate.copy()
                        item['query'] = query
                        item['description'] = inspiration.get('description', '')
                        items.append(item)
                
                logger.info(f"Loaded {len(items)} items with overlay corrections")
                return items
                
            except ImportError:
                logger.warning("Overlay loader not available, falling back to frozen snapshot")
                use_overlay = False
        
        if not use_overlay:
            # Load original frozen snapshot
            if not self.snapshot_path.exists():
                logger.error("No frozen snapshot found")
                return []
                
            with open(self.snapshot_path, 'r') as f:
                snapshot = json.load(f)
            
            # Convert to flat list of items
            items = []
            for inspiration in snapshot['data']['inspirations']:
                query = inspiration['query']
                for candidate in inspiration['candidates']:
                    item = candidate.copy()
                    item['query'] = query
                    item['description'] = inspiration.get('description', '')
                    items.append(item)
            
            logger.info(f"Loaded {len(items)} items from frozen snapshot (no overlay)")
            return items

def main():
    """Main function for dataset management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Management System')
    parser.add_argument('--source', required=True, help='Source JSON file')
    parser.add_argument('--download', action='store_true', help='Download images')
    parser.add_argument('--validate', action='store_true', help='Validate dataset')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    dm = DatasetManager()
    
    if args.source:
        # Process dataset
        stats = dm.process_dataset(args.source, download_images=args.download)
        print(f"Processing complete: {stats}")
    
    if args.stats:
        # Show statistics
        stats = dm.get_dataset_stats()
        print("\n=== Dataset Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    if args.validate:
        # Validate dataset
        validation = dm.validate_dataset()
        print(f"\n=== Dataset Validation ===")
        print(f"Valid: {validation['is_valid']}")
        if validation['issues']:
            print("Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

if __name__ == "__main__":
    main()