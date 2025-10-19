#!/usr/bin/env python3
"""
RA-Guard Local Gallery Import System
Import existing gallery images into candidate library system with proper metadata

Usage:
    python import_local_gallery.py --source-dir gallery_300 --target-dir candidate_gallery
    python import_local_gallery.py --source-dir production_gallery --validate-compliance
"""

import json
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import argparse
from datetime import datetime
import hashlib
from PIL import Image
import sqlite3
import pickle
import imagehash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    """Minimal required metadata fields for candidate library"""
    id: str
    url_path: str  # Local file path
    domain: str
    provider: str  # 'local_gallery'
    license: str
    clip_vec: Optional[np.ndarray] = None
    det_cache: Optional[Dict] = None
    phash: Optional[str] = None
    created_at: str = None
    compliance_status: str = "approved"  # Pre-approved since from existing gallery
    content_hash: str = None

class CLIPEncoder:
    """CLIP encoder for feature extraction"""
    
    def __init__(self):
        self.embedding_dim = 512
        logger.info("CLIP encoder initialized (using mock embeddings for demo)")
    
    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Generate CLIP embedding for image"""
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        seed = int(content_hash[:8], 16)
        np.random.seed(seed)
        
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

class ObjectDetector:
    """Object detection for caching boxes/labels/scores"""
    
    def __init__(self):
        logger.info("Object detector initialized (using mock detections for demo)")
    
    def detect(self, image_bytes: bytes) -> Dict:
        """Generate object detection results"""
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        seed = int(content_hash[:8], 16)
        np.random.seed(seed)
        
        num_objects = np.random.randint(1, 6)
        detections = {
            "boxes": [],
            "labels": [],
            "scores": [],
            "count": num_objects
        }
        
        object_classes = ["glass", "liquid", "garnish", "container", "background"]
        
        for i in range(num_objects):
            x1, y1 = np.random.uniform(0, 0.7, 2)
            x2, y2 = x1 + np.random.uniform(0.1, 0.3), y1 + np.random.uniform(0.1, 0.3)
            
            detections["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
            detections["labels"].append(object_classes[i % len(object_classes)])
            detections["scores"].append(float(np.random.uniform(0.7, 0.95)))
        
        return detections

class LocalGalleryImporter:
    """Import existing gallery images into candidate library"""
    
    def __init__(self, target_dir: str = "candidate_gallery"):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.clip_encoder = CLIPEncoder()
        self.detector = ObjectDetector()
        
        # Database setup
        self.db_path = self.target_dir / "candidate_library.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    url_path TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    license TEXT NOT NULL,
                    clip_vec BLOB,
                    det_cache TEXT,
                    phash TEXT,
                    created_at TEXT NOT NULL,
                    compliance_status TEXT DEFAULT 'approved',
                    content_hash TEXT UNIQUE
                )
            ''')
            
            conn.execute('''CREATE INDEX IF NOT EXISTS idx_domain ON candidates(domain)''')
            conn.execute('''CREATE INDEX IF NOT EXISTS idx_provider ON candidates(provider)''')
    
    def import_gallery_directory(self, source_dir: Path):
        """Import all images from a gallery directory structure"""
        logger.info(f"Importing gallery from: {source_dir}")
        
        # Look for domain directories
        domain_dirs = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not domain_dirs:
            # If no subdirectories, treat as single domain
            self._import_domain_images(source_dir, "mixed", source_dir.name)
        else:
            # Import each domain directory
            for domain_dir in domain_dirs:
                if domain_dir.name in ['validation_results', 'variations']:
                    continue  # Skip utility directories
                    
                self._import_domain_images(domain_dir, domain_dir.name, "local_gallery")
    
    def _import_domain_images(self, domain_dir: Path, domain_name: str, provider: str):
        """Import images from a specific domain directory"""
        logger.info(f"Processing domain: {domain_name}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(domain_dir.glob(f"*{ext}")))
            image_files.extend(list(domain_dir.glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_files)} images in {domain_name}")
        
        candidates = []
        
        for img_path in image_files:
            try:
                # Read image
                with open(img_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Content hash
                content_hash = hashlib.sha256(image_bytes).hexdigest()
                
                # Extract features
                clip_vec = self.clip_encoder.encode_image(image_bytes)
                det_cache = self.detector.detect(image_bytes)
                
                # Get perceptual hash
                image = Image.open(img_path)
                phash = str(imagehash.phash(image))
                
                # Create metadata
                img_id = f"local_{domain_name}_{img_path.stem}"
                
                # Copy to target directory
                target_domain_dir = self.target_dir / domain_name
                target_domain_dir.mkdir(exist_ok=True)
                
                target_path = target_domain_dir / img_path.name
                
                # Copy file if not already there
                if not target_path.exists():
                    with open(target_path, 'wb') as f:
                        f.write(image_bytes)
                
                metadata = ImageMetadata(
                    id=img_id,
                    url_path=str(target_path),
                    domain=domain_name,
                    provider=provider,
                    license="Local Gallery License",
                    clip_vec=clip_vec,
                    det_cache=det_cache,
                    phash=phash,
                    created_at=datetime.now().isoformat(),
                    compliance_status="approved",
                    content_hash=content_hash
                )
                
                candidates.append(metadata)
                logger.info(f"Imported: {img_id}")
                
            except Exception as e:
                logger.error(f"Error importing {img_path}: {e}")
                continue
        
        # Save to database
        self._save_candidates_to_db(candidates)
        logger.info(f"Completed domain {domain_name}: {len(candidates)} candidates imported")
    
    def _save_candidates_to_db(self, candidates: List[ImageMetadata]):
        """Save candidate metadata to database"""
        with sqlite3.connect(self.db_path) as conn:
            for candidate in candidates:
                try:
                    clip_vec_bytes = pickle.dumps(candidate.clip_vec) if candidate.clip_vec is not None else None
                    det_cache_json = json.dumps(candidate.det_cache) if candidate.det_cache else None
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO candidates 
                        (id, url_path, domain, provider, license, clip_vec, det_cache, 
                         phash, created_at, compliance_status, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        candidate.id,
                        candidate.url_path,
                        candidate.domain,
                        candidate.provider,
                        candidate.license,
                        clip_vec_bytes,
                        det_cache_json,
                        candidate.phash,
                        candidate.created_at,
                        candidate.compliance_status,
                        candidate.content_hash
                    ))
                except Exception as e:
                    logger.error(f"Error saving candidate {candidate.id}: {e}")
    
    def get_library_stats(self) -> Dict:
        """Get candidate library statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total candidates by domain
            cursor = conn.execute('''
                SELECT domain, COUNT(*) 
                FROM candidates 
                WHERE compliance_status = 'approved'
                GROUP BY domain
            ''')
            by_domain = dict(cursor.fetchall())
            
            # Total candidates by provider
            cursor = conn.execute('''
                SELECT provider, COUNT(*) 
                FROM candidates 
                WHERE compliance_status = 'approved'
                GROUP BY provider
            ''')
            by_provider = dict(cursor.fetchall())
            
            # Total approved
            total_approved = sum(by_domain.values())
            
            return {
                "total_approved": total_approved,
                "by_domain": by_domain,
                "by_provider": by_provider
            }
    
    def get_candidates_for_query(self, domain: str, limit: int = 100) -> List[Dict]:
        """Retrieve candidates for reranking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, url_path, domain, provider, phash, content_hash
                FROM candidates 
                WHERE domain = ? AND compliance_status = 'approved'
                ORDER BY RANDOM()
                LIMIT ?
            ''', (domain, limit))
            
            candidates = []
            for row in cursor.fetchall():
                candidates.append({
                    "id": row[0],
                    "url_path": row[1],
                    "domain": row[2],
                    "provider": row[3],
                    "phash": row[4],
                    "content_hash": row[5]
                })
            
            return candidates

def main():
    parser = argparse.ArgumentParser(description="Import local gallery into candidate library")
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Source gallery directory to import")
    parser.add_argument("--target-dir", type=str, default="candidate_gallery",
                        help="Target candidate library directory")
    parser.add_argument("--stats", action="store_true",
                        help="Show library statistics after import")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return
    
    # Initialize importer
    importer = LocalGalleryImporter(args.target_dir)
    
    # Import gallery
    start_time = datetime.now()
    importer.import_gallery_directory(source_dir)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Show statistics
    stats = importer.get_library_stats()
    
    print(f"\n=== Local Gallery Import Complete ===")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Total imported candidates: {stats['total_approved']}")
    print(f"By domain: {json.dumps(stats['by_domain'], indent=2)}")
    print(f"By provider: {json.dumps(stats['by_provider'], indent=2)}")
    
    # Test candidate retrieval
    print(f"\n=== Testing Candidate Retrieval ===")
    for domain in stats['by_domain'].keys():
        candidates = importer.get_candidates_for_query(domain, limit=50)
        print(f"Domain '{domain}': {len(candidates)} candidates available for reranking")
        
        # Show sample candidates
        if candidates:
            print(f"  Sample candidates:")
            for i, candidate in enumerate(candidates[:3]):
                print(f"    {i+1}. ID: {candidate['id']}, Hash: {candidate['content_hash'][:8]}...")

if __name__ == "__main__":
    main()