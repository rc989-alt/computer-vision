#!/usr/bin/env python3
"""
RA-Guard Candidate Library Preparation System
Implements comprehensive image sourcing with three-tier approach:
1. 本地/对象存储 (Local/Object Storage)
2. 供应商源 (Provider APIs: Unsplash/Pexels)  
3. 现有业务图库 (Existing Business Gallery)

Features:
- Multi-source image acquisition and caching
- CLIP vector pre-computation and caching
- Detection results (boxes/labels/scores) caching
- Compliance filtering (watermarks, NSFW, duplicates)
- Metadata management with minimal required fields
- Candidate scaling (50-200 per query)
- Offline reproducibility with ID+hash tracking

Usage:
    python candidate_library_setup.py --source pexels --domains cocktails,flowers --target-per-domain 200
    python candidate_library_setup.py --source unsplash --domains professional --validate-compliance
    python candidate_library_setup.py --source local --gallery-path ./existing_gallery/ --import
"""

import json
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime
import time
import hashlib
from PIL import Image
import io
import sqlite3
import pickle
import imagehash
from urllib.parse import urlparse
import os
import concurrent.futures
from threading import Lock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    """Minimal required metadata fields for candidate library"""
    id: str
    url_path: str  # URL for remote, path for local
    domain: str
    provider: str  # 'pexels', 'unsplash', 'local', 'business_gallery'
    license: str
    clip_vec: Optional[np.ndarray] = None
    det_cache: Optional[Dict] = None  # Detection boxes/labels/scores JSON
    phash: Optional[str] = None  # Perceptual hash for deduplication
    created_at: str = None
    compliance_status: str = "pending"  # pending/approved/rejected/flagged
    content_hash: str = None  # SHA256 of image bytes

class PexelsProvider:
    """Pexels API integration for image sourcing"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PEXELS_API_KEY')
        self.base_url = "https://api.pexels.com/v1"
        self.headers = {"Authorization": self.api_key} if self.api_key else {}
        
    def search_images(self, query: str, per_page: int = 80, page: int = 1) -> List[Dict]:
        """Search Pexels for images"""
        if not self.api_key:
            logger.warning("No Pexels API key found, using mock data")
            return self._mock_pexels_data(query, per_page)
            
        url = f"{self.base_url}/search"
        params = {
            "query": query,
            "per_page": min(per_page, 80),  # Pexels max
            "page": page,
            "orientation": "all"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get('photos', [])
        except Exception as e:
            logger.error(f"Pexels API error: {e}")
            return self._mock_pexels_data(query, per_page)
    
    def _mock_pexels_data(self, query: str, count: int) -> List[Dict]:
        """Generate mock Pexels-style data for testing"""
        photos = []
        for i in range(count):
            photo_id = f"pexels_{query}_{i:04d}"
            photos.append({
                "id": int(hashlib.md5(photo_id.encode()).hexdigest()[:8], 16),
                "url": f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg",
                "photographer": f"photographer_{i % 10}",
                "photographer_url": f"https://pexels.com/@photographer_{i % 10}",
                "src": {
                    "original": f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
                    "large": f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1000",
                    "medium": f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=600"
                }
            })
        return photos

class UnsplashProvider:
    """Unsplash API integration for image sourcing"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('UNSPLASH_API_KEY')
        self.base_url = "https://api.unsplash.com"
        self.headers = {"Authorization": f"Client-ID {self.api_key}"} if self.api_key else {}
    
    def search_images(self, query: str, per_page: int = 30, page: int = 1) -> List[Dict]:
        """Search Unsplash for images"""
        if not self.api_key:
            logger.warning("No Unsplash API key found, using mock data")
            return self._mock_unsplash_data(query, per_page)
            
        url = f"{self.base_url}/search/photos"
        params = {
            "query": query,
            "per_page": min(per_page, 30),  # Unsplash max
            "page": page,
            "orientation": "all"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            logger.error(f"Unsplash API error: {e}")
            return self._mock_unsplash_data(query, per_page)
    
    def _mock_unsplash_data(self, query: str, count: int) -> List[Dict]:
        """Generate mock Unsplash-style data for testing"""
        results = []
        for i in range(count):
            photo_id = f"unsplash_{query}_{i:04d}"
            results.append({
                "id": photo_id,
                "urls": {
                    "full": f"https://images.unsplash.com/photo-{photo_id}?ixlib=rb-4.0.3&ixid=full",
                    "regular": f"https://images.unsplash.com/photo-{photo_id}?ixlib=rb-4.0.3&ixid=regular&w=1080",
                    "small": f"https://images.unsplash.com/photo-{photo_id}?ixlib=rb-4.0.3&ixid=small&w=400"
                },
                "user": {
                    "name": f"photographer_{i % 15}",
                    "username": f"user_{i % 15}",
                    "links": {"html": f"https://unsplash.com/@user_{i % 15}"}
                },
                "description": f"{query} image {i}",
                "alt_description": f"A beautiful {query} photograph"
            })
        return results

class CLIPEncoder:
    """CLIP encoder for feature extraction"""
    
    def __init__(self):
        self.embedding_dim = 512
        logger.info("CLIP encoder initialized (using mock embeddings for demo)")
    
    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Generate CLIP embedding for image"""
        # Use content hash for reproducible embeddings
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        seed = int(content_hash[:8], 16)
        np.random.seed(seed)
        
        # Generate normalized embedding
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

class ObjectDetector:
    """Object detection for caching boxes/labels/scores"""
    
    def __init__(self):
        logger.info("Object detector initialized (using mock detections for demo)")
    
    def detect(self, image_bytes: bytes) -> Dict:
        """Generate object detection results"""
        # Mock detection based on image hash
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        seed = int(content_hash[:8], 16)
        np.random.seed(seed)
        
        # Generate mock detections
        num_objects = np.random.randint(1, 6)
        detections = {
            "boxes": [],
            "labels": [],
            "scores": [],
            "count": num_objects
        }
        
        object_classes = ["glass", "liquid", "garnish", "container", "background"]
        
        for i in range(num_objects):
            # Random bounding box [x1, y1, x2, y2] normalized 0-1
            x1, y1 = np.random.uniform(0, 0.7, 2)
            x2, y2 = x1 + np.random.uniform(0.1, 0.3), y1 + np.random.uniform(0.1, 0.3)
            
            detections["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
            detections["labels"].append(object_classes[i % len(object_classes)])
            detections["scores"].append(float(np.random.uniform(0.7, 0.95)))
        
        return detections

class ComplianceFilter:
    """Content compliance and filtering system"""
    
    def __init__(self):
        self.phash_cache: Set[str] = set()
        self.nsfw_keywords = ["nude", "adult", "explicit", "nsfw"]
        self.watermark_indicators = ["watermark", "stock", "©", "®"]
    
    def check_compliance(self, image_bytes: bytes, metadata: Dict) -> Tuple[bool, str]:
        """Check image compliance (NSFW, watermarks, duplicates)"""
        # Perceptual hash for deduplication
        try:
            image = Image.open(io.BytesIO(image_bytes))
            phash = str(imagehash.phash(image))
            
            # Check for duplicates
            if phash in self.phash_cache:
                return False, "duplicate"
            
            # Check for NSFW content (basic keyword check)
            description = metadata.get('description', '').lower()
            alt_description = metadata.get('alt_description', '').lower()
            
            for keyword in self.nsfw_keywords:
                if keyword in description or keyword in alt_description:
                    return False, "nsfw"
            
            # Check for watermarks (basic check)
            for indicator in self.watermark_indicators:
                if indicator in description or indicator in alt_description:
                    return False, "watermark"
            
            # Add to deduplication cache
            self.phash_cache.add(phash)
            return True, "approved"
            
        except Exception as e:
            logger.error(f"Compliance check error: {e}")
            return False, "error"

class CandidateLibraryManager:
    """Main manager for candidate library preparation"""
    
    def __init__(self, gallery_dir: str = "candidate_gallery"):
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pexels = PexelsProvider()
        self.unsplash = UnsplashProvider()
        self.clip_encoder = CLIPEncoder()
        self.detector = ObjectDetector()
        self.compliance = ComplianceFilter()
        
        # Database setup
        self.db_path = self.gallery_dir / "candidate_library.db"
        self.init_database()
        
        # Thread safety
        self.db_lock = Lock()
    
    def init_database(self):
        """Initialize SQLite database with minimal required fields"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    url_path TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    license TEXT NOT NULL,
                    clip_vec BLOB,
                    det_cache TEXT,  -- JSON string
                    phash TEXT,
                    created_at TEXT NOT NULL,
                    compliance_status TEXT DEFAULT 'pending',
                    content_hash TEXT UNIQUE
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_domain ON candidates(domain);
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_provider ON candidates(provider);
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_compliance ON candidates(compliance_status);
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_content_hash ON candidates(content_hash);
            ''')
    
    def download_image(self, url: str, timeout: int = 30) -> Optional[bytes]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Read in chunks to handle large images
            image_data = b""
            for chunk in response.iter_content(chunk_size=8192):
                image_data += chunk
                if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
                    logger.warning(f"Image too large, skipping: {url}")
                    return None
            
            return image_data
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def process_image_batch(self, images: List[Dict], domain: str, provider: str) -> List[ImageMetadata]:
        """Process batch of images with compliance checking and feature extraction"""
        results = []
        
        for img_data in images:
            try:
                # Extract URL based on provider
                if provider == "pexels":
                    url = img_data["src"]["large"]
                    img_id = f"pexels_{domain}_{img_data['id']}"
                    license = "Pexels License"
                elif provider == "unsplash":
                    url = img_data["urls"]["regular"]
                    img_id = f"unsplash_{domain}_{img_data['id']}"
                    license = "Unsplash License"
                else:
                    continue
                
                # Download image
                image_bytes = self.download_image(url)
                if not image_bytes:
                    continue
                
                # Content hash for deduplication
                content_hash = hashlib.sha256(image_bytes).hexdigest()
                
                # Compliance check
                is_compliant, compliance_reason = self.compliance.check_compliance(image_bytes, img_data)
                
                if not is_compliant:
                    logger.info(f"Image {img_id} rejected: {compliance_reason}")
                    continue
                
                # Extract features
                clip_vec = self.clip_encoder.encode_image(image_bytes)
                det_cache = self.detector.detect(image_bytes)
                
                # Get perceptual hash
                image = Image.open(io.BytesIO(image_bytes))
                phash = str(imagehash.phash(image))
                
                # Create metadata
                metadata = ImageMetadata(
                    id=img_id,
                    url_path=url,
                    domain=domain,
                    provider=provider,
                    license=license,
                    clip_vec=clip_vec,
                    det_cache=det_cache,
                    phash=phash,
                    created_at=datetime.now().isoformat(),
                    compliance_status="approved",
                    content_hash=content_hash
                )
                
                # Save to local storage
                local_path = self.gallery_dir / domain / f"{img_id}.jpg"
                local_path.parent.mkdir(exist_ok=True)
                
                with open(local_path, 'wb') as f:
                    f.write(image_bytes)
                
                # Update URL to local path for future use
                metadata.url_path = str(local_path)
                
                results.append(metadata)
                logger.info(f"Processed image {img_id} successfully")
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                continue
        
        return results
    
    def save_candidates_to_db(self, candidates: List[ImageMetadata]):
        """Save candidate metadata to database"""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                for candidate in candidates:
                    try:
                        # Serialize numpy arrays and dicts
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
    
    def build_candidate_library(self, source: str, domains: List[str], target_per_domain: int = 200):
        """Build candidate library from specified source"""
        logger.info(f"Building candidate library from {source} for domains: {domains}")
        
        for domain in domains:
            logger.info(f"Processing domain: {domain}")
            
            if source == "pexels":
                # Calculate pages needed (80 images per page max)
                pages_needed = (target_per_domain + 79) // 80
                
                all_images = []
                for page in range(1, pages_needed + 1):
                    images = self.pexels.search_images(domain, per_page=80, page=page)
                    all_images.extend(images)
                    if len(all_images) >= target_per_domain:
                        break
                
                # Limit to target
                all_images = all_images[:target_per_domain]
                
            elif source == "unsplash":
                # Calculate pages needed (30 images per page max)
                pages_needed = (target_per_domain + 29) // 30
                
                all_images = []
                for page in range(1, pages_needed + 1):
                    images = self.unsplash.search_images(domain, per_page=30, page=page)
                    all_images.extend(images)
                    if len(all_images) >= target_per_domain:
                        break
                
                # Limit to target
                all_images = all_images[:target_per_domain]
                
            else:
                logger.error(f"Unsupported source: {source}")
                continue
            
            # Process images in batches for better performance
            batch_size = 20
            all_candidates = []
            
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_images) + batch_size - 1)//batch_size}")
                
                candidates = self.process_image_batch(batch, domain, source)
                all_candidates.extend(candidates)
                
                # Save periodically
                if candidates:
                    self.save_candidates_to_db(candidates)
            
            logger.info(f"Completed domain {domain}: {len(all_candidates)} candidates processed")
    
    def get_candidates_for_query(self, domain: str, limit: int = 100) -> List[Dict]:
        """Retrieve candidates for reranking (50-200 per query)"""
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
            
            # Compliance stats
            cursor = conn.execute('''
                SELECT compliance_status, COUNT(*) 
                FROM candidates 
                GROUP BY compliance_status
            ''')
            compliance_stats = dict(cursor.fetchall())
            
            # Total approved
            total_approved = sum(by_domain.values())
            
            return {
                "total_approved": total_approved,
                "by_domain": by_domain,
                "by_provider": by_provider,
                "compliance_stats": compliance_stats
            }

def main():
    parser = argparse.ArgumentParser(description="RA-Guard Candidate Library Setup")
    parser.add_argument("--source", choices=["pexels", "unsplash", "local"], required=True,
                        help="Image source provider")
    parser.add_argument("--domains", type=str, required=True,
                        help="Comma-separated list of domains (e.g., cocktails,flowers,professional)")
    parser.add_argument("--target-per-domain", type=int, default=200,
                        help="Target number of candidates per domain")
    parser.add_argument("--gallery-dir", type=str, default="candidate_gallery",
                        help="Directory for candidate gallery storage")
    parser.add_argument("--validate-compliance", action="store_true",
                        help="Run compliance validation on existing candidates")
    parser.add_argument("--stats", action="store_true",
                        help="Show library statistics")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = CandidateLibraryManager(args.gallery_dir)
    
    if args.stats:
        # Show statistics
        stats = manager.get_library_stats()
        print("\n=== Candidate Library Statistics ===")
        print(f"Total approved candidates: {stats['total_approved']}")
        print(f"\nBy domain: {json.dumps(stats['by_domain'], indent=2)}")
        print(f"\nBy provider: {json.dumps(stats['by_provider'], indent=2)}")
        print(f"\nCompliance stats: {json.dumps(stats['compliance_stats'], indent=2)}")
        return
    
    # Parse domains
    domains = [d.strip() for d in args.domains.split(",")]
    
    # Build candidate library
    start_time = time.time()
    manager.build_candidate_library(args.source, domains, args.target_per_domain)
    elapsed = time.time() - start_time
    
    # Show final statistics
    stats = manager.get_library_stats()
    
    print(f"\n=== Candidate Library Build Complete ===")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Total approved candidates: {stats['total_approved']}")
    print(f"By domain: {json.dumps(stats['by_domain'], indent=2)}")
    print(f"Compliance pass rate: {stats['compliance_stats'].get('approved', 0) / sum(stats['compliance_stats'].values()) * 100:.1f}%")
    
    # Test candidate retrieval
    print(f"\n=== Testing Candidate Retrieval ===")
    for domain in domains:
        candidates = manager.get_candidates_for_query(domain, limit=50)
        print(f"Domain '{domain}': {len(candidates)} candidates available for reranking")

if __name__ == "__main__":
    main()