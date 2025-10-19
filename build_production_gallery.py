#!/usr/bin/env python3
"""
RA-Guard Production Gallery Setup
Scales from 25 → 1,000 → 3,000 candidates with quality gates and deduplication

Target Architecture:
- Floor: ≥200 images (end-to-end testing)
- Pilot: ≥1,000 images (reliable nDCG@10)
- Validation: ≈3,000 images (300 per domain × 3 domains)

Quality Gates:
- Resolution ≥ 512px shorter side
- License compliance
- pHash deduplication
- CLIP + detection pre-computation
"""

import os
import sys
from pathlib import Path
import sqlite3
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

# Add scripts directory to path for imports
sys.path.append('scripts')
from candidate_library_setup import PexelsProvider, UnsplashProvider, ImageProcessor

@dataclass
class GalleryTarget:
    """Target gallery configuration"""
    domain: str
    target_count: int
    min_resolution: int = 512
    sources: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = ['pexels', 'unsplash']

class ProductionGalleryBuilder:
    """Builds production-scale candidate galleries with quality gates"""
    
    def __init__(self, gallery_dir: str = "production_candidate_gallery"):
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(exist_ok=True)
        
        self.db_path = self.gallery_dir / "candidate_library.db"
        self.image_processor = ImageProcessor()
        
        # Initialize providers
        self.providers = {
            'pexels': PexelsProvider(),
            'unsplash': UnsplashProvider()
        }
        
        self._init_database()
    
    def _init_database(self):
        """Initialize production database with quality tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                url_path TEXT NOT NULL,
                domain TEXT NOT NULL,
                provider TEXT NOT NULL,
                license TEXT DEFAULT 'unknown',
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT,
                perceptual_hash TEXT,
                clip_vec BLOB,
                det_cache TEXT,
                quality_score REAL DEFAULT 0.0,
                pass_license INTEGER DEFAULT 1,
                pass_resolution INTEGER DEFAULT 0,
                pass_dedup INTEGER DEFAULT 0,
                pass_quality INTEGER DEFAULT 0
            )
        ''')
        
        # Index for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain_provider ON candidates(domain, provider)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_phash ON candidates(perceptual_hash)')
        
        conn.commit()
        conn.close()
    
    def build_target_gallery(self, targets: List[GalleryTarget]) -> Dict[str, int]:
        """Build gallery to meet target specifications"""
        
        print("🎯 PRODUCTION RA-GUARD GALLERY SETUP")
        print("=" * 50)
        
        total_target = sum(t.target_count for t in targets)
        print(f"📊 Target: {total_target:,} images across {len(targets)} domains")
        
        for target in targets:
            print(f"   • {target.domain}: {target.target_count} images (≥{target.min_resolution}px)")
        
        results = {}
        
        for target in targets:
            print(f"\n🔄 Building {target.domain} domain...")
            domain_count = self._build_domain(target)
            results[target.domain] = domain_count
            
            print(f"   ✅ {target.domain}: {domain_count}/{target.target_count} collected")
        
        # Final validation
        total_collected = sum(results.values())
        print(f"\n📈 GALLERY BUILD COMPLETE")
        print(f"   • Total collected: {total_collected:,}/{total_target:,} images")
        
        if total_collected >= 200:
            print(f"   ✅ Floor met: ≥200 for end-to-end testing")
        if total_collected >= 1000:
            print(f"   ✅ Pilot ready: ≥1,000 for reliable nDCG@10")
        if total_collected >= 3000:
            print(f"   ✅ Validation ready: ≈3,000 for credible evaluation")
            
        return results
    
    def _build_domain(self, target: GalleryTarget) -> int:
        """Build single domain with quality gates"""
        
        domain_dir = self.gallery_dir / target.domain
        domain_dir.mkdir(exist_ok=True)
        
        # Check current count
        current_count = self._get_domain_count(target.domain)
        if current_count >= target.target_count:
            print(f"   ✅ {target.domain} already has {current_count} images")
            return current_count
        
        needed = target.target_count - current_count
        print(f"   📥 Need {needed} more images for {target.domain}")
        
        collected = 0
        
        # Collect from each source
        for source in target.sources:
            if collected >= needed:
                break
                
            print(f"   🌐 Collecting from {source}...")
            
            provider = self.providers.get(source)
            if not provider:
                print(f"   ❌ Unknown provider: {source}")
                continue
            
            # Collect batch
            source_needed = min(needed - collected, target.target_count // len(target.sources) + 50)
            source_collected = self._collect_from_source(
                provider, target.domain, source_needed, target.min_resolution, domain_dir
            )
            
            collected += source_collected
            print(f"   ✅ {source}: +{source_collected} images")
        
        return current_count + collected
    
    def _collect_from_source(self, provider, domain: str, count: int, min_res: int, 
                           domain_dir: Path) -> int:
        """Collect images from single source with quality gates"""
        
        collected = 0
        batch_size = 20
        
        for batch_start in range(0, count, batch_size):
            batch_count = min(batch_size, count - collected)
            
            try:
                # Get images from provider
                images = provider.search_images(
                    query=domain.rstrip('s'),  # Remove plural
                    count=batch_count,
                    page=batch_start // batch_size + 1
                )
                
                for img_data in images:
                    if collected >= count:
                        break
                        
                    # Quality gate pipeline
                    if self._process_candidate(img_data, domain, provider.name, 
                                             min_res, domain_dir):
                        collected += 1
                        
            except Exception as e:
                print(f"   ⚠️  Batch error: {e}")
                continue
        
        return collected
    
    def _process_candidate(self, img_data: Dict, domain: str, provider: str,
                          min_res: int, domain_dir: Path) -> bool:
        """Process single candidate through quality gates"""
        
        try:
            # Download and validate
            img_path, img_array = self.image_processor.download_and_validate_image(
                img_data['url'], domain_dir, img_data['id'], provider
            )
            
            if img_array is None:
                return False
            
            # Quality gate 1: Resolution
            height, width = img_array.shape[:2]
            if min(width, height) < min_res:
                os.remove(img_path)
                return False
            
            # Quality gate 2: Deduplication
            phash = self.image_processor.compute_perceptual_hash(img_array)
            if self._is_duplicate(phash):
                os.remove(img_path)
                return False
            
            # Quality gate 3: Basic quality score
            quality_score = self._compute_quality_score(img_array)
            if quality_score < 0.3:  # Threshold for very low quality
                os.remove(img_path)
                return False
            
            # Pre-compute features for fast reranking
            clip_vec = self.image_processor.extract_clip_features(img_array)
            det_cache = self.image_processor.extract_detection_features(img_array)
            content_hash = self.image_processor.compute_content_hash(img_array)
            
            # Store in database
            self._store_candidate(
                img_data['id'], str(img_path), domain, provider,
                img_data.get('license', 'unknown'), width, height,
                img_path.stat().st_size, content_hash, phash,
                clip_vec, det_cache, quality_score
            )
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Processing error for {img_data['id']}: {e}")
            return False
    
    def _is_duplicate(self, phash: str) -> bool:
        """Check if perceptual hash indicates duplicate"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT COUNT(*) FROM candidates WHERE perceptual_hash = ?',
            (phash,)
        )
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def _compute_quality_score(self, img_array) -> float:
        """Simple quality score based on contrast and sharpness"""
        import numpy as np
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Contrast (standard deviation)
        contrast = np.std(gray) / 255.0
        
        # Sharpness (Laplacian variance)
        from scipy import ndimage
        laplacian = ndimage.laplace(gray)
        sharpness = np.var(laplacian) / 10000.0  # Normalize
        
        # Combine metrics
        quality = min(1.0, (contrast * 0.6 + sharpness * 0.4))
        return quality
    
    def _store_candidate(self, id: str, path: str, domain: str, provider: str,
                        license: str, width: int, height: int, file_size: int,
                        content_hash: str, phash: str, clip_vec, det_cache, quality: float):
        """Store candidate with quality flags"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO candidates 
            (id, url_path, domain, provider, license, width, height, file_size,
             content_hash, perceptual_hash, clip_vec, det_cache, quality_score,
             pass_license, pass_resolution, pass_dedup, pass_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 1, ?)
        ''', (
            id, path, domain, provider, license, width, height, file_size,
            content_hash, phash, clip_vec, json.dumps(det_cache), quality,
            1 if quality >= 0.3 else 0
        ))
        
        conn.commit()
        conn.close()
    
    def _get_domain_count(self, domain: str) -> int:
        """Get current count for domain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT COUNT(*) FROM candidates WHERE domain = ?',
            (domain,)
        )
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_gallery_stats(self) -> Dict:
        """Get comprehensive gallery statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('SELECT COUNT(*) FROM candidates')
        total = cursor.fetchone()[0]
        
        # By domain
        cursor.execute('SELECT domain, COUNT(*) FROM candidates GROUP BY domain')
        by_domain = dict(cursor.fetchall())
        
        # By provider
        cursor.execute('SELECT provider, COUNT(*) FROM candidates GROUP BY provider')
        by_provider = dict(cursor.fetchall())
        
        # Quality gates
        cursor.execute('''
            SELECT 
                SUM(pass_license) as license_pass,
                SUM(pass_resolution) as resolution_pass,
                SUM(pass_dedup) as dedup_pass,
                SUM(pass_quality) as quality_pass,
                AVG(quality_score) as avg_quality
            FROM candidates
        ''')
        
        quality_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total': total,
            'by_domain': by_domain,
            'by_provider': by_provider,
            'quality_gates': {
                'license_pass': quality_stats[0] or 0,
                'resolution_pass': quality_stats[1] or 0,
                'dedup_pass': quality_stats[2] or 0,
                'quality_pass': quality_stats[3] or 0,
                'avg_quality': quality_stats[4] or 0.0
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Build Production RA-Guard Gallery")
    parser.add_argument("--scale", choices=['floor', 'pilot', 'validation'], 
                       default='pilot', help="Target scale")
    parser.add_argument("--domains", nargs='+', 
                       default=['cocktails', 'flowers', 'professional'],
                       help="Domains to collect")
    parser.add_argument("--gallery-dir", default="production_candidate_gallery",
                       help="Gallery directory")
    
    args = parser.parse_args()
    
    # Define scale targets
    scale_configs = {
        'floor': 200,      # End-to-end testing
        'pilot': 1000,     # Reliable nDCG@10 
        'validation': 3000 # Credible evaluation
    }
    
    total_target = scale_configs[args.scale]
    per_domain = total_target // len(args.domains)
    
    # Build targets
    targets = [
        GalleryTarget(
            domain=domain,
            target_count=per_domain,
            min_resolution=512,
            sources=['pexels', 'unsplash']
        )
        for domain in args.domains
    ]
    
    # Build gallery
    builder = ProductionGalleryBuilder(args.gallery_dir)
    results = builder.build_target_gallery(targets)
    
    # Show final stats
    stats = builder.get_gallery_stats()
    print(f"\n📊 FINAL GALLERY STATISTICS")
    print(f"   • Total: {stats['total']:,} images")
    print(f"   • Average quality: {stats['quality_gates']['avg_quality']:.2f}")
    print(f"   • Pass rates: Resolution {stats['quality_gates']['resolution_pass']}, "
          f"Dedup {stats['quality_gates']['dedup_pass']}, "
          f"Quality {stats['quality_gates']['quality_pass']}")
    
    # Readiness assessment
    if stats['total'] >= 3000:
        print(f"\n🚀 VALIDATION READY: {stats['total']} images for 300-query evaluation")
    elif stats['total'] >= 1000:
        print(f"\n🎯 PILOT READY: {stats['total']} images for reliable nDCG@10")
    elif stats['total'] >= 200:
        print(f"\n✅ FLOOR MET: {stats['total']} images for end-to-end testing")
    else:
        print(f"\n⚠️  MORE NEEDED: {stats['total']} < 200 minimum for testing")

if __name__ == "__main__":
    main()