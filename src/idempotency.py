#!/usr/bin/env python3
"""
Idempotency & Versioning System - Immutable Manifests

Ensures reliable processing at scale with:
- Global IDs: item_id = sha256(url_normalized) 
- Run tracking: run_id, overlay_id, model_version
- Idempotent writes: "upsert if (item_id, run_id) not seen"
- Immutable manifests: manifest_<run_id>.jsonl with counts + checksums
- Versioned storage: datasets/snapshot_<date>/manifest_<run_id>.jsonl

Supports rollback and A/B testing with complete provenance tracking.
"""

import json
import hashlib
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import gzip
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass 
class ManifestEntry:
    """Single entry in processing manifest"""
    item_id: str
    run_id: str
    model_version: str
    overlay_id: str
    stage: str  # raw, scored, borderline, approved, quarantined
    timestamp: str
    checksum: str  # SHA256 of item content
    metadata: Dict[str, Any]
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format"""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_jsonl(cls, line: str) -> 'ManifestEntry':
        """Create from JSONL line"""
        data = json.loads(line.strip())
        return cls(**data)

@dataclass
class ManifestSummary:
    """Summary statistics for a manifest"""
    run_id: str
    model_version: str
    overlay_id: str
    total_items: int
    stage_counts: Dict[str, int]
    processing_time: float
    checksum: str  # Manifest checksum
    created_at: str
    storage_path: str

class ItemIdGenerator:
    """Generate deterministic global IDs"""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for consistent hashing"""
        if not url:
            return ""
        
        url = url.strip().lower()
        
        # Remove common query parameters that don't affect image content
        if '?' in url:
            base_url, params = url.split('?', 1)
            # Keep important params like image transforms, remove tracking
            important_params = []
            for param in params.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    # Keep size/quality parameters
                    if key in ['w', 'h', 'q', 'format', 'crop', 'fit']:
                        important_params.append(param)
            
            if important_params:
                url = f"{base_url}?{'&'.join(sorted(important_params))}"
            else:
                url = base_url
        
        # Remove trailing slash
        url = url.rstrip('/')
        
        return url
    
    @staticmethod
    def generate_item_id(url: str) -> str:
        """Generate deterministic item ID from URL"""
        normalized = ItemIdGenerator.normalize_url(url)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    @staticmethod
    def generate_run_id(prefix: str = "run") -> str:
        """Generate unique run ID"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        random_suffix = uuid.uuid4().hex[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    @staticmethod
    def compute_item_checksum(item: Dict) -> str:
        """Compute checksum of item content"""
        # Create canonical representation
        canonical = {
            'url': item.get('url', ''),
            'query': item.get('query', ''),
            'domain': item.get('domain', ''),
            # Include scores if present
            'sim_cocktail': item.get('sim_cocktail'),
            'sim_not_cocktail': item.get('sim_not_cocktail'),
            'detected_objects': sorted(item.get('detected_objects', [])),
        }
        
        # Remove None values
        canonical = {k: v for k, v in canonical.items() if v is not None}
        
        content = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]  # Short hash

class ManifestManager:
    """Manages immutable processing manifests"""
    
    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create standard directories
        (self.base_path / "manifests").mkdir(exist_ok=True)
        (self.base_path / "snapshots").mkdir(exist_ok=True)
        (self.base_path / "eval").mkdir(exist_ok=True)
        
    def create_manifest(self, run_id: str, model_version: str, overlay_id: str) -> 'ProcessingManifest':
        """Create new processing manifest"""
        manifest_path = self.base_path / "manifests" / f"manifest_{run_id}.jsonl"
        
        return ProcessingManifest(
            run_id=run_id,
            model_version=model_version, 
            overlay_id=overlay_id,
            manifest_path=manifest_path,
            manager=self
        )
    
    def load_manifest(self, run_id: str) -> Optional['ProcessingManifest']:
        """Load existing manifest"""
        manifest_path = self.base_path / "manifests" / f"manifest_{run_id}.jsonl"
        
        if not manifest_path.exists():
            return None
        
        # Read first entry to get metadata
        with open(manifest_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                first_entry = ManifestEntry.from_jsonl(first_line)
                return ProcessingManifest(
                    run_id=first_entry.run_id,
                    model_version=first_entry.model_version,
                    overlay_id=first_entry.overlay_id,
                    manifest_path=manifest_path,
                    manager=self
                )
        
        return None
    
    def list_manifests(self) -> List[ManifestSummary]:
        """List all available manifests"""
        manifests = []
        manifest_dir = self.base_path / "manifests"
        
        for manifest_file in manifest_dir.glob("manifest_*.jsonl"):
            try:
                summary = self._read_manifest_summary(manifest_file)
                manifests.append(summary)
            except Exception as e:
                logger.warning(f"Failed to read manifest {manifest_file}: {e}")
        
        return sorted(manifests, key=lambda x: x.created_at, reverse=True)
    
    def _read_manifest_summary(self, manifest_path: Path) -> ManifestSummary:
        """Read manifest summary without loading all entries"""
        entries = []
        stage_counts = Counter()
        
        with open(manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = ManifestEntry.from_jsonl(line)
                    entries.append(entry)
                    stage_counts[entry.stage] += 1
        
        if not entries:
            raise ValueError("Empty manifest")
        
        first_entry = entries[0]
        last_entry = entries[-1]
        
        # Compute processing time
        start_time = datetime.fromisoformat(first_entry.timestamp.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(last_entry.timestamp.replace('Z', '+00:00'))
        processing_time = (end_time - start_time).total_seconds()
        
        # Compute manifest checksum
        manifest_content = manifest_path.read_text()
        manifest_checksum = hashlib.sha256(manifest_content.encode()).hexdigest()[:16]
        
        return ManifestSummary(
            run_id=first_entry.run_id,
            model_version=first_entry.model_version,
            overlay_id=first_entry.overlay_id,
            total_items=len(entries),
            stage_counts=dict(stage_counts),
            processing_time=processing_time,
            checksum=manifest_checksum,
            created_at=first_entry.timestamp,
            storage_path=str(manifest_path)
        )
    
    def archive_snapshot(self, run_id: str) -> str:
        """Archive completed run as immutable snapshot"""
        manifest_path = self.base_path / "manifests" / f"manifest_{run_id}.jsonl"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        # Create snapshot directory
        snapshot_date = datetime.now().strftime('%Y%m%d')
        snapshot_dir = self.base_path / "snapshots" / f"snapshot_{snapshot_date}"
        snapshot_dir.mkdir(exist_ok=True)
        
        # Copy manifest to snapshot
        snapshot_manifest = snapshot_dir / f"manifest_{run_id}.jsonl"
        
        # Compress for storage efficiency
        with open(manifest_path, 'rb') as f_in:
            with gzip.open(f"{snapshot_manifest}.gz", 'wb') as f_out:
                f_out.write(f_in.read())
        
        logger.info(f"Archived snapshot: {snapshot_manifest}.gz")
        return str(snapshot_manifest)

class ProcessingManifest:
    """Active processing manifest for a run"""
    
    def __init__(self, run_id: str, model_version: str, overlay_id: str, 
                 manifest_path: Path, manager: ManifestManager):
        self.run_id = run_id
        self.model_version = model_version
        self.overlay_id = overlay_id
        self.manifest_path = manifest_path
        self.manager = manager
        
        self.seen_items: Set[str] = set()
        self.entries_written = 0
        self.start_time = time.time()
        
        # Load existing entries if resuming
        if manifest_path.exists():
            self._load_existing_entries()
    
    def _load_existing_entries(self):
        """Load existing entries when resuming"""
        with open(self.manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = ManifestEntry.from_jsonl(line)
                    self.seen_items.add(entry.item_id)
                    self.entries_written += 1
        
        logger.info(f"Resumed manifest with {self.entries_written} existing entries")
    
    def has_item(self, item_id: str) -> bool:
        """Check if item already processed in this run"""
        return item_id in self.seen_items
    
    def add_items(self, items: List[Dict], stage: str) -> int:
        """Add items to manifest (idempotent)"""
        new_items = 0
        
        with open(self.manifest_path, 'a') as f:
            for item in items:
                item_id = ItemIdGenerator.generate_item_id(item.get('url', ''))
                
                # Skip if already seen (idempotent)
                if item_id in self.seen_items:
                    continue
                
                checksum = ItemIdGenerator.compute_item_checksum(item)
                
                entry = ManifestEntry(
                    item_id=item_id,
                    run_id=self.run_id,
                    model_version=self.model_version,
                    overlay_id=self.overlay_id,
                    stage=stage,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    checksum=checksum,
                    metadata={
                        'domain': item.get('domain'),
                        'query': item.get('query'),
                        'score': item.get('final_score'),
                        'sim_cocktail': item.get('sim_cocktail')
                    }
                )
                
                f.write(entry.to_jsonl() + '\n')
                self.seen_items.add(item_id)
                new_items += 1
                self.entries_written += 1
        
        if new_items > 0:
            logger.info(f"Added {new_items} items to manifest (stage: {stage})")
        
        return new_items
    
    def get_summary(self) -> ManifestSummary:
        """Get current manifest summary"""
        return self.manager._read_manifest_summary(self.manifest_path)
    
    def finalize(self) -> ManifestSummary:
        """Finalize manifest and create snapshot"""
        processing_time = time.time() - self.start_time
        
        # Archive as snapshot
        snapshot_path = self.manager.archive_snapshot(self.run_id)
        
        summary = self.get_summary()
        logger.info(f"Finalized manifest: {self.entries_written} items, {processing_time:.1f}s")
        
        return summary

class IdempotentProcessor:
    """Processor with idempotency guarantees"""
    
    def __init__(self, base_path: str = "datasets"):
        self.manifest_manager = ManifestManager(base_path)
        
    def start_run(self, model_version: str, overlay_id: str, run_id: str = None) -> ProcessingManifest:
        """Start new processing run"""
        if run_id is None:
            run_id = ItemIdGenerator.generate_run_id()
        
        manifest = self.manifest_manager.create_manifest(run_id, model_version, overlay_id)
        logger.info(f"Started run {run_id} (model: {model_version}, overlay: {overlay_id})")
        
        return manifest
    
    def resume_run(self, run_id: str) -> Optional[ProcessingManifest]:
        """Resume existing run"""
        manifest = self.manifest_manager.load_manifest(run_id)
        if manifest:
            logger.info(f"Resumed run {run_id}")
        return manifest
    
    def list_runs(self) -> List[ManifestSummary]:
        """List all processing runs"""
        return self.manifest_manager.list_manifests()

def demo_idempotency():
    """Demo the idempotency and versioning system"""
    print("🔒 Idempotency & Versioning Demo")
    print("=" * 40)
    
    # Initialize processor
    processor = IdempotentProcessor("demo_datasets")
    
    # Test data
    test_items = [
        {
            'url': 'https://example.com/cocktail1.jpg',
            'query': 'blue tropical cocktail',
            'domain': 'blue_tropical',
            'sim_cocktail': 0.85,
            'final_score': 0.82
        },
        {
            'url': 'https://example.com/cocktail2.jpg', 
            'query': 'red berry cocktail',
            'domain': 'red_berry',
            'sim_cocktail': 0.75,
            'final_score': 0.73
        }
    ]
    
    # Start new run
    manifest = processor.start_run(
        model_version="clip_v1.0",
        overlay_id="overlay_v1.2"
    )
    
    print(f"📝 Started run: {manifest.run_id}")
    
    # Add items (stage 1)
    added = manifest.add_items(test_items, 'raw')
    print(f"   Added {added} raw items")
    
    # Simulate processing and add scored items
    scored_items = [
        {**item, 'processed': True, 'detected_objects': ['glass']}
        for item in test_items
    ]
    
    added = manifest.add_items(scored_items, 'scored')
    print(f"   Added {added} scored items")
    
    # Test idempotency - add same items again
    added = manifest.add_items(test_items, 'raw')
    print(f"   Idempotent add: {added} items (should be 0)")
    
    # Test ID generation
    print(f"\n🆔 ID Generation:")
    for item in test_items:
        item_id = ItemIdGenerator.generate_item_id(item['url'])
        checksum = ItemIdGenerator.compute_item_checksum(item)
        print(f"   {item['url']}")
        print(f"     ID: {item_id[:16]}...")
        print(f"     Checksum: {checksum}")
    
    # Finalize and get summary
    summary = manifest.finalize()
    
    print(f"\n📊 Final Summary:")
    print(f"   Run ID: {summary.run_id}")
    print(f"   Model: {summary.model_version}")
    print(f"   Total items: {summary.total_items}")
    print(f"   Stage counts: {summary.stage_counts}")
    print(f"   Processing time: {summary.processing_time:.1f}s")
    print(f"   Checksum: {summary.checksum}")
    
    # List all runs
    all_runs = processor.list_runs()
    print(f"\n📋 All Runs ({len(all_runs)}):")
    for run_summary in all_runs:
        print(f"   {run_summary.run_id}: {run_summary.total_items} items")
    
    print(f"\n✅ Idempotency system ready for scale!")
    
    return manifest, summary

if __name__ == "__main__":
    demo_idempotency()