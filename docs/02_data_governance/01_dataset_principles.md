# Dataset Management Principles & Configuration

## 0) Prep & Principles

### Reproducibility
- **Frozen Snapshots**: All datasets are versioned with immutable snapshots
- **Deterministic Processing**: Same input always produces same output
- **Content Hashing**: SHA256 for exact duplicates, pHash/dHash for near-duplicates
- **URL Preservation**: Original image URLs maintained for reproducibility

### Domain Granularity

We enforce uniqueness and balance at multiple domain levels:

#### Primary Domains (Highest Priority)
1. **Color-Based** (`color_*`): Most visually distinguishable
   - pink, golden, blue, green, red, clear, purple, orange, black, white, silver, rainbow
   
2. **Cocktail Type** (`cocktail_type`): Semantic categories
   - classic, tropical, floral, citrus, creamy, modern, wine_based, seasonal

3. **Glass Type** (`glass_*`): Structural categories  
   - coupe, martini, rocks, highball, wine, specialty, shot

#### Sub-Domains (Secondary Classification)
- Within each primary domain, items are further classified by cocktail type
- Example: `color_pink` + `floral` = pink floral cocktails
- Enables fine-grained analysis while maintaining high-level balance

### Data Schema

#### Core Fields (Required)
```sql
image_id TEXT PRIMARY KEY      -- Unique identifier (e.g., demo_001)
url TEXT NOT NULL             -- Original image URL  
local_path TEXT               -- Downloaded image path
width INTEGER                 -- Image width in pixels
height INTEGER                -- Image height in pixels
sha256 TEXT                   -- Content hash for exact duplicates
phash TEXT                    -- Perceptual hash for near-duplicates  
dhash TEXT                    -- Difference hash for near-duplicates
domain TEXT                   -- Primary domain classification
sub_domain TEXT               -- Secondary domain classification
split TEXT                    -- train/val/test assignment
```

#### Extended Fields (Optional)
```sql
clip_vec BLOB                 -- CLIP embedding vector (768-dim)
query_source TEXT             -- Original query that found this image
alt_description TEXT          -- Image description/caption
score_baseline REAL           -- Baseline CLIP-only score
score_enhanced REAL           -- Enhanced region-control score
download_timestamp TEXT       -- When image was downloaded
processing_timestamp TEXT     -- When metadata was extracted
metadata_json TEXT            -- Full candidate metadata as JSON
```

### Uniqueness Enforcement

#### Level 1: Exact Duplicates (SHA256)
- Identical files are completely deduplicated
- Only first occurrence is kept

#### Level 2: Near Duplicates (Perceptual Hash)
- Images that are visually similar (crops, filters, etc.)
- Hamming distance threshold applied to pHash/dHash
- Best quality version retained

#### Level 3: Domain Balance
- Within each domain, maintain representative diversity
- Avoid over-sampling from any single query or source
- Stratified sampling across sub-domains

### Split Strategy

#### Stratified Domain Splitting
- **Train**: 70% - Balanced across all domains
- **Val**: 15% - Same domain distribution as train  
- **Test**: 15% - Same domain distribution as train

#### Constraints
- No domain should be entirely in one split
- Minimum 2 images per domain per split (when possible)
- Query-level splitting (all candidates from same query in same split)

### Quality Controls

#### Image Requirements
- Minimum resolution: 224x224 (CLIP input size)
- Maximum file size: 10MB
- Supported formats: JPEG, PNG, WebP
- Valid image headers and readable by PIL

#### Content Requirements  
- Must contain visible cocktail/drink
- Glass or container must be present
- No explicit content or watermarks
- Clear enough for visual analysis

#### Metadata Requirements
- Valid URL format
- Non-empty alt_description
- Reasonable score range [0, 1]
- Consistent image_id format

### Storage Structure

```
data/dataset/
├── images/                    # Downloaded images
│   ├── demo_001.jpg
│   ├── demo_002.jpg
│   └── ...
├── metadata/
│   ├── dataset.db            # SQLite database (primary)
│   ├── dataset.csv           # CSV export (backup)
│   ├── frozen_snapshot.json  # Immutable snapshot
│   └── processing_log.txt    # Processing history
└── splits/
    ├── train.json            # Train split image IDs
    ├── val.json              # Validation split image IDs
    └── test.json             # Test split image IDs
```

### Version Control

#### Snapshot Versioning
- `v1.0.0`: Initial demo dataset (15 queries, 30 images)
- `v1.1.0`: Extended dataset with balanced domains
- `v2.0.0`: Production dataset with full coverage

#### Schema Evolution
- Backward compatible database migrations
- CSV exports preserve column ordering
- New fields added as optional extensions

### Validation Checks

#### Automated Validation
1. **File Integrity**: All local_path files exist and are readable
2. **Hash Consistency**: SHA256 matches file content
3. **Domain Balance**: No domain has <5% or >40% of total images
4. **Split Balance**: Train/val/test ratios within acceptable ranges
5. **Metadata Completeness**: Required fields are non-null
6. **URL Accessibility**: Sample URLs are still reachable

#### Manual Review Points
- Domain classification accuracy (sample review)
- Image quality assessment (resolution, clarity)
- Content appropriateness (family-friendly)
- Description accuracy (matches visual content)

This configuration ensures reproducible, balanced, and high-quality datasets for reliable computer vision experiments.