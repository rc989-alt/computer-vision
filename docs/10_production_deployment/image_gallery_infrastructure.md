# 🖼️ RA-Guard Production Image Gallery Infrastructure

**Purpose**: Real image candidate infrastructure for RA-Guard reranking in production A/B testing  
**Critical**: RA-Guard is a **reranker** - it requires real images to score and reorder  
**Status**: ⚠️ **REQUIRED FOR PRODUCTION DEPLOYMENT**  

---

## 🎯 **ARCHITECTURE OVERVIEW**

### **What RA-Guard Actually Does**
- **Input**: Query (text/structured) + Candidate images (50-200 real images)
- **Process**: Rerank candidates using multimodal scoring
- **Output**: Reordered candidate list for nDCG@K evaluation
- **Critical**: T3-Verified +4.24 nDCG improvement was achieved **on real image galleries**

### **Production Requirements**
```yaml
query_flow:
  1. user_query: "cocktail with mint"
  2. retrieval_stage: fetch 50-200 candidate images
  3. ra_guard_reranking: score and reorder candidates
  4. display_results: top-K reranked images
  5. evaluation: nDCG@10 measurement
```

---

## 🏗️ **INFRASTRUCTURE COMPONENTS**

### **1. Image Gallery Storage**
```yaml
storage_architecture:
  primary: "s3://production-gallery/"
  structure:
    - cocktails/
      - unsplash/{image_id}.jpg
      - pexels/{image_id}.jpg
      - internal/{image_id}.jpg
    - flowers/
      - unsplash/{image_id}.jpg
      - pexels/{image_id}.jpg
      - internal/{image_id}.jpg
    - professional/
      - unsplash/{image_id}.jpg
      - pexels/{image_id}.jpg
      - internal/{image_id}.jpg

access_patterns:
  read_latency: "< 50ms P95"
  throughput: "1000 QPS peak"
  availability: "99.9%"
```

### **2. Metadata Database Schema**
```sql
CREATE TABLE image_gallery (
    id VARCHAR(64) PRIMARY KEY,           -- content_hash + provider_id
    url_path VARCHAR(512) NOT NULL,       -- s3://production-gallery/...
    domain VARCHAR(32) NOT NULL,          -- cocktails|flowers|professional
    provider VARCHAR(32) NOT NULL,        -- unsplash|pexels|internal
    license VARCHAR(128),                 -- usage/attribution flags
    clip_vec VECTOR(512),                 -- precomputed CLIP embedding
    det_cache JSON,                       -- cached detection results
    phash VARCHAR(32),                    -- perceptual hash for dedup
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_domain (domain),
    INDEX idx_provider (provider),
    INDEX idx_phash (phash)
);
```

### **3. Feature Cache Infrastructure**
```yaml
clip_cache:
  type: "redis_cluster"
  key_pattern: "clip:{image_id}"
  value: "float32[512] embedding vector"
  ttl: "7_days"
  memory: "16GB per node"

detection_cache:
  type: "redis_cluster" 
  key_pattern: "det:{image_id}"
  value: "json detection results"
  structure:
    boxes: "[[x1,y1,x2,y2], ...]"
    labels: "['object1', 'object2', ...]"
    scores: "[0.95, 0.87, ...]"
  ttl: "7_days"

candidate_cache:
  type: "application_cache"
  key_pattern: "candidates:{query_hash}"
  value: "list of candidate image_ids"
  ttl: "1_hour"
  size: "50-200 candidates per query"
```

---

## 📊 **CANDIDATE SELECTION STRATEGY**

### **Retrieval Pipeline**
```python
def get_candidates_for_query(query: str, domain: str) -> List[str]:
    """
    Retrieve candidate images for RA-Guard reranking
    Returns 50-200 candidate image IDs
    """
    # 1. Text-based retrieval (existing system)
    text_candidates = text_search(query, domain, limit=100)
    
    # 2. CLIP similarity retrieval
    query_embedding = clip_encode_text(query)
    clip_candidates = vector_search(query_embedding, domain, limit=100)
    
    # 3. Merge and deduplicate
    all_candidates = merge_dedupe(text_candidates, clip_candidates)
    
    # 4. Filter compliance (license, NSFW, watermark)
    filtered_candidates = apply_compliance_filters(all_candidates)
    
    # 5. Return 50-200 candidates for reranking
    return filtered_candidates[:200]
```

### **Domain-Specific Pools**
| Domain | Pool Size | Sources | Special Filters |
|--------|-----------|---------|-----------------|
| **Cocktails** | 10K+ images | Unsplash, Pexels, Internal | License: commercial OK |
| **Flowers** | 15K+ images | Unsplash, Pexels, Stock | Filter: no artificial/digital |
| **Professional** | 20K+ images | Internal, Stock | License: business use only |

---

## ⚡ **PERFORMANCE OPTIMIZATION**

### **Latency Targets**
```yaml
end_to_end_latency:
  candidate_retrieval: "< 100ms P95"
  feature_loading: "< 10ms P95"  # from cache
  ra_guard_scoring: "< 50ms P95"  # for 100 candidates
  total_rerank_overhead: "< 150ms P95"

throughput_targets:
  peak_qps: 1000
  concurrent_queries: 200
  candidates_per_second: 100K
```

### **Caching Strategy**
```python
# Precompute expensive operations
@cache_result(ttl=3600)
def get_clip_embedding(image_id: str) -> np.ndarray:
    """Cache CLIP embeddings for sub-ms access"""
    
@cache_result(ttl=3600) 
def get_detection_results(image_id: str) -> Dict:
    """Cache detector outputs for fast conflict detection"""

@cache_result(ttl=600)
def get_candidates_for_query(query_hash: str) -> List[str]:
    """Cache candidate lists for consistent A/B testing"""
```

---

## 🔄 **DATA INGESTION PIPELINE**

### **Multi-Source Ingestion**
```yaml
ingestion_sources:
  unsplash_api:
    endpoint: "https://api.unsplash.com/"
    rate_limit: "5000/hour"
    license: "free_for_commercial"
    domains: ["cocktails", "flowers", "professional"]
    
  pexels_api:
    endpoint: "https://api.pexels.com/"
    rate_limit: "200/hour"
    license: "free_for_commercial" 
    domains: ["cocktails", "flowers", "professional"]
    
  internal_sources:
    s3_buckets: ["s3://internal-stock/", "s3://curated-gallery/"]
    license: "full_commercial_rights"
    domains: ["professional", "cocktails"]
```

### **Processing Pipeline**
```python
def ingest_image(source_url: str, metadata: Dict) -> str:
    """
    Complete image ingestion pipeline
    Returns: image_id for gallery
    """
    # 1. Download and validate image
    image_bytes = download_with_retry(source_url)
    validate_image_format(image_bytes)
    
    # 2. Generate unique ID and check for duplicates
    content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    image_id = f"{metadata['provider']}_{content_hash}"
    
    if exists_in_gallery(image_id):
        return image_id  # Already processed
    
    # 3. Store image in S3
    s3_path = f"s3://production-gallery/{metadata['domain']}/{metadata['provider']}/{image_id}.jpg"
    upload_to_s3(image_bytes, s3_path)
    
    # 4. Precompute features
    clip_embedding = compute_clip_embedding(image_bytes)
    detection_results = run_object_detection(image_bytes)
    perceptual_hash = compute_phash(image_bytes)
    
    # 5. Store metadata and features
    store_in_database({
        'id': image_id,
        'url_path': s3_path,
        'domain': metadata['domain'],
        'provider': metadata['provider'],
        'license': metadata['license'],
        'clip_vec': clip_embedding,
        'det_cache': detection_results,
        'phash': perceptual_hash
    })
    
    # 6. Cache features for fast access
    cache_clip_embedding(image_id, clip_embedding)
    cache_detection_results(image_id, detection_results)
    
    return image_id
```

---

## 🔍 **QUALITY ASSURANCE**

### **Deduplication Strategy**
```python
def check_duplicate_images(new_image: bytes, domain: str) -> bool:
    """Prevent near-duplicate images in gallery"""
    
    # 1. Perceptual hash matching
    new_phash = compute_phash(new_image)
    existing_phashes = get_phashes_for_domain(domain)
    
    for existing_phash in existing_phashes:
        if hamming_distance(new_phash, existing_phash) < 5:
            return True  # Too similar
    
    # 2. CLIP embedding similarity
    new_clip = compute_clip_embedding(new_image)
    similarity_threshold = 0.95
    
    similar_images = vector_search(new_clip, domain, threshold=similarity_threshold)
    return len(similar_images) > 0
```

### **Content Filtering**
```python
def apply_compliance_filters(candidates: List[str]) -> List[str]:
    """Filter candidates for compliance and quality"""
    
    filtered = []
    for image_id in candidates:
        metadata = get_image_metadata(image_id)
        
        # License compliance
        if not is_license_compliant(metadata['license']):
            continue
            
        # NSFW filtering
        if is_nsfw_content(image_id):
            continue
            
        # Watermark detection
        if has_watermark(image_id):
            continue
            
        # Quality checks
        if not meets_quality_standards(image_id):
            continue
            
        filtered.append(image_id)
    
    return filtered
```

---

## 📈 **MONITORING & ANALYTICS**

### **Gallery Health Metrics**
```yaml
storage_metrics:
  - total_images_per_domain
  - storage_utilization_gb
  - image_access_patterns
  - cache_hit_rates

performance_metrics:
  - candidate_retrieval_latency_p95
  - feature_cache_latency_p95
  - gallery_query_throughput
  - error_rates_by_domain

quality_metrics:
  - duplicate_detection_rate
  - compliance_filter_rate
  - user_engagement_by_domain
  - nDCG_improvement_by_gallery_size
```

### **A/B Test Consistency**
```python
def log_candidate_sets(query: str, candidates: List[str]) -> None:
    """Log candidate sets for offline/online consistency validation"""
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'query_hash': hashlib.md5(query.encode()).hexdigest(),
        'query_text': query,
        'candidate_count': len(candidates),
        'candidate_ids': candidates,
        'retrieval_method': 'hybrid_text_clip',
        'domain_distribution': count_by_domain(candidates)
    }
    
    # Log to both real-time stream and batch storage
    send_to_kinesis(log_entry)
    write_to_s3_logs(log_entry)
```

---

## 🚀 **DEPLOYMENT CHECKLIST**

### **Gallery Infrastructure**
- [ ] **S3 buckets created with proper IAM policies**
- [ ] **Metadata database deployed and indexed**
- [ ] **Redis clusters for feature caching**
- [ ] **CDN configuration for image delivery**

### **Data Pipeline**
- [ ] **Multi-source ingestion scripts deployed**
- [ ] **CLIP embedding computation at scale**
- [ ] **Object detection batch processing**
- [ ] **Deduplication and quality filtering**

### **Performance Validation**
- [ ] **Load testing with 1000 QPS**
- [ ] **Latency validation < 150ms P95**
- [ ] **Cache hit rate > 90%**
- [ ] **Feature computation throughput validated**

### **Compliance & Security**
- [ ] **License tracking and validation**
- [ ] **NSFW and watermark filtering**
- [ ] **Data privacy and retention policies**
- [ ] **Security review of image storage**

---

## ⚠️ **CRITICAL DEPENDENCIES**

### **For A/B Test Success**
1. **Real Image Gallery**: RA-Guard requires actual images to rerank
2. **Feature Caching**: Sub-ms latency needs precomputed embeddings
3. **Candidate Consistency**: Same candidate sets for fair A/B comparison
4. **Compliance Filtering**: Production-ready content filtering

### **Blocking Issues if Missing**
- **No gallery = No RA-Guard functionality** (reranker needs candidates)
- **No feature cache = Unacceptable latency** (real-time embedding computation)
- **No compliance = Legal/brand risk** (inappropriate content display)
- **No logging = Invalid A/B results** (can't verify offline/online consistency)

---

## 📊 **SCALE ESTIMATES**

### **Production Capacity**
```yaml
image_gallery_size:
  cocktails: 15000 images
  flowers: 20000 images  
  professional: 25000 images
  total: 60000 images

storage_requirements:
  raw_images: "300GB" # ~5MB avg per image
  clip_embeddings: "120MB" # 512 floats × 60K images
  metadata_db: "50MB"
  detection_cache: "500MB"
  total: "~301GB"

query_load:
  peak_qps: 1000
  candidates_per_query: 100 # average
  images_scored_per_second: 100K
  cache_lookups_per_second: 200K
```

---

## 🎯 **SUCCESS CRITERIA**

### **Infrastructure Ready**
- ✅ 60K+ images across 3 domains ingested and cached
- ✅ Sub-150ms P95 latency for candidate retrieval + reranking
- ✅ >90% cache hit rate for CLIP embeddings and detections
- ✅ Compliance filtering operational with <1% false positives

### **A/B Test Ready**
- ✅ Candidate logging enabled for offline/online validation
- ✅ Domain balance maintained (consistent with 300-query validation)
- ✅ Feature consistency verified between test and production
- ✅ Gallery size adequate for meaningful candidate pools

---

## 🚀 **IMMEDIATE ACTION ITEMS**

### **Week 1: Gallery Deployment**
1. **Data Engineering**: Deploy S3 + metadata DB infrastructure
2. **ML Engineering**: Set up CLIP embedding and detection pipelines  
3. **DevOps**: Configure Redis clusters for feature caching
4. **Quality Assurance**: Implement deduplication and compliance filtering

### **Week 2: Integration Testing**
1. **End-to-end validation**: Query → candidates → reranking → results
2. **Performance testing**: 1000 QPS load with latency measurement
3. **Consistency validation**: Offline vs online candidate matching
4. **A/B framework**: Candidate logging and analysis tools

---

**🎯 CRITICAL: The +4.24 nDCG T3-Verified result was achieved on real image galleries. Production A/B testing REQUIRES this same infrastructure to deliver the validated performance improvement.**

---

*🖼️ Gallery infrastructure is the foundation of RA-Guard's production success!*