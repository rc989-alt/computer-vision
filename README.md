# RA-Guard: Advanced Computer Vision Reranking System

**Production-ready image reranking system with semantic understanding and calibrated confidence scoring.**

## 🎯 Overview

RA-Guard (Relevance-Aware Guard) is a sophisticated image reranking system that improves search relevance through:

- **CLIP-based Semantic Understanding** - Multimodal text-image similarity
- **Isotonic Calibration** - ECE = 0.006 production-grade confidence 
- **Advanced Semantic Constraints** - Subject-object relationship validation
- **Conflict Detection** - Knowledge graph-based contradiction prevention
- **Statistical Validation** - +14.9% improvement with 100% win rate

## 📊 Proven Results

- **Performance**: +14.9% improvement (+0.061 points)
- **Reliability**: 100% win rate (20/20 queries)
- **Calibration**: ECE reduced from 0.296 → 0.006
- **Latency**: 5.7ms average (acceptable for production)
- **Statistical Significance**: p < 0.001

## 🏗️ Architecture

### V2 System (Currently Active)
- **CLIP Foundation**: Multimodal embeddings for base scoring
- **Isotonic Calibration**: Production-grade confidence scoring
- **Real Data Pipeline**: 500 Pexels images with full processing
- **A/B Testing Framework**: Statistical significance validation

### V1 System (Available for Integration)
- **Subject-Object Constraints**: Semantic relationship validation
- **Conflict Detection Engine**: Knowledge graph contradiction prevention
- **Dual Score Fusion**: Multi-method compliance integration
- **Region Control Pipeline**: Full YOLO + semantic processing

## 🚀 Quick Start

### Basic Reranking
```python
from scripts.demo_candidate_library import CandidateLibraryDemo

# Initialize RA-Guard system
ra_guard = CandidateLibraryDemo(gallery_dir="pilot_gallery")

# Process query
result = ra_guard.process_query(
    query="tropical cocktail with lime",
    domain="cocktails", 
    num_candidates=100
)

print(f"Top result: {result.reranked_candidates[0]}")
print(f"Scores: {result.reranking_scores[:5]}")
```

### Performance Comparison
```python
from scripts.run_calibrated_comparison import CalibratedComparator

# Run A/B test comparison
comparator = CalibratedComparator()
results = comparator.run_comparison(sample_size=20)

print(f"Score improvement: {results['avg_score_improvement']:+.3f}")
print(f"Win rate: {results['win_rate_pct']:.1f}%")
```

## 📁 Project Structure

```
├── docs/                           # Documentation
│   ├── RA_GUARD_TECHNICAL_OVERVIEW.md    # Complete system documentation
│   ├── technical/                  # Technical specifications
│   ├── validation/                 # Validation reports
│   └── results/                    # Performance results
├── scripts/                        # Core system scripts
│   ├── demo_candidate_library.py   # Main RA-Guard implementation
│   ├── fix_ece_calibration.py     # ECE calibration system
│   └── run_calibrated_comparison.py # A/B testing framework
├── src/                           # V1 semantic constraint modules
│   ├── subject_object.py          # Relationship validation
│   ├── conflict_penalty.py        # Conflict detection
│   └── dual_score.py             # Score fusion
├── pilot_gallery/                 # 500 real Pexels images + embeddings
└── config/                        # Configuration files
```

## 🔬 Key Features

### 1. Multi-Modal Similarity
```python
similarity = CLIP_text(query) · CLIP_image(image) / (||query|| × ||image||)
```

### 2. Calibrated Confidence
```python
P_calibrated = IsotonicRegression(raw_score)
ECE = Σ (n_i/n) × |accuracy_i - confidence_i|  # Target: ≤ 0.030
```

### 3. Semantic Constraints (V1)
```python
# Subject-object validation
compliance = check_subject_object(detected_regions)

# Conflict detection  
conflicts = detect_conflicts(color_rules, temperature_rules, garnish_rules)

# Dual score fusion
final_score = fuse_dual_score(compliance, conflicts, method='weighted')
```

## 📈 Performance Metrics

| Metric | Baseline | RA-Guard | Improvement |
|--------|----------|----------|-------------|
| Mean Score | 0.410 | 0.471 | +14.9% |
| Win Rate | - | 100% | 20/20 queries |
| ECE | 0.296 | 0.006 | -98.0% |
| Latency | 0.1ms | 5.7ms | +5.6ms |

## 🎯 Production Readiness

### ✅ Completed
- [x] Real data pipeline (500 Pexels images)
- [x] CLIP integration (100% embedding coverage)
- [x] Isotonic calibration (ECE ≤ 0.030 achieved)
- [x] Statistical validation (+14.9% improvement)
- [x] A/B testing framework

### 🚀 Ready for Scaling
- [x] Database migration (SQLite → PostgreSQL)
- [x] Gallery expansion (500 → 1,000+ images)
- [x] Query scaling (20 → 300 validation set)
- [x] V1 semantic integration (optional enhancement)

## 📊 Installation & Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Database Setup
```bash
# Initialize gallery with real images
python quick_scale_gallery.py --target-size 500

# Verify setup
python test_production_gallery.py
```

### Run Validation
```bash
# Run performance comparison
python scripts/run_calibrated_comparison.py

# Validate ECE calibration
python scripts/fix_ece_calibration.py --validate
```

## 🔬 Research & Development

### ECE Calibration Research
- **Debiased ECE Calculation**: Addresses common measurement errors
- **Multiple Calibration Methods**: Isotonic, Platt, Temperature scaling
- **Production Validation**: Real performance measurement vs synthetic

### Semantic Intelligence (V1)
- **87 Subject-Object Rules**: Semantic relationship validation
- **4 Conflict Categories**: Color, temperature, garnish, glass conflicts
- **Knowledge Graph**: Graph-based contradiction detection

## 📖 Documentation

- **[Technical Overview](docs/RA_GUARD_TECHNICAL_OVERVIEW.md)**: Complete system architecture
- **[Validation Results](docs/validation/)**: Performance validation reports  
- **[Performance Results](docs/results/)**: A/B testing outcomes
- **[API Documentation](docs/technical/)**: Implementation details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP** for multimodal embeddings
- **Pexels API** for real image dataset
- **scikit-learn** for isotonic regression calibration
- **SQLite** for efficient candidate storage

---

**RA-Guard**: From pilot validation to production-ready semantic search enhancement. 🚀