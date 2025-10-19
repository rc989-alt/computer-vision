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



## 🏗️ Project Structure- **CLIP Embeddings**: State-of-the-art vision-language understanding for cocktail-image matching

- **YOLO Object Detection**: Advanced ingredient and garnish detection in cocktail images

```- **Dual Scoring System**: Comprehensive evaluation with visual appeal and ingredient matching

computer-vision/- **Conflict Penalty Mechanism**: Prevents color-ingredient mismatches (e.g., no orange juice in pink cocktails)

├── 📁 docs/                           # 📚 Complete Documentation System- **Subject-Object Constraints**: Maintains semantic consistency in generated cocktails

│   ├── 01_foundation/                 # 🏗️ Project Foundation (5 docs)- **CoTRR-lite Reranking**: Training-free listwise reranker with LLM integration

│   ├── 02_data_governance/            # 📊 Data Quality & Governance (8 docs)  - **Enhanced Garnish Intelligence**: 30-point scoring system with fruit/floral categorization

│   ├── 03_research_exploration/       # 🔬 Research Process (20 docs)

│   │   ├── day1/                      # 🌅 Initial Research (4 docs)## Demo Results

│   │   ├── day2/                      # ⚡ CoTRR Development (3 docs)

│   │   ├── day3/                      # 💎 Breakthrough Attempts (10 docs)Our 10-minute demo on 15 diverse cocktail queries shows significant improvements:

│   │   └── day4/                      # 🔬 Validation & Closure (3 docs)

│   ├── 04_production_deployment/      # 🚀 Production Systems (8 docs)| System | Compliance↑ | Conflict↓ | Avg Score |

│   ├── 05_analysis_reports/           # 📈 Analysis & Reports (7 docs)|--------|-------------|-----------|-----------|

│   └── 06_guides_references/          # 📖 Usage Guides (1 doc)| **Baseline (CLIP-only)** | 85% | 12% | 0.72 |

├── 📁 config/                         # ⚙️ Configuration Files| **Enhanced (+Region Control)** | **100%** | **0%** | **0.89** |

├── 📁 data/                           # 📊 Data & Datasets

├── 📁 scripts/                        # 🛠️ Utility Scripts### Side-by-Side Example: Pink Floral Cocktail

├── 📁 production/                     # 🚀 Production Code

├── 📁 research/                       # 🔬 Research Archives| Baseline Selection | Enhanced Selection |

└── 📄 pipeline.py                     # 🎯 Main Pipeline (302 lines)|-------------------|-------------------|

```| Score: 0.73 | Score: **0.92** (+0.19) |

| ❌ Generic pink drink | ✅ Rose petal garnish detected |

## 🎯 Key Achievements| ❌ No glass validation | ✅ Crystal coupe identified |

| ❌ Color-ingredient mismatch risk | ✅ Floral-pink harmony confirmed |

### ✅ **V1.0 Production Success**

- **+14.2% compliance improvement** in production*Complete demo results available in `demo/samples.json` and `runs/report/`*

- Complete monitoring and rollback systems

- Robust data governance framework📊 **View Results**: [`metrics.csv`](runs/report/metrics.csv) • [`summary.md`](runs/report/summary.md) • [`grid_*.png`](runs/report/)



### 🔬 **Research Exploration**## Quick Start

- Comprehensive CoTRR (Chain-of-Thought Retrieval & Ranking) investigation

- Multimodal fusion experiments with CLIP### 1. Setup

- Statistical validation and A/B testing frameworks```bash

# Clone the repository

### 📊 **Data Governance**git clone https://github.com/rc989-alt/computer-vision.git

- Clean dataset curation and validationcd computer-vision

- Borderline case review systems

- Canary deployment monitoring# Run automated setup

chmod +x setup.sh

## 📚 Documentation Navigation./setup.sh



### 🏗️ **Getting Started**# Verify installation

- [Project Overview](docs/01_foundation/01_project_overview.md) - Original project READMEpython3 test_setup.py

- [Pipeline Architecture](docs/01_foundation/02_pipeline_architecture.md) - Technical architecture```

- [Deployment Guide](docs/04_production_deployment/01_deployment_guide.md) - Production setup

### 2. Basic Usage

### 📊 **Current Status**```bash

- [Comprehensive Tech Progress](docs/05_analysis_reports/04_comprehensive_tech_progress.md) - Latest progress analysis# Run the complete pipeline

- [Recent 2-Day Progress](docs/05_analysis_reports/03_recent_2days_progress.md) - Detailed timelinepython3 pipeline.py --config config/default.json --input data/input/sample_input.json

- [Project Status Summary](docs/05_analysis_reports/01_project_status_summary.md) - High-level overview

# Advanced configuration

### 🔬 **Research Journey**python3 pipeline.py --config config/advanced.json --input your_input.json

- [Research Overview](docs/03_research_exploration/day1/01_research_overview.md) - Research objectives```

- [CoTRR Pro Plan](docs/03_research_exploration/day2/02_cotrr_pro_plan.md) - Advanced approach

- [V2 Project Closure](docs/03_research_exploration/day4/03_v2_project_closure.md) - Research conclusions### 3. Core Modules (NEW)

```bash

### 🚀 **Production Systems**# Compare baseline vs region control modes

- [V1 Accelerated Deployment](docs/04_production_deployment/06_v1_accelerated_deployment.md) - Production successpython3 pipeline.py --config config/default.json --input demo/samples.json --output results.json --mode baseline

- [Deployment Ready Report](docs/04_production_deployment/05_deployment_ready_report.md) - Go-live readinesspython3 pipeline.py --config config/default.json --input demo/samples.json --output results.json --mode region_control

- [Rollback Procedures](docs/04_production_deployment/03_rollback_procedure.md) - Risk mitigation

# Run mode comparison demo

## 🛠️ Quick Startpython3 demo_mode_comparison.py



### 🔧 **Setup**# Test core modules directly

```bashpython3 test_core_modules.py

# Install dependencies```

pip install -r requirements.txt

### 4. Individual Components

# Configure environment```bash

./setup.sh# Core module APIs (importable)

from src.subject_object import check_subject_object

# Test setupfrom src.conflict_penalty import conflict_penalty

python test_setup.pyfrom src.dual_score import fuse_dual_score

```

# Legacy individual scripts

### ▶️ **Run Pipeline**python3 scripts/clip_probe/clip_probe_training.py

```bashpython3 scripts/yolo_detection.py --input path/to/image.jpg

# Basic pipeline executionpython3 scripts/reranking_listwise.py --input cocktails.json

python pipeline.py```



# With custom config## Pipeline Architecture

python pipeline.py --config config/advanced.json

``````

Input Image/Query

### 📊 **Monitor Performance**       ↓

```bashCLIP Embeddings ←→ YOLO Detection

# View latest results       ↓

python scripts/analyze_cotrr_impact.mjsDual Scoring Evaluation

       ↓

# Generate reportsConflict Penalty Application

python scripts/run_dual_ablation_report.mjs       ↓

```Subject-Object Constraints

       ↓

## 📈 Project TimelineCoTRR-lite Reranking

       ↓

| Phase | Duration | Key Deliverables | Status |Final Cocktail Selection

|-------|----------|------------------|---------|```

| **Foundation** | Oct 11 AM | Architecture, Dataset, Pipeline | ✅ Complete |

| **Data Governance** | Oct 11 PM | Quality Systems, Monitoring | ✅ Complete |## Configuration

| **Research Exploration** | Oct 11-12 | CoTRR, Multimodal, Validation | ✅ Complete |

| **V1.0 Production** | Oct 12 AM | Deployment, Monitoring | ✅ **Live** |The pipeline supports multiple configuration levels:

| **V2.0 Research** | Oct 12 PM | Advanced Methods | 🔄 Closed |

- **Basic** (`config/default.json`): Standard cocktail generation

## 🎯 Next Steps- **Advanced** (`config/advanced.json`): Enhanced scoring with all features

- **Custom**: Create your own configuration following the schema

### 🔄 **Immediate Actions**

1. **Monitor V1.0 Performance**: Track production metrics and user feedback### Key Configuration Options

2. **Optimize Current Pipeline**: Fine-tune existing detection and classification

3. **Scale Infrastructure**: Prepare for increased load and usage```json

{

### 🚀 **Future Development**  "clip_model": "ViT-B/32",

1. **Enhanced Models**: Explore better base models and architectures  "yolo_model": "yolov8n.pt",

2. **Real-time Processing**: Implement streaming capabilities  "scoring_weights": {

3. **Advanced Analytics**: Develop deeper insights and reporting    "visual_appeal": 0.4,

    "ingredient_match": 0.3,

---    "garnish_score": 0.2,

    "dataset_priority": 0.1

## 📞 Support & Contact  },

  "conflict_penalty": 0.15,

For technical questions, refer to the comprehensive documentation in the `docs/` directory. Each phase and component has detailed documentation with implementation notes and lessons learned.  "reranking_enabled": true

}

**🎯 This project represents a successful journey from research exploration to production deployment, with a +14.2% compliance improvement and robust scalable architecture.**```

## Data Sources

- **Primary Dataset**: HuggingFace erwanlc/cocktails_recipe (6,956 cocktails)
- **Visual Database**: Comprehensive ingredient-color mapping
- **Garnish Intelligence**: 30-point scoring with fruit/floral categorization

## Scoring System

- **Dataset Priority**: 10 points (HuggingFace dataset preference)
- **Color Matching**: 40 points (visual-first strategy)
- **Garnish Matching**: 30 points (fruit vs floral intelligence)
- **Ingredient Match**: 20 points (recipe accuracy)
- **Visual Appeal**: 10 points (aesthetic scoring)
- **Total**: 110 points maximum

## Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- CLIP (OpenAI)
- Ultralytics YOLO
- scikit-learn
- NumPy
- Pillow

## Directory Structure

```
computer-vision/
├── pipeline.py              # Main pipeline interface
├── setup.sh                # Automated setup script
├── test_setup.py           # Verification script
├── requirements.txt        # Dependencies
├── config/                 # Configuration files
├── scripts/               # Individual components
│   ├── image_model.py     # CLIP integration
│   ├── yolo_detection.py  # Object detection
│   ├── reranking_listwise.py # CoTRR-lite reranking
│   ├── clip_probe/        # CLIP training scripts
│   └── ...               # Additional utilities
├── data/                  # Input/output data
└── docs/                  # Documentation
```

## Core Modules

The pipeline's key capabilities are implemented as explicit, importable modules:

### 1. Subject-Object Constraints (`src/subject_object.py`)
```python
compliance, details = check_subject_object(triples, regions)
# Returns: (compliance: float, details: dict)
```
Validates semantic consistency between detected objects and their relationships.

### 2. Conflict Penalty (`src/conflict_penalty.py`)  
```python
penalty, details = conflict_penalty(regions, graph, alpha=0.3)
# Returns: (penalty: float, details: dict)
```
Detects and penalizes semantic conflicts (e.g., pink cocktails with orange garnish).

### 3. Dual Score Fusion (`src/dual_score.py`)
```python
final_score = fuse_dual_score(compliance, conflict, w_c=0.5, w_n=0.5, normalize=True)
# Returns: float
```
Combines compliance and conflict scores with configurable weighting.

## Advanced Features

### Visual-First Strategy
The pipeline prioritizes visual appearance over traditional ingredient-based matching, ensuring cocktails look appealing before considering taste compatibility.

### Garnish Intelligence
Enhanced fruit vs floral detection with generic term matching:
- Fruit garnishes: citrus, berries, tropical fruits
- Floral garnishes: edible flowers, petals, botanical elements

### Conflict Resolution
Prevents common mismatches:
- No orange juice in pink cocktails
- Appropriate garnish-color combinations
- Seasonal ingredient constraints

### Reranking System
CoTRR-lite integration provides training-free listwise reranking with LLM integration for improved cocktail selection quality.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{cocktail_cv_pipeline,
  title={Computer Vision Pipeline for Cocktail Generation},
  author={RC989-ALT},
  year={2024},
  url={https://github.com/rc989-alt/computer-vision}
}
```

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review configuration examples in `config/`

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Compatibility**: Python 3.8+ | PyTorch 2.0+ | CUDA Optional