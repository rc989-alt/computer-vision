# Computer Vision Pipeline for Cocktail Generation

A comprehensive computer vision pipeline for cocktail image generation and selection, featuring CLIP embeddings, YOLO object detection, dual scoring evaluation, conflict penalty mechanisms, and subject-object constraints.

## Features

- **CLIP Embeddings**: State-of-the-art vision-language understanding for cocktail-image matching
- **YOLO Object Detection**: Advanced ingredient and garnish detection in cocktail images
- **Dual Scoring System**: Comprehensive evaluation with visual appeal and ingredient matching
- **Conflict Penalty Mechanism**: Prevents color-ingredient mismatches (e.g., no orange juice in pink cocktails)
- **Subject-Object Constraints**: Maintains semantic consistency in generated cocktails
- **CoTRR-lite Reranking**: Training-free listwise reranker with LLM integration
- **Enhanced Garnish Intelligence**: 30-point scoring system with fruit/floral categorization

## Demo Results

Our 10-minute demo on 15 diverse cocktail queries shows significant improvements:

| System | Compliance↑ | Conflict↓ | Avg Score |
|--------|-------------|-----------|-----------|
| **Baseline (CLIP-only)** | 85% | 12% | 0.72 |
| **Enhanced (+Region Control)** | **100%** | **0%** | **0.89** |

### Side-by-Side Example: Pink Floral Cocktail

| Baseline Selection | Enhanced Selection |
|-------------------|-------------------|
| Score: 0.73 | Score: **0.92** (+0.19) |
| ❌ Generic pink drink | ✅ Rose petal garnish detected |
| ❌ No glass validation | ✅ Crystal coupe identified |
| ❌ Color-ingredient mismatch risk | ✅ Floral-pink harmony confirmed |

*Complete demo results available in `demo/samples.json` and `runs/report/`*

## Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/rc989-alt/computer-vision.git
cd computer-vision

# Run automated setup
chmod +x setup.sh
./setup.sh

# Verify installation
python3 test_setup.py
```

### 2. Basic Usage
```bash
# Run the complete pipeline
python3 pipeline.py --config config/default.json --input data/input/sample_input.json

# Advanced configuration
python3 pipeline.py --config config/advanced.json --input your_input.json
```

### 3. Individual Components
```bash
# CLIP embeddings only
python3 scripts/clip_probe/clip_probe_training.py

# YOLO detection only
python3 scripts/yolo_detection.py --input path/to/image.jpg

# Reranking only
python3 scripts/reranking_listwise.py --input cocktails.json
```

## Pipeline Architecture

```
Input Image/Query
       ↓
CLIP Embeddings ←→ YOLO Detection
       ↓
Dual Scoring Evaluation
       ↓
Conflict Penalty Application
       ↓
Subject-Object Constraints
       ↓
CoTRR-lite Reranking
       ↓
Final Cocktail Selection
```

## Configuration

The pipeline supports multiple configuration levels:

- **Basic** (`config/default.json`): Standard cocktail generation
- **Advanced** (`config/advanced.json`): Enhanced scoring with all features
- **Custom**: Create your own configuration following the schema

### Key Configuration Options

```json
{
  "clip_model": "ViT-B/32",
  "yolo_model": "yolov8n.pt",
  "scoring_weights": {
    "visual_appeal": 0.4,
    "ingredient_match": 0.3,
    "garnish_score": 0.2,
    "dataset_priority": 0.1
  },
  "conflict_penalty": 0.15,
  "reranking_enabled": true
}
```

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