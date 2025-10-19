# 🔬 Research Track: CoTRR-lite Reranker

## Step 5: Train a CoTRR-lite reranker

### Overview
This research kit integrates with your Step 4 production pipeline to train a lightweight reranker that optimizes the dual objective: **λ·Compliance + (1−λ)·(1−p_conflict)**.

### Architecture
- **Input Features**: CLIP image/text embeddings, region stats (glass/garnish/ice), color ΔE, subject_ratio, conflict scores
- **Labels**: `y = λ·Compliance + (1−λ)·(1−p_conflict)` (domain-normalized)
- **Loss**: Pairwise RankNet on (positive > negative) pairs from each query's candidate set
- **Output**: Calibrated reranking scores with uncertainty estimates

### Components

#### 1. Feature Engineering (`src/feature_extractor.py`)
- Loads scored JSONL from Step 4 pipeline
- Extracts multi-modal features: CLIP, visual attributes, conflict scores
- Handles domain normalization and feature scaling

#### 2. Reranker Training (`src/reranker_train.py`)
- PyTorch pairwise RankNet implementation
- Supports CLIP-only vs +SO vs +SO+Conflict vs +Reranker ablations
- Automatic hyperparameter tuning with validation split

#### 3. Evaluation Suite (`src/eval_suite.py`)
- Compliance@1/3/5, nDCG@10, Conflict AUC/ECE with 95% CI
- Bootstrap confidence intervals (1000 samples)
- Failure analysis with explanations

#### 4. Model Registry Integration (`src/model_registry.py`)
- Connects to Step 4 A/B testing framework
- Automated promotion based on performance thresholds
- Version management and rollback capabilities

### Quick Start

1. **Generate training data from Step 4 pipeline**:
```bash
python research/src/generate_training_data.py --input runs/latest --output research/data/training.jsonl
```

2. **Train CoTRR-lite reranker**:
```bash
python research/src/reranker_train.py --config research/configs/cotrr_lite.yaml
```

3. **Evaluate with confidence intervals**:
```bash
python research/src/eval_suite.py --model research/models/cotrr_lite_v1 --test research/data/test.jsonl
```

### Acceptance Criteria

- **Compliance@1**: +3–5 pts vs CLIP-only (with 95% CI)
- **nDCG@10**: +6–10 pts vs CLIP-only  
- **Conflict**: AUC ≥ 0.90, ECE ≤ 0.05
- **Latency**: Within Step 4 production budget
- Complete ablation study + failure analysis

### Integration with Step 4

The research track maintains full compatibility with your production pipeline:
- Uses same feature extraction as Step 4 batched scoring
- Integrates with A/B testing framework for safe deployment
- Maintains canary monitoring and governance CI
- No train/test leakage via canonical_id splitting