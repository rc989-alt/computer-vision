# Computer Vision Pipeline Documentation

## Pipeline Components

### 1. Image Model (`scripts/image_model.py`)
Main detection pipeline using Faster R-CNN for object detection.

**Features:**
- COCO object detection (bottles, cups, wine glasses)
- Confidence scoring
- Batch processing
- Download retry logic

**Usage:**
```bash
python scripts/image_model.py --in input.json --out output.json --threshold 0.5
```

### 2. YOLO Detection (`scripts/yolo_detector.py`)
Advanced YOLO-based object detection with custom classes.

**Features:**
- YOLOv8 integration
- Custom class detection
- Bounding box extraction
- Performance optimization

### 3. CLIP Probe Training (`scripts/clip_probe/`)
Training and inference pipeline for CLIP-based classification.

**Components:**
- `train_clip_probe_balanced.py`: Balanced training with cross-validation
- `run_clip_probe_inference.py`: Inference on new data
- `embedding_cache.py`: Efficient embedding caching

**Features:**
- Balanced positive/negative sampling
- Cross-validation with metrics
- Embedding caching for efficiency
- ROC-AUC evaluation

### 4. Reranking System
Two-tier reranking system with compliance and LLM integration.

#### Basic Reranking (`scripts/rerank_with_compliance.mjs`)
- Compliance scoring
- Subject-object constraints
- Conflict penalty mechanism
- Graph-based smoothing

#### Advanced Reranking (`scripts/rerank_listwise_llm.mjs`)
- CoTRR-lite listwise reranking
- LLM integration via proxy
- Training-free approach
- Configurable weights

### 5. Evaluation Framework
Comprehensive evaluation and ablation studies.

#### Dual Scoring Ablation (`scripts/run_dual_scored_ablation.mjs`)
- Compare CLIP vs YOLO scoring
- Performance metrics
- Statistical analysis

#### Impact Analysis (`scripts/analyze_cotrr_impact.mjs`)
- CoTRR-lite vs subject-object comparison
- Effectiveness metrics

## Configuration Options

### Detection Parameters
- `detection_threshold`: YOLO confidence threshold (0.0-1.0)
- `require_glass`: Enforce glass detection
- `hard_require_glass`: Hard filter vs soft penalty

### Scoring Weights
- `compliance_weight`: Base compliance score weight
- `llm_weight`: LLM reranking influence
- `conflict_penalty`: Penalty for visual-text mismatches

### Constraint System
- `require_subjects`: List of required detected objects
- `forbid_subjects`: List of forbidden objects
- `forbid_penalty`: Penalty for forbidden content

### Training Parameters
- `per_class_samples`: Samples per class for balanced training
- `cross_validation_folds`: K-fold CV configuration
- `batch_size`: Training batch size

## Data Flow

```
Input Images → Feature Extraction → Dual Scoring → Constraint Application → Reranking → Final Results
     ↓              ↓                    ↓                ↓                    ↓           ↓
   URLs/Paths    CLIP+YOLO         Combined Score    Filter/Penalty      LLM Rerank    Ranked List
```

## Performance Tuning

### GPU Optimization
- Use CUDA-enabled PyTorch for faster inference
- Batch processing for CLIP embeddings
- Model quantization for deployment

### Memory Management
- Embedding caching to avoid recomputation
- Streaming processing for large datasets
- Garbage collection for long-running processes

### Quality vs Speed Trade-offs
- Higher detection thresholds = faster but less sensitive
- More cross-validation folds = better validation but slower training
- LLM reranking = better quality but higher latency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU mode with `--device cpu`
   - Enable gradient checkpointing

2. **Download Failures**
   - Check internet connection
   - Verify API keys in .env
   - Increase timeout values

3. **Model Loading Errors**
   - Ensure models are downloaded
   - Check file permissions
   - Verify CLIP installation

4. **Node.js Script Failures**
   - Install dependencies with `npm install`
   - Check Node.js version (16+)
   - Verify JSON input format

### Performance Optimization

1. **For Large Datasets**
   - Use embedding cache
   - Enable multiprocessing
   - Consider distributed processing

2. **For Real-time Applications**
   - Pre-compute embeddings
   - Use model quantization
   - Implement result caching

3. **For High Accuracy**
   - Increase detection threshold
   - Use ensemble methods
   - Fine-tune on domain data