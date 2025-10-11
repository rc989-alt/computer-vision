# Canary Monitoring System Implementation

## Overview

Complete canary monitoring system for detecting dataset drift and quality degradation in the computer vision pipeline. The system provides automated monitoring with drift detection, alerting, and pipeline blocking capabilities.

## System Components

### 1. Probe Set Generator (`probe_set_generator.py`)
- **Purpose**: Creates balanced, stable probe sets for consistent monitoring
- **Size**: 85 items (40 positive cocktails + 45 hard negatives)
- **Balance**: Covers 17 domains including classic, colored, and tricky cocktails
- **Stability**: Fixed random seed ensures reproducible probe sets

```bash
# Generate probe set
python3 probe_set_generator.py --version v1.0 --validate

# Output: data/probe/probe_set_v1.0.json
```

### 2. Canary Monitor (`canary_monitor.py`)
- **Purpose**: Computes CLIP margins and detects quality drift
- **Metrics**: Mean/median/P95 margins, positive/negative accuracy
- **Drift Detection**: 3% mean drop threshold, 95% confidence intervals
- **Baseline**: 7-run rolling window for stable comparisons

```bash
# Run canary check
python3 canary_monitor.py --probe-version v1.0

# Output: data/canary/canary_metrics_[run_id].json
```

### 3. Canary Automation (`canary_automation.py`)
- **Purpose**: Automated triggering and incident management
- **Triggers**: Overlay updates, source syncs, model changes
- **Actions**: Pipeline blocking, incident creation, alert routing
- **Integration**: Git hooks, CI/CD pipeline integration

```bash
# Check triggers and run if needed
python3 canary_automation.py --check

# Force canary check
python3 canary_automation.py --force "manual_validation"
```

## Key Metrics & Thresholds

### CLIP Margin Calculation
```python
clip_margin = sim_cocktail - sim_not_cocktail
```

Where:
- `sim_cocktail` = CLIP similarity to "a cocktail drink"
- `sim_not_cocktail` = CLIP similarity to "not a cocktail"

### Alert Thresholds
- **Critical**: Mean margin drops > 5% relative to baseline
- **Warning**: Mean margin drops > 3% relative to baseline  
- **CI Exclusion**: Current mean outside 95% confidence interval
- **Tail Drift**: P95 margin drops > 5% relative to baseline

### Baseline Computation
- **Rolling Window**: Last 7 successful runs
- **Minimum History**: 3 days of data required
- **Update Frequency**: After each canary run

## Probe Set Design

### Positive Examples (40 items)
- **Classic Clear**: Martinis, gin cocktails (5 items)
- **Classic Amber**: Old fashioneds, whiskey cocktails (5 items)
- **Color Pink**: Cosmopolitans, rose cocktails (5 items)
- **Color Blue**: Tropical blue cocktails (5 items)
- **Color Green**: Mint cocktails, margaritas (5 items)
- **Tricky Dark**: Charcoal cocktails (5 items)
- **Tricky Cream**: White/cream cocktails (5 items)
- **Tricky Layered**: Multi-layer cocktails (5 items)

### Hard Negative Examples (45 items)
- **Beverage Tea**: Tea in glassware (5 items)
- **Beverage Soda**: Sodas in cocktail glasses (5 items)
- **Beverage Juice**: Juices in wine glasses (5 items)
- **Beverage Water**: Infused waters (5 items)
- **Liquid Perfume**: Perfume bottles (5 items)
- **Liquid Soup**: Soups in glass bowls (5 items)
- **Liquid Sauce**: Oils, vinegars in glass (5 items)
- **Decorative Vase**: Glass vases with water (5 items)
- **Decorative Candle**: Glass candle holders (5 items)

## Integration Points

### 1. Pipeline Integration
Add to your main pipeline script:

```bash
#!/bin/bash
# Before processing
python3 canary_automation.py --check
if [ $? -ne 0 ]; then
    echo "❌ Canary check failed - pipeline blocked"
    exit 1
fi

# Your existing pipeline
python3 pipeline.py --mode region_control
```

### 2. Git Hooks Integration
Add to `.git/hooks/pre-push`:

```bash
#!/bin/bash
echo "Running canary check..."
python3 canary_automation.py --check
if [ $? -ne 0 ]; then
    echo "❌ Canary drift detected - push blocked"
    echo "   Review drift incident in data/incidents/"
    exit 1
fi
```

### 3. CI/CD Integration
Add to your GitHub Actions or CI system:

```yaml
- name: Canary Check
  run: |
    python3 canary_automation.py --check
    if [ $? -ne 0 ]; then
      echo "::error::Dataset drift detected"
      exit 1
    fi
```

## Configuration

### Canary Config (`config/canary.json`)
```json
{
  "triggers": {
    "overlay_update": true,
    "source_sync": true, 
    "model_change": true,
    "scheduled": false
  },
  "thresholds": {
    "mean_drop_critical": 0.05,
    "mean_drop_warning": 0.03,
    "ci_confidence": 0.95,
    "tail_drop_threshold": 0.05
  },
  "baseline": {
    "rolling_window": 7,
    "min_history_days": 3,
    "max_history_days": 30
  },
  "alerts": {
    "block_on_critical": true,
    "create_incidents": true
  }
}
```

## Incident Management

### Drift Incident Structure
```json
{
  "incident_id": "drift_auto_20251011_140529_model_change",
  "created_at": "2025-10-11T14:05:29",
  "trigger_reason": "model_change",
  "status": "open",
  "severity": "critical",
  "summary": "Dataset drift detected (2 alerts)",
  "alerts": [...],
  "investigation_notes": [],
  "resolution": null
}
```

### Incident Files
- **Location**: `data/incidents/`
- **Format**: `drift_[run_id].json` + `_summary.txt`
- **Contents**: Alert details, affected images, investigation steps

## Monitoring Dashboard

### Key Metrics to Track
1. **Mean Margin Trend**: Track over time for gradual drift
2. **Alert Frequency**: Monitor false positive rates
3. **Accuracy Rates**: Positive/negative classification performance
4. **P95 Tail Behavior**: Early indicator of quality degradation

### Sample Dashboard Queries
```python
# Load recent canary metrics
import json
from pathlib import Path

metrics_files = list(Path("data/canary").glob("canary_metrics_*.json"))
recent_metrics = []

for file in sorted(metrics_files)[-10:]:  # Last 10 runs
    with open(file) as f:
        data = json.load(f)
        recent_metrics.append({
            'timestamp': data['timestamp'],
            'mean_margin': data['aggregated_metrics']['mean_margin'],
            'positive_accuracy': data['aggregated_metrics']['positive_accuracy']
        })

# Plot trend
import matplotlib.pyplot as plt
plt.plot([m['mean_margin'] for m in recent_metrics])
plt.title('Canary Mean Margin Trend')
plt.show()
```

## Testing & Validation

### Test Canary System
```bash
# 1. Generate probe set
python3 probe_set_generator.py --version v1.0 --validate

# 2. Run baseline checks
for i in {1..5}; do
    python3 canary_monitor.py --probe-version v1.0 --run-id "baseline_$i"
done

# 3. Test automation
touch src/dual_score.py  # Trigger model change
python3 canary_automation.py --check

# 4. Verify incident creation
ls data/incidents/
```

### Mock Drift Scenario
To test alert generation, modify the mock score generation in `canary_monitor.py`:

```python
# Simulate degraded performance
if example['label'] == 'positive':
    base_cocktail = 0.55  # Lower than normal 0.75
    base_not_cocktail = 0.45  # Higher than normal 0.25
```

## Production Deployment

### 1. Initialize System
```bash
# Create directories
mkdir -p data/{probe,canary,incidents}

# Generate stable probe set
python3 probe_set_generator.py --version v1.0 --validate

# Initialize config
python3 canary_automation.py --init-config

# Build baseline (run 5-7 times over 3+ days)
python3 canary_monitor.py --probe-version v1.0
```

### 2. Enable Automation
```bash
# Add to your deployment pipeline
python3 canary_automation.py --check

# Set up scheduled monitoring (optional)
# Add to crontab: 0 6 * * * cd /path/to/project && python3 canary_automation.py --check
```

### 3. Monitor & Maintain
- **Weekly**: Review incident trends and false positive rates
- **Monthly**: Update probe set if domain coverage changes
- **Quarterly**: Retrain baseline if data distribution shifts significantly

## Expected Results

### Baseline Performance
- **Mean Margin**: ~0.40 (positive examples much higher than negatives)
- **Positive Accuracy**: >95% (cocktails correctly scored higher)
- **Negative Accuracy**: >90% (non-cocktails correctly scored lower)
- **P95 Margin**: >0.65 (strong positive examples very confident)

### Alert Scenarios
1. **Model Degradation**: CLIP model quality drops → lower margins
2. **Dataset Shift**: New data distribution → changed accuracy patterns  
3. **Pipeline Bug**: Processing errors → systematic score drops
4. **Infrastructure Issues**: GPU/memory problems → inconsistent results

The canary system provides early warning for all these scenarios, enabling proactive quality management and preventing degraded models from reaching production.

## Files Created

```
├── probe_set_generator.py     # Balanced probe set generation
├── canary_monitor.py          # CLIP scoring & drift detection
├── canary_automation.py       # Automated triggering & incidents
├── config/canary.json         # Configuration settings
├── data/
│   ├── probe/
│   │   └── probe_set_v1.0.json    # Stable probe set (85 items)
│   ├── canary/
│   │   └── canary_metrics_*.json  # Historical metrics
│   └── incidents/
│       └── drift_*.json           # Drift incident records
```

This comprehensive canary system provides production-grade monitoring for dataset quality with automated drift detection, incident management, and pipeline integration.