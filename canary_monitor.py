#!/usr/bin/env python3
"""
Canary Monitor System

Monitors dataset quality using CLIP-based scoring and drift detection.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import hashlib
import statistics
from dataclasses import dataclass
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CanaryScore:
    """Single canary score measurement."""
    image_id: str
    sim_cocktail: float
    sim_not_cocktail: float
    clip_margin: float
    ground_truth_label: str
    description: str
    domain: str
    timestamp: str

@dataclass
class CanaryMetrics:
    """Aggregated canary metrics for a run."""
    run_id: str
    timestamp: str
    mean_margin: float
    median_margin: float
    std_margin: float
    p95_margin: float
    p05_margin: float
    positive_accuracy: float
    negative_accuracy: float
    total_items: int
    individual_scores: List[CanaryScore]

@dataclass
class DriftAlert:
    """Drift detection alert."""
    alert_type: str  # 'mean_drop', 'ci_exclusion', 'tail_drift'
    severity: str    # 'warning', 'critical'
    baseline_value: float
    current_value: float
    delta: float
    threshold: float
    confidence: float
    message: str
    affected_images: List[str]

class CanaryMonitor:
    """Monitor dataset quality using canary probe set."""
    
    def __init__(self, probe_path: str = "data/probe", baseline_path: str = "data/canary"):
        self.probe_path = Path(probe_path)
        self.baseline_path = Path(baseline_path)
        self.baseline_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.mean_drop_threshold = 0.03  # 3% relative drop
        self.confidence_level = 0.95     # 95% confidence interval
        self.rolling_window = 7          # 7-run rolling baseline
        
        logger.info(f"Initialized canary monitor: probe={probe_path}, baseline={baseline_path}")
    
    def load_probe_set(self, version: str = "v1.0") -> Dict[str, Any]:
        """Load probe set for monitoring."""
        probe_file = self.probe_path / f"probe_set_{version}.json"
        
        if not probe_file.exists():
            raise FileNotFoundError(f"Probe set not found: {probe_file}")
        
        with open(probe_file, 'r') as f:
            probe_set = json.load(f)
        
        logger.info(f"Loaded probe set {version}: {len(probe_set['examples'])} items")
        return probe_set
    
    def compute_clip_scores(self, probe_set: Dict[str, Any], mock: bool = True) -> List[CanaryScore]:
        """
        Compute CLIP scores for probe set.
        
        Args:
            probe_set: Probe set with examples
            mock: If True, generate mock scores for testing
            
        Returns:
            List of CanaryScore objects
        """
        if mock:
            return self._generate_mock_scores(probe_set)
        else:
            # TODO: Implement real CLIP scoring
            raise NotImplementedError("Real CLIP scoring not yet implemented")
    
    def _generate_mock_scores(self, probe_set: Dict[str, Any]) -> List[CanaryScore]:
        """Generate realistic mock CLIP scores for testing."""
        scores = []
        
        for example in probe_set['examples']:
            # Simulate realistic CLIP similarities
            if example['label'] == 'positive':
                # Positive examples: higher cocktail similarity
                base_cocktail = 0.75 + np.random.normal(0, 0.1)
                base_not_cocktail = 0.25 + np.random.normal(0, 0.08)
                
                # Add difficulty-based variation
                if example['difficulty'] == 'hard':
                    base_cocktail -= 0.1  # Harder positives score lower
                    base_not_cocktail += 0.05
                
            else:  # negative
                # Negative examples: higher not-cocktail similarity
                base_cocktail = 0.35 + np.random.normal(0, 0.1)
                base_not_cocktail = 0.65 + np.random.normal(0, 0.08)
                
                # Hard negatives (glassware) might confuse the model
                if 'beverage' in example['domain']:
                    base_cocktail += 0.1  # Beverages in glass confuse model
                    base_not_cocktail -= 0.05
            
            # Clip to valid range
            sim_cocktail = np.clip(base_cocktail, 0.0, 1.0)
            sim_not_cocktail = np.clip(base_not_cocktail, 0.0, 1.0)
            clip_margin = sim_cocktail - sim_not_cocktail
            
            score = CanaryScore(
                image_id=example['id'],
                sim_cocktail=sim_cocktail,
                sim_not_cocktail=sim_not_cocktail,
                clip_margin=clip_margin,
                ground_truth_label=example['label'],
                description=example['description'],
                domain=example['domain'],
                timestamp=datetime.now().isoformat()
            )
            scores.append(score)
        
        return scores
    
    def compute_metrics(self, scores: List[CanaryScore], run_id: str = None) -> CanaryMetrics:
        """Compute aggregated metrics from individual scores."""
        
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        margins = [score.clip_margin for score in scores]
        
        # Compute percentiles
        p95 = np.percentile(margins, 95)
        p05 = np.percentile(margins, 5)
        
        # Compute accuracy by label
        positive_scores = [s for s in scores if s.ground_truth_label == 'positive']
        negative_scores = [s for s in scores if s.ground_truth_label == 'negative']
        
        positive_accuracy = sum(1 for s in positive_scores if s.clip_margin > 0) / len(positive_scores)
        negative_accuracy = sum(1 for s in negative_scores if s.clip_margin < 0) / len(negative_scores)
        
        metrics = CanaryMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            mean_margin=np.mean(margins),
            median_margin=np.median(margins),
            std_margin=np.std(margins),
            p95_margin=p95,
            p05_margin=p05,
            positive_accuracy=positive_accuracy,
            negative_accuracy=negative_accuracy,
            total_items=len(scores),
            individual_scores=scores
        )
        
        return metrics
    
    def save_metrics(self, metrics: CanaryMetrics) -> str:
        """Save metrics to baseline directory."""
        
        output_file = self.baseline_path / f"canary_metrics_{metrics.run_id}.json"
        
        # Convert to serializable format
        metrics_dict = {
            'run_id': metrics.run_id,
            'timestamp': metrics.timestamp,
            'aggregated_metrics': {
                'mean_margin': metrics.mean_margin,
                'median_margin': metrics.median_margin,
                'std_margin': metrics.std_margin,
                'p95_margin': metrics.p95_margin,
                'p05_margin': metrics.p05_margin,
                'positive_accuracy': metrics.positive_accuracy,
                'negative_accuracy': metrics.negative_accuracy,
                'total_items': metrics.total_items
            },
            'individual_scores': [
                {
                    'image_id': score.image_id,
                    'sim_cocktail': score.sim_cocktail,
                    'sim_not_cocktail': score.sim_not_cocktail,
                    'clip_margin': score.clip_margin,
                    'ground_truth_label': score.ground_truth_label,
                    'description': score.description,
                    'domain': score.domain,
                    'timestamp': score.timestamp
                }
                for score in metrics.individual_scores
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Saved canary metrics to {output_file}")
        return str(output_file)
    
    def load_historical_metrics(self, days_back: int = 30) -> List[CanaryMetrics]:
        """Load historical metrics for baseline computation."""
        
        # Find all metric files within time window
        cutoff_date = datetime.now() - timedelta(days=days_back)
        metric_files = []
        
        for file_path in self.baseline_path.glob("canary_metrics_*.json"):
            try:
                # Extract timestamp from filename
                timestamp_str = file_path.stem.split('_', 2)[2]  # After "canary_metrics_"
                file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if file_date >= cutoff_date:
                    metric_files.append(file_path)
            except:
                continue  # Skip files with bad timestamp format
        
        # Load and convert to CanaryMetrics objects
        historical_metrics = []
        for file_path in sorted(metric_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert individual scores
                scores = []
                for score_data in data['individual_scores']:
                    score = CanaryScore(
                        image_id=score_data['image_id'],
                        sim_cocktail=score_data['sim_cocktail'],
                        sim_not_cocktail=score_data['sim_not_cocktail'],
                        clip_margin=score_data['clip_margin'],
                        ground_truth_label=score_data['ground_truth_label'],
                        description=score_data['description'],
                        domain=score_data['domain'],
                        timestamp=score_data['timestamp']
                    )
                    scores.append(score)
                
                # Convert to CanaryMetrics
                agg = data['aggregated_metrics']
                metrics = CanaryMetrics(
                    run_id=data['run_id'],
                    timestamp=data['timestamp'],
                    mean_margin=agg['mean_margin'],
                    median_margin=agg['median_margin'],
                    std_margin=agg['std_margin'],
                    p95_margin=agg['p95_margin'],
                    p05_margin=agg['p05_margin'],
                    positive_accuracy=agg['positive_accuracy'],
                    negative_accuracy=agg['negative_accuracy'],
                    total_items=agg['total_items'],
                    individual_scores=scores
                )
                historical_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(historical_metrics)} historical metric files")
        return historical_metrics
    
    def compute_rolling_baseline(self, historical_metrics: List[CanaryMetrics]) -> Dict[str, float]:
        """Compute rolling baseline from recent runs."""
        
        if len(historical_metrics) == 0:
            logger.warning("No historical metrics for baseline computation")
            return {}
        
        # Take last N runs for rolling baseline
        recent_metrics = historical_metrics[-self.rolling_window:]
        
        # Compute baseline statistics
        mean_margins = [m.mean_margin for m in recent_metrics]
        median_margins = [m.median_margin for m in recent_metrics]
        p95_margins = [m.p95_margin for m in recent_metrics]
        positive_accuracies = [m.positive_accuracy for m in recent_metrics]
        
        baseline = {
            'mean_margin_baseline': np.mean(mean_margins),
            'mean_margin_std': np.std(mean_margins),
            'median_margin_baseline': np.mean(median_margins),
            'p95_margin_baseline': np.mean(p95_margins),
            'positive_accuracy_baseline': np.mean(positive_accuracies),
            'runs_in_baseline': len(recent_metrics),
            'baseline_computed_at': datetime.now().isoformat()
        }
        
        return baseline
    
    def detect_drift(self, current_metrics: CanaryMetrics, baseline: Dict[str, float]) -> List[DriftAlert]:
        """Detect drift in current metrics vs baseline."""
        
        alerts = []
        
        if not baseline:
            logger.warning("No baseline available for drift detection")
            return alerts
        
        # 1. Mean drop detection
        baseline_mean = baseline.get('mean_margin_baseline', 0)
        current_mean = current_metrics.mean_margin
        
        if baseline_mean > 0:
            relative_drop = (baseline_mean - current_mean) / baseline_mean
            
            if relative_drop > self.mean_drop_threshold:
                # Find worst-performing images
                margins = [s.clip_margin for s in current_metrics.individual_scores]
                worst_indices = np.argsort(margins)[:10]  # Top 10 worst
                worst_images = [current_metrics.individual_scores[i].image_id for i in worst_indices]
                
                alert = DriftAlert(
                    alert_type='mean_drop',
                    severity='critical' if relative_drop > 0.05 else 'warning',
                    baseline_value=baseline_mean,
                    current_value=current_mean,
                    delta=relative_drop,
                    threshold=self.mean_drop_threshold,
                    confidence=1.0,  # Deterministic check
                    message=f"Mean margin dropped by {relative_drop:.1%} (baseline: {baseline_mean:.3f}, current: {current_mean:.3f})",
                    affected_images=worst_images
                )
                alerts.append(alert)
        
        # 2. Confidence interval exclusion
        baseline_std = baseline.get('mean_margin_std', 0)
        if baseline_std > 0:
            # Compute 95% CI for baseline
            n_baseline = baseline.get('runs_in_baseline', 1)
            se = baseline_std / np.sqrt(n_baseline)
            t_critical = stats.t.ppf(1 - (1 - self.confidence_level) / 2, n_baseline - 1)
            
            ci_lower = baseline_mean - t_critical * se
            ci_upper = baseline_mean + t_critical * se
            
            if not (ci_lower <= current_mean <= ci_upper):
                alert = DriftAlert(
                    alert_type='ci_exclusion',
                    severity='warning',
                    baseline_value=baseline_mean,
                    current_value=current_mean,
                    delta=abs(current_mean - baseline_mean),
                    threshold=t_critical * se,
                    confidence=self.confidence_level,
                    message=f"Current mean {current_mean:.3f} outside 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]",
                    affected_images=[]
                )
                alerts.append(alert)
        
        # 3. Tail drift detection (P95)
        baseline_p95 = baseline.get('p95_margin_baseline', 0)
        current_p95 = current_metrics.p95_margin
        
        if baseline_p95 > 0:
            p95_drop = (baseline_p95 - current_p95) / baseline_p95
            
            if p95_drop > 0.05:  # 5% drop in P95
                alert = DriftAlert(
                    alert_type='tail_drift',
                    severity='warning',
                    baseline_value=baseline_p95,
                    current_value=current_p95,
                    delta=p95_drop,
                    threshold=0.05,
                    confidence=0.95,
                    message=f"P95 margin dropped by {p95_drop:.1%} (baseline: {baseline_p95:.3f}, current: {current_p95:.3f})",
                    affected_images=[]
                )
                alerts.append(alert)
        
        return alerts
    
    def run_canary_check(self, probe_version: str = "v1.0", run_id: str = None) -> Dict[str, Any]:
        """Run complete canary check with drift detection."""
        
        logger.info(f"Starting canary check with probe {probe_version}")
        
        # Load probe set
        probe_set = self.load_probe_set(probe_version)
        
        # Compute CLIP scores (mock for now)
        scores = self.compute_clip_scores(probe_set, mock=True)
        
        # Compute metrics
        metrics = self.compute_metrics(scores, run_id)
        
        # Save metrics
        metrics_file = self.save_metrics(metrics)
        
        # Load historical data and compute baseline
        historical_metrics = self.load_historical_metrics()
        baseline = self.compute_rolling_baseline(historical_metrics)
        
        # Detect drift
        alerts = self.detect_drift(metrics, baseline)
        
        # Compile results
        result = {
            'canary_check': {
                'run_id': metrics.run_id,
                'timestamp': metrics.timestamp,
                'probe_version': probe_version,
                'status': 'PASS' if len(alerts) == 0 else 'ALERT',
                'metrics_file': metrics_file
            },
            'current_metrics': {
                'mean_margin': metrics.mean_margin,
                'median_margin': metrics.median_margin,
                'std_margin': metrics.std_margin,
                'p95_margin': metrics.p95_margin,
                'positive_accuracy': metrics.positive_accuracy,
                'negative_accuracy': metrics.negative_accuracy,
                'total_items': metrics.total_items
            },
            'baseline': baseline,
            'alerts': [
                {
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'baseline_value': alert.baseline_value,
                    'current_value': alert.current_value,
                    'delta': alert.delta,
                    'affected_images': alert.affected_images[:5]  # Limit for display
                }
                for alert in alerts
            ],
            'summary': {
                'total_alerts': len(alerts),
                'critical_alerts': sum(1 for a in alerts if a.severity == 'critical'),
                'warning_alerts': sum(1 for a in alerts if a.severity == 'warning')
            }
        }
        
        logger.info(f"Canary check complete: {result['canary_check']['status']}")
        return result

def main():
    """CLI interface for canary monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Canary Monitor System')
    parser.add_argument('--probe-version', default='v1.0', help='Probe set version')
    parser.add_argument('--run-id', help='Custom run ID')
    parser.add_argument('--baseline-days', type=int, default=7, help='Days of history for baseline')
    parser.add_argument('--export', help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Run canary check
    monitor = CanaryMonitor()
    results = monitor.run_canary_check(args.probe_version, args.run_id)
    
    # Print summary
    print("🕯️  CANARY MONITOR RESULTS")
    print("=" * 50)
    
    check = results['canary_check']
    print(f"Status: {check['status']}")
    print(f"Run ID: {check['run_id']}")
    print(f"Probe Version: {check['probe_version']}")
    
    current = results['current_metrics']
    print(f"\n📊 Current Metrics:")
    print(f"   Mean margin: {current['mean_margin']:.3f}")
    print(f"   Median margin: {current['median_margin']:.3f}")
    print(f"   P95 margin: {current['p95_margin']:.3f}")
    print(f"   Positive accuracy: {current['positive_accuracy']:.1%}")
    print(f"   Negative accuracy: {current['negative_accuracy']:.1%}")
    
    baseline = results['baseline']
    if baseline:
        print(f"\n📈 Baseline (last {baseline['runs_in_baseline']} runs):")
        print(f"   Mean margin: {baseline['mean_margin_baseline']:.3f}")
        print(f"   Positive accuracy: {baseline['positive_accuracy_baseline']:.1%}")
    
    alerts = results['alerts']
    if alerts:
        print(f"\n🚨 Alerts ({len(alerts)} total):")
        for alert in alerts:
            icon = "🔴" if alert['severity'] == 'critical' else "🟡"
            print(f"   {icon} {alert['type']}: {alert['message']}")
            if alert['affected_images']:
                print(f"      Worst images: {', '.join(alert['affected_images'])}")
    else:
        print(f"\n✅ No drift detected")
    
    # Export if requested
    if args.export:
        with open(args.export, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results exported to {args.export}")

if __name__ == "__main__":
    main()