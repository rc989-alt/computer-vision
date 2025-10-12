#!/usr/bin/env python3
"""
Evaluation Suite for CoTRR-lite Reranker

Comprehensive evaluation with bootstrap confidence intervals:
- Compliance@1/3/5
- nDCG@10  
- Conflict AUC/ECE
- Failure analysis with explanations

Integrates with Step 4 A/B testing framework.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import torch
from sklearn.metrics import roc_auc_score, ndcg_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from reranker_train import CoTRRLiteReranker, ModelConfig
from feature_extractor import FeatureExtractor, FeatureConfig

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Comprehensive evaluation results"""
    # Ranking metrics
    compliance_at_1: float
    compliance_at_3: float  
    compliance_at_5: float
    ndcg_at_10: float
    
    # Conflict metrics
    conflict_auc: float
    conflict_ece: float
    conflict_brier: float
    
    # Confidence intervals (95%)
    compliance_at_1_ci: Tuple[float, float]
    compliance_at_3_ci: Tuple[float, float]
    ndcg_at_10_ci: Tuple[float, float]
    conflict_auc_ci: Tuple[float, float]
    
    # Additional metrics
    mean_reciprocal_rank: float
    precision_at_5: float
    recall_at_5: float
    
    # Failure analysis
    failure_cases: List[Dict[str, Any]]
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BootstrapEvaluator:
    """Bootstrap confidence interval evaluator"""
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def bootstrap_metric(self, metric_fn, *args, **kwargs) -> Tuple[float, Tuple[float, float]]:
        """Compute metric with bootstrap confidence interval"""
        # Original metric
        original_metric = metric_fn(*args, **kwargs)
        
        # Bootstrap samples
        n_samples = len(args[0]) if args else len(kwargs[list(kwargs.keys())[0]])
        bootstrap_metrics = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Apply indices to all arguments
            bootstrap_args = []
            for arg in args:
                if hasattr(arg, '__len__') and len(arg) == n_samples:
                    bootstrap_args.append([arg[i] for i in indices])
                else:
                    bootstrap_args.append(arg)
            
            bootstrap_kwargs = {}
            for key, value in kwargs.items():
                if hasattr(value, '__len__') and len(value) == n_samples:
                    bootstrap_kwargs[key] = [value[i] for i in indices]
                else:
                    bootstrap_kwargs[key] = value
            
            # Compute bootstrap metric
            try:
                bootstrap_metric = metric_fn(*bootstrap_args, **bootstrap_kwargs)
                bootstrap_metrics.append(bootstrap_metric)
            except:
                continue  # Skip failed bootstrap samples
        
        # Compute confidence interval
        if bootstrap_metrics:
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
            ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
            ci = (ci_lower, ci_upper)
        else:
            ci = (original_metric, original_metric)
        
        return original_metric, ci

class RankingMetrics:
    """Ranking evaluation metrics"""
    
    @staticmethod
    def compliance_at_k(y_true: List[float], y_pred: List[float], k: int) -> float:
        """Compliance@K metric"""
        if len(y_true) == 0:
            return 0.0
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        
        # Take top-k
        top_k_indices = sorted_indices[:k]
        top_k_true = [y_true[i] for i in top_k_indices]
        
        # Compliance@K is the mean of top-K true scores
        return np.mean(top_k_true)
    
    @staticmethod
    def ndcg_at_k(y_true: List[float], y_pred: List[float], k: int) -> float:
        """nDCG@K metric"""
        if len(y_true) == 0:
            return 0.0
        
        try:
            # Reshape for sklearn
            y_true_array = np.array(y_true).reshape(1, -1)
            y_pred_array = np.array(y_pred).reshape(1, -1)
            
            return ndcg_score(y_true_array, y_pred_array, k=k)
        except:
            return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(y_true: List[float], y_pred: List[float], threshold: float = 0.7) -> float:
        """Mean Reciprocal Rank for relevant items (compliance > threshold)"""
        if len(y_true) == 0:
            return 0.0
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        
        # Find first relevant item
        for rank, idx in enumerate(sorted_indices):
            if y_true[idx] >= threshold:
                return 1.0 / (rank + 1)
        
        return 0.0  # No relevant items found
    
    @staticmethod
    def precision_recall_at_k(y_true: List[float], y_pred: List[float], k: int, threshold: float = 0.7) -> Tuple[float, float]:
        """Precision and Recall@K"""
        if len(y_true) == 0:
            return 0.0, 0.0
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        top_k_indices = sorted_indices[:k]
        
        # Relevant items
        relevant_items = [i for i, score in enumerate(y_true) if score >= threshold]
        top_k_relevant = [i for i in top_k_indices if i in relevant_items]
        
        # Precision@K
        precision = len(top_k_relevant) / k if k > 0 else 0.0
        
        # Recall@K
        recall = len(top_k_relevant) / len(relevant_items) if len(relevant_items) > 0 else 0.0
        
        return precision, recall

class ConflictMetrics:
    """Conflict prediction evaluation metrics"""
    
    @staticmethod
    def conflict_auc(conflict_probs: List[float], conflict_labels: List[int]) -> float:
        """AUC for conflict prediction"""
        if len(set(conflict_labels)) < 2:
            return 0.5  # Random performance if only one class
        
        try:
            return roc_auc_score(conflict_labels, conflict_probs)
        except:
            return 0.5
    
    @staticmethod
    def expected_calibration_error(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)"""
        if len(probs) == 0:
            return 0.0
        
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find samples in this bin
                in_bin = (probs >= bin_lower) & (probs < bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Accuracy and confidence in this bin
                    accuracy_in_bin = labels[in_bin].mean()
                    avg_confidence_in_bin = probs[in_bin].mean()
                    
                    # ECE contribution
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        except:
            return 0.0
    
    @staticmethod
    def brier_score(probs: List[float], labels: List[int]) -> float:
        """Brier score for probability calibration"""
        try:
            return brier_score_loss(labels, probs)
        except:
            return 0.5

class FailureAnalyzer:
    """Analyze failure cases and provide explanations"""
    
    def __init__(self, items: List[Dict], features: np.ndarray, 
                 predictions: np.ndarray, labels: np.ndarray):
        self.items = items
        self.features = features
        self.predictions = predictions
        self.labels = labels
    
    def analyze_failures(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Identify and analyze failure cases"""
        failures = []
        
        # Group by query for ranking analysis
        query_groups = {}
        for i, item in enumerate(self.items):
            query = item.get('query', 'unknown')
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((i, self.labels[i], self.predictions[i]))
        
        for query, group_items in query_groups.items():
            if len(group_items) < 2:
                continue
            
            # Sort by true labels (descending)
            group_items.sort(key=lambda x: x[1], reverse=True)
            
            # Check if ranking is correct
            true_ranking = [x[0] for x in group_items]
            pred_ranking = sorted(group_items, key=lambda x: x[2], reverse=True)
            pred_ranking = [x[0] for x in pred_ranking]
            
            # Compute ranking error
            ranking_error = self._compute_ranking_error(true_ranking, pred_ranking)
            
            if ranking_error > threshold:
                # This is a failure case
                failure_case = self._analyze_failure_case(
                    query, group_items, true_ranking, pred_ranking, ranking_error
                )
                failures.append(failure_case)
        
        return failures
    
    def _compute_ranking_error(self, true_ranking: List[int], pred_ranking: List[int]) -> float:
        """Compute ranking error (normalized Kendall's tau distance)"""
        if len(true_ranking) <= 1:
            return 0.0
        
        # Count inversions
        inversions = 0
        total_pairs = 0
        
        for i in range(len(true_ranking)):
            for j in range(i + 1, len(true_ranking)):
                total_pairs += 1
                
                # Find positions in predicted ranking
                pos_i = pred_ranking.index(true_ranking[i])
                pos_j = pred_ranking.index(true_ranking[j])
                
                # Check if order is inverted
                if pos_i > pos_j:  # Should be pos_i < pos_j
                    inversions += 1
        
        return inversions / total_pairs if total_pairs > 0 else 0.0
    
    def _analyze_failure_case(self, query: str, group_items: List[Tuple], 
                             true_ranking: List[int], pred_ranking: List[int], 
                             ranking_error: float) -> Dict[str, Any]:
        """Analyze a specific failure case"""
        # Get the most problematic pair (highest-ranked item that should be lower)
        worst_inversion = None
        max_inversion_score = 0
        
        for i, true_idx in enumerate(true_ranking):
            pred_pos = pred_ranking.index(true_idx)
            if pred_pos > i:  # Predicted position is worse than true position
                inversion_score = pred_pos - i
                if inversion_score > max_inversion_score:
                    max_inversion_score = inversion_score
                    worst_inversion = (true_idx, i, pred_pos)
        
        failure_analysis = {
            'query': query,
            'ranking_error': ranking_error,
            'num_items': len(group_items),
            'worst_inversion': None,
            'explanation': "Unknown failure cause",
            'suggested_fix': "Review feature extraction or model architecture"
        }
        
        if worst_inversion:
            item_idx, true_pos, pred_pos = worst_inversion
            item = self.items[item_idx]
            
            failure_analysis['worst_inversion'] = {
                'item_id': item.get('id', 'unknown'),
                'true_position': true_pos + 1,
                'predicted_position': pred_pos + 1,
                'true_score': float(self.labels[item_idx]),
                'predicted_score': float(self.predictions[item_idx])
            }
            
            # Analyze potential causes
            explanation, suggested_fix = self._explain_failure(item, item_idx)
            failure_analysis['explanation'] = explanation
            failure_analysis['suggested_fix'] = suggested_fix
        
        return failure_analysis
    
    def _explain_failure(self, item: Dict, item_idx: int) -> Tuple[str, str]:
        """Explain why this item failed and suggest fixes"""
        explanations = []
        suggestions = []
        
        # Check conflict score
        conflict_score = item.get('conflict_score', 0.0)
        if conflict_score > 0.5:
            explanations.append(f"High conflict score ({conflict_score:.3f})")
            suggestions.append("Review conflict detection calibration")
        
        # Check compliance score
        compliance_score = item.get('compliance_score', 0.5)
        if compliance_score < 0.7:
            explanations.append(f"Low compliance score ({compliance_score:.3f})")
            suggestions.append("Check compliance model calibration")
        
        # Check visual features
        subject_ratio = item.get('subject_ratio', 0.5)
        if subject_ratio < 0.3:
            explanations.append(f"Low subject ratio ({subject_ratio:.3f})")
            suggestions.append("Improve subject detection or penalty")
        
        # Default explanation
        if not explanations:
            explanations.append("Feature mismatch or model uncertainty")
            suggestions.append("Add more training data or regularization")
        
        explanation = "; ".join(explanations)
        suggested_fix = "; ".join(suggestions)
        
        return explanation, suggested_fix

class CoTRREvaluator:
    """Comprehensive evaluator for CoTRR-lite reranker"""
    
    def __init__(self, bootstrap_samples: int = 1000):
        self.bootstrap_evaluator = BootstrapEvaluator(n_bootstrap=bootstrap_samples)
        self.ranking_metrics = RankingMetrics()
        self.conflict_metrics = ConflictMetrics()
    
    def evaluate_model(self, model: CoTRRLiteReranker, items: List[Dict], 
                      features: np.ndarray, labels: np.ndarray, 
                      device: str = "cpu") -> EvaluationResults:
        """Comprehensive model evaluation"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Get model predictions
        model.eval()
        features_tensor = torch.FloatTensor(features).to(device)
        
        with torch.no_grad():
            predictions = model.predict_batch(features_tensor).cpu().numpy()
        
        # Group predictions by query for ranking evaluation
        query_groups = self._group_by_query(items, labels, predictions)
        
        # Compute ranking metrics with confidence intervals
        compliance_1, compliance_1_ci = self.bootstrap_evaluator.bootstrap_metric(
            self._evaluate_compliance_at_k_across_queries, query_groups, 1
        )
        
        compliance_3, compliance_3_ci = self.bootstrap_evaluator.bootstrap_metric(
            self._evaluate_compliance_at_k_across_queries, query_groups, 3
        )
        
        compliance_5, compliance_5_ci = self.bootstrap_evaluator.bootstrap_metric(
            self._evaluate_compliance_at_k_across_queries, query_groups, 5
        )
        
        ndcg_10, ndcg_10_ci = self.bootstrap_evaluator.bootstrap_metric(
            self._evaluate_ndcg_at_k_across_queries, query_groups, 10
        )
        
        # Conflict metrics
        conflict_probs = [item.get('conflict_prob', 0.0) for item in items]
        conflict_labels = [1 if item.get('conflict_score', 0.0) > 0.3 else 0 for item in items]
        
        conflict_auc, conflict_auc_ci = self.bootstrap_evaluator.bootstrap_metric(
            self.conflict_metrics.conflict_auc, conflict_probs, conflict_labels
        )
        
        conflict_ece = self.conflict_metrics.expected_calibration_error(
            np.array(conflict_probs), np.array(conflict_labels)
        )
        
        conflict_brier = self.conflict_metrics.brier_score(conflict_probs, conflict_labels)
        
        # Additional metrics
        mrr = np.mean([
            self.ranking_metrics.mean_reciprocal_rank(group['labels'], group['predictions'])
            for group in query_groups.values()
        ])
        
        precision_5, recall_5 = zip(*[
            self.ranking_metrics.precision_recall_at_k(group['labels'], group['predictions'], 5)
            for group in query_groups.values()
        ])
        
        # Failure analysis
        failure_analyzer = FailureAnalyzer(items, features, predictions, labels)
        failure_cases = failure_analyzer.analyze_failures()
        success_rate = 1.0 - len(failure_cases) / max(len(query_groups), 1)
        
        # Create results
        results = EvaluationResults(
            compliance_at_1=compliance_1,
            compliance_at_3=compliance_3,
            compliance_at_5=compliance_5,
            ndcg_at_10=ndcg_10,
            compliance_at_1_ci=compliance_1_ci,
            compliance_at_3_ci=compliance_3_ci,
            ndcg_at_10_ci=ndcg_10_ci,
            conflict_auc=conflict_auc,
            conflict_auc_ci=conflict_auc_ci,
            conflict_ece=conflict_ece,
            conflict_brier=conflict_brier,
            mean_reciprocal_rank=mrr,
            precision_at_5=np.mean(precision_5),
            recall_at_5=np.mean(recall_5),
            failure_cases=failure_cases,
            success_rate=success_rate
        )
        
        logger.info("Model evaluation completed")
        return results
    
    def _group_by_query(self, items: List[Dict], labels: np.ndarray, 
                       predictions: np.ndarray) -> Dict[str, Dict]:
        """Group items by query for ranking evaluation"""
        query_groups = {}
        
        for i, item in enumerate(items):
            query = item.get('query', 'unknown')
            if query not in query_groups:
                query_groups[query] = {'labels': [], 'predictions': [], 'items': []}
            
            query_groups[query]['labels'].append(labels[i])
            query_groups[query]['predictions'].append(predictions[i])
            query_groups[query]['items'].append(item)
        
        return query_groups
    
    def _evaluate_compliance_at_k_across_queries(self, query_groups: Dict, k: int) -> float:
        """Evaluate Compliance@K across all queries"""
        compliance_scores = []
        
        for group in query_groups.values():
            if len(group['labels']) >= k:
                compliance = self.ranking_metrics.compliance_at_k(
                    group['labels'], group['predictions'], k
                )
                compliance_scores.append(compliance)
        
        return np.mean(compliance_scores) if compliance_scores else 0.0
    
    def _evaluate_ndcg_at_k_across_queries(self, query_groups: Dict, k: int) -> float:
        """Evaluate nDCG@K across all queries"""
        ndcg_scores = []
        
        for group in query_groups.values():
            if len(group['labels']) >= 2:  # Need at least 2 items for nDCG
                ndcg = self.ranking_metrics.ndcg_at_k(
                    group['labels'], group['predictions'], k
                )
                ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

def generate_evaluation_report(results: EvaluationResults, output_path: str):
    """Generate comprehensive evaluation report"""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create report
    report = {
        'evaluation_summary': {
            'compliance_at_1': f"{results.compliance_at_1:.4f} ± {(results.compliance_at_1_ci[1] - results.compliance_at_1_ci[0])/2:.4f}",
            'compliance_at_3': f"{results.compliance_at_3:.4f} ± {(results.compliance_at_3_ci[1] - results.compliance_at_3_ci[0])/2:.4f}",
            'ndcg_at_10': f"{results.ndcg_at_10:.4f} ± {(results.ndcg_at_10_ci[1] - results.ndcg_at_10_ci[0])/2:.4f}",
            'conflict_auc': f"{results.conflict_auc:.4f} ± {(results.conflict_auc_ci[1] - results.conflict_auc_ci[0])/2:.4f}",
            'conflict_ece': f"{results.conflict_ece:.4f}",
            'success_rate': f"{results.success_rate:.4f}"
        },
        'detailed_metrics': results.to_dict(),
        'failure_analysis': {
            'total_failures': len(results.failure_cases),
            'failure_rate': 1.0 - results.success_rate,
            'common_failure_patterns': []
        }
    }
    
    # Analyze failure patterns
    if results.failure_cases:
        failure_explanations = [case['explanation'] for case in results.failure_cases]
        from collections import Counter
        explanation_counts = Counter(failure_explanations)
        
        for explanation, count in explanation_counts.most_common(5):
            report['failure_analysis']['common_failure_patterns'].append({
                'pattern': explanation,
                'frequency': count,
                'percentage': count / len(results.failure_cases)
            })
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {report_path}")

def demo_evaluation():
    """Demo evaluation on trained models"""
    print("📊 CoTRR-lite Evaluation Demo")
    print("=" * 30)
    
    # This would normally load a trained model and test data
    # For demo, we'll create mock predictions
    
    # Mock test data
    n_test = 200
    items = []
    labels = []
    features = []
    
    for i in range(n_test):
        item = {
            'id': f'test_{i:03d}',
            'query': f'cocktail query {i % 10}',
            'domain': ['blue_tropical', 'red_berry'][i % 2],
            'compliance_score': np.random.uniform(0.5, 1.0),
            'conflict_prob': np.random.uniform(0.0, 0.5),
            'conflict_score': np.random.uniform(0.0, 0.8)
        }
        items.append(item)
        
        # Mock label (dual score)
        label = 0.7 * item['compliance_score'] + 0.3 * (1 - item['conflict_prob'])
        labels.append(label)
        
        # Mock features
        feature = np.random.normal(0, 1, 1029)  # 1024 CLIP + 5 visual
        features.append(feature)
    
    labels = np.array(labels)
    features = np.array(features)
    
    # Mock model (simple linear model for demo)
    from reranker_train import CoTRRLiteReranker, ModelConfig
    
    config = ModelConfig(hidden_dims=[64, 1])
    model = CoTRRLiteReranker(features.shape[1], config)
    
    # Evaluate
    evaluator = CoTRREvaluator(bootstrap_samples=100)  # Reduced for demo
    results = evaluator.evaluate_model(model, items, features, labels)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Compliance@1: {results.compliance_at_1:.4f} ({results.compliance_at_1_ci[0]:.4f}, {results.compliance_at_1_ci[1]:.4f})")
    print(f"   Compliance@3: {results.compliance_at_3:.4f} ({results.compliance_at_3_ci[0]:.4f}, {results.compliance_at_3_ci[1]:.4f})")
    print(f"   nDCG@10: {results.ndcg_at_10:.4f} ({results.ndcg_at_10_ci[0]:.4f}, {results.ndcg_at_10_ci[1]:.4f})")
    print(f"   Conflict AUC: {results.conflict_auc:.4f} ({results.conflict_auc_ci[0]:.4f}, {results.conflict_auc_ci[1]:.4f})")
    print(f"   Conflict ECE: {results.conflict_ece:.4f}")
    print(f"   Success Rate: {results.success_rate:.4f}")
    
    # Generate report
    report_path = "research/reports/demo_evaluation.json"
    generate_evaluation_report(results, report_path)
    
    print(f"\n📝 Report saved to: {report_path}")
    print(f"✅ Evaluation demo complete!")
    
    return results

if __name__ == "__main__":
    demo_evaluation()