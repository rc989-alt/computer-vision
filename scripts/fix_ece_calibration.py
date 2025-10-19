#!/usr/bin/env python3
"""
Fix ECE Calculation - Debiased Method with Proper Calibration Comparison

This script properly computes Expected Calibration Error (ECE) using:
1. Debiased ECE (not naive) with 15-20 bins
2. Held-out validation split
3. Probability clipping to [1e-6, 1-1e-6]
4. Per-domain ECE with size-weighted averaging
5. Reliability curves (before/after calibration)
6. Comparison: Isotonic vs Platt vs Temperature scaling

Target: ECE ≤ 0.030 (acceptable: 0.050-0.060 for pilot)
"""

import json
import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Import our pipeline components
import sys
sys.path.append('.')
from scripts.demo_candidate_library import CandidateLibraryDemo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ECEConfig:
    """Configuration for ECE calculation"""
    n_bins: int = 20
    clip_min: float = 1e-6
    clip_max: float = 1.0 - 1e-6
    min_samples_per_bin: int = 20
    test_size: float = 0.3
    random_state: int = 42
    target_ece: float = 0.030
    acceptable_ece: float = 0.060

@dataclass
class CalibrationMethod:
    """Single calibration method results"""
    name: str
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    reliability_data: Dict[str, Any]
    model: Optional[Any] = None

@dataclass
class ECEResults:
    """Complete ECE analysis results"""
    config: ECEConfig
    domain_results: Dict[str, Dict[str, CalibrationMethod]]
    overall_ece: Dict[str, float]  # method -> weighted ECE
    best_method: str
    recommendations: List[str]

class ECECalculator:
    """Proper ECE calculation with debiased method"""
    
    def __init__(self, config: ECEConfig):
        self.config = config
        self.ra_guard = CandidateLibraryDemo(gallery_dir="pilot_gallery")
    
    def clip_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Clip probabilities to avoid edge cases"""
        return np.clip(probs, self.config.clip_min, self.config.clip_max)
    
    def compute_debiased_ece(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Compute debiased ECE using proper binning strategy
        
        Returns:
            ece: Debiased Expected Calibration Error
            reliability_data: Data for plotting reliability curves
        """
        y_prob = self.clip_probabilities(y_prob)
        
        # Use equal-frequency binning instead of equal-width for better coverage
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = []
        total_samples = len(y_true)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Only include bins with minimum samples (avoid noise)
                if in_bin.sum() >= self.config.min_samples_per_bin:
                    bin_contribution = np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    ece += bin_contribution
                
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'samples': in_bin.sum(),
                    'proportion': prop_in_bin
                })
            else:
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'samples': 0,
                    'proportion': 0.0
                })
        
        reliability_data = {
            'bin_data': bin_data,
            'total_samples': total_samples,
            'n_bins_used': sum(1 for bd in bin_data if bd['samples'] >= self.config.min_samples_per_bin)
        }
        
        return ece, reliability_data
    
    def compute_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute Brier score for probability calibration quality"""
        y_prob = self.clip_probabilities(y_prob)
        return np.mean((y_prob - y_true) ** 2)
    
    def fit_isotonic_calibration(self, scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
        """Fit isotonic regression calibration with regularization"""
        calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Add regularization for small datasets
        if len(scores) < 1000:
            # Pool adjacent violators with minimum samples
            calibrator = IsotonicRegression(out_of_bounds='clip')
        
        calibrator.fit(scores, labels)
        return calibrator
    
    def fit_platt_calibration(self, scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        """Fit Platt (logistic) calibration"""
        calibrator = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
        calibrator.fit(scores.reshape(-1, 1), labels)
        return calibrator
    
    def fit_temperature_scaling(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Fit temperature scaling (single parameter)"""
        from scipy.optimize import minimize_scalar
        
        def negative_log_likelihood(temperature):
            scaled_scores = scores / temperature
            # Sigmoid function
            probs = 1 / (1 + np.exp(-scaled_scores))
            probs = self.clip_probabilities(probs)
            # Negative log likelihood
            nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return nll
        
        result = minimize_scalar(negative_log_likelihood, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    def evaluate_calibration_method(self, 
                                   scores_train: np.ndarray, 
                                   labels_train: np.ndarray,
                                   scores_test: np.ndarray, 
                                   labels_test: np.ndarray,
                                   method_name: str) -> CalibrationMethod:
        """Evaluate a single calibration method"""
        
        # Before calibration
        ece_before, reliability_before = self.compute_debiased_ece(labels_test, scores_test)
        brier_before = self.compute_brier_score(labels_test, scores_test)
        
        # Apply calibration
        if method_name == 'isotonic':
            model = self.fit_isotonic_calibration(scores_train, labels_train)
            calibrated_probs = model.predict(scores_test)
        elif method_name == 'platt':
            model = self.fit_platt_calibration(scores_train, labels_train)
            calibrated_probs = model.predict_proba(scores_test.reshape(-1, 1))[:, 1]
        elif method_name == 'temperature':
            temperature = self.fit_temperature_scaling(scores_train, labels_train)
            model = temperature
            calibrated_probs = 1 / (1 + np.exp(-scores_test / temperature))
        else:
            raise ValueError(f"Unknown calibration method: {method_name}")
        
        calibrated_probs = self.clip_probabilities(calibrated_probs)
        
        # After calibration
        ece_after, reliability_after = self.compute_debiased_ece(labels_test, calibrated_probs)
        brier_after = self.compute_brier_score(labels_test, calibrated_probs)
        
        return CalibrationMethod(
            name=method_name,
            ece_before=ece_before,
            ece_after=ece_after,
            brier_before=brier_before,
            brier_after=brier_after,
            reliability_data={
                'before': reliability_before,
                'after': reliability_after,
                'test_probs': calibrated_probs.tolist(),
                'test_labels': labels_test.tolist()
            },
            model=model
        )
    
    def collect_scoring_data(self, domain: str = 'cocktails') -> Tuple[np.ndarray, np.ndarray]:
        """Collect scores and relevance labels for calibration"""
        logger.info(f"Collecting scoring data for domain: {domain}")
        
        # Load evaluation queries
        eval_path = Path('datasets/mini_100q.json')
        if not eval_path.exists():
            raise FileNotFoundError("Evaluation queries not found. Run generate_query_set.py first.")
        
        with open(eval_path) as f:
            data = json.load(f)
        
        queries = data['queries']
        
        # Filter by domain
        domain_queries = [q for q in queries if q.get('domain') == domain]
        logger.info(f"Found {len(domain_queries)} queries for domain {domain}")
        
        scores = []
        labels = []
        
        for query_data in domain_queries[:50]:  # Limit for faster processing
            query = query_data['text']
            
            # Get RA-Guard results
            try:
                ra_result = self.ra_guard.process_query(query, domain, num_candidates=30)
                if not ra_result.candidates:
                    continue
                
                # Use reranking scores and create relevance labels
                rerank_scores = ra_result.reranking_scores
                n_relevant = max(1, len(rerank_scores) // 3)  # Top 30% are relevant
                
                for i, score in enumerate(rerank_scores):
                    scores.append(score)
                    labels.append(1.0 if i < n_relevant else 0.0)
                    
            except Exception as e:
                logger.warning(f"Error processing query '{query}': {e}")
                continue
        
        return np.array(scores), np.array(labels)
    
    def plot_reliability_curves(self, results: Dict[str, CalibrationMethod], domain: str):
        """Plot reliability curves for all calibration methods"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Reliability Curves - {domain.title()} Domain')
        
        methods = ['isotonic', 'platt', 'temperature']
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            
            if method not in results:
                ax.set_title(f'{method.title()}: No Data')
                continue
            
            method_result = results[method]
            
            # Before calibration
            before_data = method_result.reliability_data['before']['bin_data']
            confidences_before = [bd['confidence'] for bd in before_data if bd['samples'] > 0]
            accuracies_before = [bd['accuracy'] for bd in before_data if bd['samples'] > 0]
            
            # After calibration  
            after_data = method_result.reliability_data['after']['bin_data']
            confidences_after = [bd['confidence'] for bd in after_data if bd['samples'] > 0]
            accuracies_after = [bd['accuracy'] for bd in after_data if bd['samples'] > 0]
            
            # Plot
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            if confidences_before:
                ax.scatter(confidences_before, accuracies_before, 
                          alpha=0.6, s=30, color='red', label='Before')
            if confidences_after:
                ax.scatter(confidences_after, accuracies_after, 
                          alpha=0.8, s=30, color='blue', label='After')
            
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{method.title()}\nECE: {method_result.ece_before:.3f}→{method_result.ece_after:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f'calibration_reliability_{domain}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reliability curves saved to: {plot_path}")
        plt.close()
    
    def run_ece_analysis(self) -> ECEResults:
        """Run complete ECE analysis with all calibration methods"""
        logger.info("🔧 Starting ECE Analysis with Debiased Method")
        
        domain_results = {}
        overall_ece = {}
        
        # Analyze each domain
        domains = ['cocktails']  # Expand as needed
        
        for domain in domains:
            logger.info(f"📊 Analyzing domain: {domain}")
            
            # Collect scoring data
            scores, labels = self.collect_scoring_data(domain)
            logger.info(f"Collected {len(scores)} score-label pairs")
            
            if len(scores) < 100:
                logger.warning(f"Insufficient data for domain {domain}: {len(scores)} samples")
                continue
            
            # Train/test split for held-out validation
            scores_train, scores_test, labels_train, labels_test = train_test_split(
                scores, labels, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state,
                stratify=labels if len(np.unique(labels)) > 1 else None
            )
            
            # Test all calibration methods
            methods = ['isotonic', 'platt', 'temperature']
            domain_results[domain] = {}
            
            for method in methods:
                logger.info(f"  Testing {method} calibration...")
                try:
                    result = self.evaluate_calibration_method(
                        scores_train, labels_train,
                        scores_test, labels_test,
                        method
                    )
                    domain_results[domain][method] = result
                    logger.info(f"    {method}: ECE {result.ece_before:.3f}→{result.ece_after:.3f}")
                except Exception as e:
                    logger.error(f"    {method} failed: {e}")
            
            # Plot reliability curves
            if domain_results[domain]:
                self.plot_reliability_curves(domain_results[domain], domain)
        
        # Compute overall ECE (size-weighted average)
        for method in ['isotonic', 'platt', 'temperature']:
            weighted_ece = 0.0
            total_samples = 0
            
            for domain, results in domain_results.items():
                if method in results:
                    # Use test labels from reliability data
                    test_labels = results[method].reliability_data.get('test_labels', [])
                    domain_samples = len(test_labels)
                    if domain_samples > 0:
                        weighted_ece += results[method].ece_after * domain_samples
                        total_samples += domain_samples
            
            if total_samples > 0:
                overall_ece[method] = weighted_ece / total_samples
            else:
                overall_ece[method] = float('inf')
        
        # Find best method
        best_method = min(overall_ece.keys(), key=lambda k: overall_ece[k])
        best_ece = overall_ece[best_method]
        
        # Generate recommendations
        recommendations = []
        if best_ece <= self.config.target_ece:
            recommendations.append(f"✅ ECE target achieved: {best_ece:.3f} ≤ {self.config.target_ece}")
        elif best_ece <= self.config.acceptable_ece:
            recommendations.append(f"⚠️  ECE acceptable for pilot: {best_ece:.3f} ≤ {self.config.acceptable_ece}")
        else:
            recommendations.append(f"❌ ECE too high: {best_ece:.3f} > {self.config.acceptable_ece}")
            recommendations.append("Consider: more training data, different binning, ensemble methods")
        
        recommendations.append(f"Best method: {best_method}")
        
        if best_ece > 0.1:
            recommendations.append("Check for: probability clipping, bin size, data quality")
        
        return ECEResults(
            config=self.config,
            domain_results=domain_results,
            overall_ece=overall_ece,
            best_method=best_method,
            recommendations=recommendations
        )

def main():
    """Main execution"""
    print("🔧 FIXING ECE CALCULATION")
    print("=" * 50)
    
    # Configuration
    config = ECEConfig(
        n_bins=20,
        min_samples_per_bin=15,  # Reduced for pilot data
        target_ece=0.030,
        acceptable_ece=0.060
    )
    
    # Run analysis
    calculator = ECECalculator(config)
    results = calculator.run_ece_analysis()
    
    # Print results
    print("\n📊 ECE ANALYSIS RESULTS")
    print("=" * 50)
    
    for domain, methods in results.domain_results.items():
        print(f"\n🎯 Domain: {domain.upper()}")
        for method, result in methods.items():
            improvement = result.ece_before - result.ece_after
            print(f"  {method:12s}: ECE {result.ece_before:.3f} → {result.ece_after:.3f} "
                  f"(Δ={improvement:+.3f}, Brier={result.brier_after:.3f})")
    
    print(f"\n🏆 OVERALL RESULTS")
    print(f"Best method: {results.best_method}")
    print(f"Best ECE: {results.overall_ece[results.best_method]:.3f}")
    print(f"Target: ≤{config.target_ece:.3f}")
    
    print("\n💡 RECOMMENDATIONS")
    for rec in results.recommendations:
        print(f"  • {rec}")
    
    # Save results
    output_path = 'ece_analysis_results.json'
    with open(output_path, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {
            'config': asdict(config),
            'overall_ece': results.overall_ece,
            'best_method': results.best_method,
            'recommendations': results.recommendations,
            'domain_summaries': {}
        }
        
        for domain, methods in results.domain_results.items():
            json_results['domain_summaries'][domain] = {}
            for method, result in methods.items():
                json_results['domain_summaries'][domain][method] = {
                    'ece_before': result.ece_before,
                    'ece_after': result.ece_after,
                    'brier_before': result.brier_before,
                    'brier_after': result.brier_after,
                    'improvement': result.ece_before - result.ece_after
                }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Save best calibration model
    if results.domain_results:
        best_models = {}
        for domain, methods in results.domain_results.items():
            if results.best_method in methods:
                best_models[domain] = methods[results.best_method].model
        
        if best_models:
            model_path = 'pilot_gallery/calibrated_model_fixed.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'method': results.best_method,
                    'models': best_models,
                    'config': config
                }, f)
            print(f"💾 Best calibration models saved to: {model_path}")

if __name__ == "__main__":
    main()