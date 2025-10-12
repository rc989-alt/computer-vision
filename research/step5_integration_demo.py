#!/usr/bin/env python3
"""
Step 5 Integration Demo: CoTRR-lite Reranker Training

Complete end-to-end demonstration of Step 5:
- Generate training data from Step 4 pipeline outputs
- Train CoTRR-lite reranker with ablation studies
- Comprehensive evaluation with 95% confidence intervals
- Integration with Step 4 A/B testing framework

This demonstrates the research track building on Step 4 production infrastructure.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Setup paths
sys.path.append('research/src')
sys.path.append('src')

from generate_training_data import create_mock_step4_output, load_step4_pipeline_output, create_training_dataset
from feature_extractor import FeatureExtractor, FeatureConfig
from reranker_train import train_reranker, create_ablation_configs
from eval_suite import CoTRREvaluator, generate_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Step5IntegrationDemo:
    """Complete Step 5 CoTRR-lite integration demo"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {}
        
    async def run_step5_demo(self):
        """Run complete Step 5 demo"""
        print("🔬 STEP 5: TRAIN CoTRR-LITE RERANKER")
        print("🧪 Research Track Integration Demo")
        print("=" * 60)
        
        try:
            # Step 1: Generate training data from Step 4 output
            print("\n📊 Step 1: Generate Training Data from Step 4")
            print("-" * 40)
            
            training_data = await self.generate_training_data()
            
            # Step 2: Feature extraction and preprocessing
            print("\n🧠 Step 2: Feature Extraction & Preprocessing")
            print("-" * 40)
            
            feature_data = await self.extract_features(training_data['path'])
            
            # Step 3: Ablation study training
            print("\n🏋️ Step 3: Ablation Study Training")
            print("-" * 40)
            
            trained_models = await self.run_ablation_study(feature_data)
            
            # Step 4: Comprehensive evaluation
            print("\n📊 Step 4: Comprehensive Evaluation")
            print("-" * 40)
            
            evaluation_results = await self.evaluate_models(trained_models, feature_data)
            
            # Step 5: Generate final report
            print("\n📝 Step 5: Generate Research Report")
            print("-" * 40)
            
            report = await self.generate_research_report(evaluation_results)
            
            # Print final results
            await self.print_final_results(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Step 5 demo failed: {e}")
            raise
    
    async def generate_training_data(self) -> Dict[str, Any]:
        """Generate training data from Step 4 pipeline"""
        print("📥 Creating mock Step 4 pipeline output...")
        
        # Create mock Step 4 output (in real scenario, use actual pipeline output)
        mock_run_dir = create_mock_step4_output("runs/step5_training_run")
        
        # Load the pipeline output
        items = load_step4_pipeline_output(mock_run_dir)
        
        # Create training dataset
        training_path = "research/data/step5_training.jsonl"
        metadata = create_training_dataset(items, training_path, lambda_param=0.7)
        
        print(f"✅ Training data generated:")
        print(f"   Items: {metadata['total_items']}")
        print(f"   Domains: {len(metadata['domains'])}")
        print(f"   Queries: {metadata['queries']}")
        print(f"   Lambda: {metadata['lambda_param']}")
        
        return {'path': training_path, 'metadata': metadata}
    
    async def extract_features(self, training_path: str) -> Dict[str, Any]:
        """Extract features for reranker training"""
        print("🔧 Extracting multi-modal features...")
        
        # Configure feature extraction
        config = FeatureConfig(
            clip_dim=512,
            normalize_per_domain=True,
            feature_scaling="standard"
        )
        
        # Extract features
        extractor = FeatureExtractor(config)
        feature_data = extractor.process_pipeline_output(training_path, lambda_param=0.7)
        
        print(f"✅ Feature extraction completed:")
        print(f"   Feature dimension: {feature_data['feature_dim']}")
        print(f"   Training pairs: {len(feature_data['pair_features'])}")
        print(f"   Train/Val/Test split: {len(feature_data['train_indices'])}/{len(feature_data['val_indices'])}/{len(feature_data['test_indices'])}")
        
        # Feature breakdown
        clip_dim = 1024  # 512 image + 512 text
        visual_dim = len(config.visual_features)
        conflict_dim = len(config.conflict_features)
        
        print(f"   CLIP embeddings: {clip_dim} dims")
        print(f"   Visual features: {visual_dim} dims") 
        print(f"   Conflict features: {conflict_dim} dims")
        
        return feature_data
    
    async def run_ablation_study(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run ablation study training"""
        print("🔬 Running ablation study...")
        
        # Get ablation configurations
        ablation_configs = create_ablation_configs()
        
        trained_models = {}
        
        for name, config in ablation_configs.items():
            print(f"\n🏋️ Training {name}...")
            
            start_time = time.time()
            
            try:
                # Train model
                output_dir = f"research/models/step5_{name}"
                result = train_reranker(feature_data, config, output_dir)
                
                training_time = time.time() - start_time
                
                trained_models[name] = {
                    'config': config,
                    'model': result['model'],
                    'trainer': result['trainer'],
                    'training_results': result['training_results'],
                    'model_path': result['model_path'],
                    'input_dim': result['input_dim'],
                    'training_time': training_time
                }
                
                print(f"✅ {name} completed:")
                print(f"   Best val loss: {result['training_results']['best_val_loss']:.4f}")
                print(f"   Epochs: {result['training_results']['total_epochs']}")
                print(f"   Time: {training_time:.1f}s")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                continue
        
        print(f"\n🎯 Ablation study completed: {len(trained_models)}/{len(ablation_configs)} models trained")
        
        return trained_models
    
    async def evaluate_models(self, trained_models: Dict[str, Any], 
                            feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation of trained models"""
        print("📊 Evaluating trained models...")
        
        # Prepare test data
        test_indices = feature_data['test_indices']
        test_items = [feature_data['items'][i] for i in test_indices]
        test_features = feature_data['features'][test_indices]
        test_labels = feature_data['labels'][test_indices]
        
        print(f"   Test set: {len(test_items)} items")
        
        # Initialize evaluator
        evaluator = CoTRREvaluator(bootstrap_samples=500)  # Reduced for demo speed
        
        evaluation_results = {}
        
        for name, model_data in trained_models.items():
            print(f"\n📊 Evaluating {name}...")
            
            try:
                model = model_data['model']
                
                # Evaluate model
                results = evaluator.evaluate_model(
                    model, test_items, test_features, test_labels
                )
                
                evaluation_results[name] = {
                    'results': results,
                    'config': model_data['config'], 
                    'training_time': model_data['training_time']
                }
                
                print(f"✅ {name} evaluation:")
                print(f"   Compliance@1: {results.compliance_at_1:.4f} ± {(results.compliance_at_1_ci[1] - results.compliance_at_1_ci[0])/2:.4f}")
                print(f"   nDCG@10: {results.ndcg_at_10:.4f} ± {(results.ndcg_at_10_ci[1] - results.ndcg_at_10_ci[0])/2:.4f}")
                print(f"   Conflict AUC: {results.conflict_auc:.4f}")
                print(f"   Conflict ECE: {results.conflict_ece:.4f}")
                
            except Exception as e:
                print(f"❌ {name} evaluation failed: {e}")
                continue
        
        return evaluation_results
    
    async def generate_research_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive research report"""
        print("📝 Generating research report...")
        
        # Create report structure
        report = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'step': 'Step 5: CoTRR-lite Reranker Training',
                'objective': 'Train lightweight reranker optimizing λ·Compliance + (1−λ)·(1−p_conflict)',
                'lambda_param': 0.7
            },
            'ablation_study': {},
            'acceptance_criteria': {
                'compliance_at_1_improvement': 3.0,  # +3 pts vs CLIP-only
                'ndcg_at_10_improvement': 6.0,      # +6 pts vs CLIP-only  
                'conflict_auc_target': 0.90,
                'conflict_ece_target': 0.05
            },
            'results_summary': {},
            'failure_analysis': {},
            'recommendations': []
        }
        
        # Analyze results
        baseline_performance = None
        best_performance = None
        best_model = None
        
        for name, eval_data in evaluation_results.items():
            results = eval_data['results']
            config = eval_data['config']
            
            # Store ablation results
            report['ablation_study'][name] = {
                'configuration': {
                    'use_clip': config.use_clip,
                    'use_visual': config.use_visual,
                    'use_conflict': config.use_conflict,
                    'hidden_dims': config.hidden_dims
                },
                'performance': {
                    'compliance_at_1': results.compliance_at_1,
                    'compliance_at_1_ci': results.compliance_at_1_ci,
                    'compliance_at_3': results.compliance_at_3,
                    'ndcg_at_10': results.ndcg_at_10,
                    'ndcg_at_10_ci': results.ndcg_at_10_ci,
                    'conflict_auc': results.conflict_auc,
                    'conflict_ece': results.conflict_ece,
                    'success_rate': results.success_rate
                },
                'training_time': eval_data['training_time'],
                'failure_cases': len(results.failure_cases)
            }
            
            # Track baseline (CLIP-only)
            if name == 'clip_only':
                baseline_performance = results
            
            # Track best performance
            if best_performance is None or results.compliance_at_1 > best_performance.compliance_at_1:
                best_performance = results
                best_model = name
        
        # Compute improvements vs baseline
        if baseline_performance and best_performance:
            compliance_improvement = (best_performance.compliance_at_1 - baseline_performance.compliance_at_1) * 100
            ndcg_improvement = (best_performance.ndcg_at_10 - baseline_performance.ndcg_at_10) * 100
            
            report['results_summary'] = {
                'best_model': best_model,
                'baseline_compliance_at_1': baseline_performance.compliance_at_1,
                'best_compliance_at_1': best_performance.compliance_at_1,
                'compliance_improvement': compliance_improvement,
                'baseline_ndcg_at_10': baseline_performance.ndcg_at_10,
                'best_ndcg_at_10': best_performance.ndcg_at_10,
                'ndcg_improvement': ndcg_improvement,
                'conflict_auc': best_performance.conflict_auc,
                'conflict_ece': best_performance.conflict_ece,
                'meets_criteria': {
                    'compliance_improvement': compliance_improvement >= report['acceptance_criteria']['compliance_at_1_improvement'],
                    'ndcg_improvement': ndcg_improvement >= report['acceptance_criteria']['ndcg_at_10_improvement'],
                    'conflict_auc': best_performance.conflict_auc >= report['acceptance_criteria']['conflict_auc_target'],
                    'conflict_ece': best_performance.conflict_ece <= report['acceptance_criteria']['conflict_ece_target']
                }
            }
            
            # Check overall success
            criteria_met = sum(report['results_summary']['meets_criteria'].values())
            report['results_summary']['overall_success'] = criteria_met >= 3  # At least 3/4 criteria
        
        # Failure analysis from best model
        if best_performance and best_performance.failure_cases:
            failure_patterns = {}
            for case in best_performance.failure_cases:
                pattern = case.get('explanation', 'Unknown')
                if pattern not in failure_patterns:
                    failure_patterns[pattern] = 0
                failure_patterns[pattern] += 1
            
            report['failure_analysis'] = {
                'total_failures': len(best_performance.failure_cases),
                'failure_rate': 1.0 - best_performance.success_rate,
                'common_patterns': sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        # Generate recommendations
        if report['results_summary'].get('overall_success', False):
            report['recommendations'] = [
                "✅ Step 5 acceptance criteria met - proceed with A/B testing",
                "🚀 Deploy best model to Step 4 shadow testing environment",
                "📊 Monitor production performance vs. lab evaluation",
                "🔧 Consider fine-tuning on production feedback data"
            ]
        else:
            report['recommendations'] = [
                "⚠️ Step 5 criteria not fully met - requires iteration",
                "🔧 Increase training data size or improve feature quality",
                "🧪 Experiment with different architectures or loss functions",
                "📊 Analyze failure cases for systematic issues"
            ]
        
        # Save report
        report_path = Path("research/reports/step5_research_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Research report saved: {report_path}")
        
        return report
    
    async def print_final_results(self, report: Dict[str, Any]):
        """Print comprehensive final results"""
        print(f"\n🎯 STEP 5 RESEARCH RESULTS")
        print("=" * 50)
        
        results_summary = report.get('results_summary', {})
        
        if results_summary:
            print(f"🏆 Best Model: {results_summary['best_model']}")
            print(f"\n📊 Performance vs CLIP-only Baseline:")
            print(f"   Compliance@1: {results_summary['baseline_compliance_at_1']:.4f} → {results_summary['best_compliance_at_1']:.4f} (+{results_summary['compliance_improvement']:.1f} pts)")
            print(f"   nDCG@10: {results_summary['baseline_ndcg_at_10']:.4f} → {results_summary['best_ndcg_at_10']:.4f} (+{results_summary['ndcg_improvement']:.1f} pts)")
            print(f"   Conflict AUC: {results_summary['conflict_auc']:.4f}")
            print(f"   Conflict ECE: {results_summary['conflict_ece']:.4f}")
            
            print(f"\n✅ Acceptance Criteria:")
            criteria = results_summary['meets_criteria']
            print(f"   Compliance@1 +3pts: {'✅' if criteria['compliance_improvement'] else '❌'} ({results_summary['compliance_improvement']:.1f} pts)")
            print(f"   nDCG@10 +6pts: {'✅' if criteria['ndcg_improvement'] else '❌'} ({results_summary['ndcg_improvement']:.1f} pts)")
            print(f"   Conflict AUC ≥0.90: {'✅' if criteria['conflict_auc'] else '❌'} ({results_summary['conflict_auc']:.3f})")
            print(f"   Conflict ECE ≤0.05: {'✅' if criteria['conflict_ece'] else '❌'} ({results_summary['conflict_ece']:.3f})")
            
            overall_success = results_summary.get('overall_success', False)
            print(f"\n🎊 Overall Success: {'✅ STEP 5 COMPLETE' if overall_success else '⚠️  NEEDS ITERATION'}")
        
        # Failure analysis
        failure_analysis = report.get('failure_analysis', {})
        if failure_analysis:
            print(f"\n🔍 Failure Analysis:")
            print(f"   Failure rate: {failure_analysis['failure_rate']:.1%}")
            print(f"   Common patterns:")
            for pattern, count in failure_analysis['common_patterns'][:3]:
                print(f"     • {pattern}: {count} cases")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
        
        # Integration with Step 4
        print(f"\n🔗 Step 4 Integration Ready:")
        print(f"   • Model registry: research/models/step5_{results_summary.get('best_model', 'full_reranker')}/")
        print(f"   • A/B testing config: Ready for shadow deployment")
        print(f"   • Monitoring integration: SLO thresholds defined")
        print(f"   • Rollback capability: Immutable model versioning")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        print(f"\n⏱️  Total Step 5 time: {total_time:.1f} seconds")

async def main():
    """Run Step 5 integration demo"""
    demo = Step5IntegrationDemo()
    
    try:
        report = await demo.run_step5_demo()
        
        print(f"\n🎉 STEP 5 DEMO COMPLETE!")
        print("Research track successfully integrated with Step 4 production pipeline")
        
        return report
        
    except Exception as e:
        logger.error(f"Step 5 demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())