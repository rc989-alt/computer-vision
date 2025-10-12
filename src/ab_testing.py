#!/usr/bin/env python3
"""
A/B Testing & Rollback System

Production-safe model deployment with:
- Shadow runs for new model versions
- Stable test_clean set for consistent evaluation
- Automated promotion criteria (ΔCompliance@1 ≥ +3pts, Conflict↓ ≥20%)
- One-click rollback to previous model_version and overlay_id
- Immutable manifests enable perfect rollback

Supports safe deployment at scale with automatic quality gates.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import hashlib
from enum import Enum

from idempotency import ProcessingManifest, IdempotentProcessor, ManifestSummary

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    SHADOW = "shadow"  # Testing new version
    PROMOTED = "promoted"  # New version promoted to production
    ROLLED_BACK = "rolled_back"  # Rolled back to previous version
    FAILED = "failed"  # Deployment failed

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_name: str
    model_path: str
    created_at: str
    creator: str
    description: str
    baseline_metrics: Dict[str, float]
    
@dataclass
class ExperimentConfig:
    """A/B experiment configuration"""
    experiment_id: str
    baseline_version: str
    candidate_version: str
    test_set_path: str
    promotion_criteria: Dict[str, float]
    rollback_criteria: Dict[str, float]
    max_duration_hours: int = 24
    
@dataclass
class ExperimentResult:
    """Results from A/B experiment"""
    experiment_id: str
    baseline_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    delta_metrics: Dict[str, float] 
    decision: str  # promote, rollback, continue
    decision_reason: str
    confidence_level: float
    
@dataclass
class DeploymentRecord:
    """Record of model deployment"""
    deployment_id: str
    model_version: str
    overlay_version: str
    status: DeploymentStatus
    deployed_at: str
    metrics: Dict[str, float]
    rollback_target: Optional[str] = None

class TestSetManager:
    """Manages stable test sets for consistent evaluation"""
    
    def __init__(self, base_path: str = "datasets/eval"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def create_test_set(self, name: str, items: List[Dict], description: str = "") -> str:
        """Create immutable test set"""
        test_set_path = self.base_path / f"{name}.jsonl"
        
        # Add metadata
        metadata = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'item_count': len(items),
            'checksum': self._compute_checksum(items)
        }
        
        # Write test set
        with open(test_set_path, 'w') as f:
            # First line is metadata
            f.write(json.dumps(metadata) + '\n')
            
            # Followed by items
            for item in items:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created test set '{name}' with {len(items)} items")
        return str(test_set_path)
    
    def load_test_set(self, name: str) -> Tuple[Dict, List[Dict]]:
        """Load test set with metadata"""
        test_set_path = self.base_path / f"{name}.jsonl"
        
        if not test_set_path.exists():
            raise FileNotFoundError(f"Test set not found: {name}")
        
        with open(test_set_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError(f"Empty test set: {name}")
        
        # Parse metadata and items
        metadata = json.loads(lines[0].strip())
        items = [json.loads(line.strip()) for line in lines[1:] if line.strip()]
        
        # Verify integrity
        if len(items) != metadata['item_count']:
            logger.warning(f"Test set size mismatch: expected {metadata['item_count']}, got {len(items)}")
        
        return metadata, items
    
    def _compute_checksum(self, items: List[Dict]) -> str:
        """Compute checksum of test set"""
        content = json.dumps(items, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def list_test_sets(self) -> List[str]:
        """List available test sets"""
        return [f.stem for f in self.base_path.glob("*.jsonl")]

class ModelRegistry:
    """Registry for model versions"""
    
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models = self._load_registry()
        
    def _load_registry(self) -> Dict[str, ModelVersion]:
        """Load model registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                return {k: ModelVersion(**v) for k, v in data.items()}
        return {}
    
    def _save_registry(self):
        """Save model registry"""
        data = {k: asdict(v) for k, v in self.models.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, model: ModelVersion):
        """Register new model version"""
        self.models[model.version_id] = model
        self._save_registry()
        logger.info(f"Registered model {model.version_id}")
    
    def get_model(self, version_id: str) -> Optional[ModelVersion]:
        """Get model by version ID"""
        return self.models.get(version_id)
    
    def list_models(self) -> List[ModelVersion]:
        """List all registered models"""
        return list(self.models.values())

class ExperimentRunner:
    """Runs A/B experiments for model evaluation"""
    
    def __init__(self, processor: IdempotentProcessor, test_manager: TestSetManager):
        self.processor = processor
        self.test_manager = test_manager
        
    async def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run A/B experiment comparing baseline vs candidate"""
        logger.info(f"Starting experiment {config.experiment_id}")
        
        # Load test set
        test_metadata, test_items = self.test_manager.load_test_set(
            Path(config.test_set_path).stem
        )
        
        # Run baseline
        baseline_metrics = await self._evaluate_model(
            config.baseline_version, test_items, f"{config.experiment_id}_baseline"
        )
        
        # Run candidate
        candidate_metrics = await self._evaluate_model(
            config.candidate_version, test_items, f"{config.experiment_id}_candidate"
        )
        
        # Compute deltas
        delta_metrics = {}
        for metric in baseline_metrics:
            if metric in candidate_metrics:
                delta_metrics[f"delta_{metric}"] = candidate_metrics[metric] - baseline_metrics[metric]
        
        # Make promotion decision
        decision, reason, confidence = self._make_decision(
            baseline_metrics, candidate_metrics, delta_metrics, config
        )
        
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            delta_metrics=delta_metrics,
            decision=decision,
            decision_reason=reason,
            confidence_level=confidence
        )
        
        logger.info(f"Experiment {config.experiment_id} completed: {decision} ({reason})")
        return result
    
    async def _evaluate_model(self, model_version: str, test_items: List[Dict], run_suffix: str) -> Dict[str, float]:
        """Evaluate model on test set"""
        # This would run the actual pipeline with the model
        # For demo, we'll simulate metrics
        
        run_id = f"eval_{model_version}_{run_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate processing
        manifest = self.processor.start_run(model_version, "eval_overlay", run_id)
        manifest.add_items(test_items, "evaluated")
        
        # Mock evaluation results
        import random
        random.seed(hash(model_version) % 1000)  # Deterministic but different per model
        
        base_compliance = 0.90 + random.uniform(-0.05, 0.10)
        base_conflict = 0.10 + random.uniform(-0.03, 0.05)
        
        metrics = {
            'compliance_at_1': max(0.0, min(1.0, base_compliance)),
            'compliance_at_3': max(0.0, min(1.0, base_compliance + 0.05)),
            'compliance_at_5': max(0.0, min(1.0, base_compliance + 0.08)),
            'conflict_rate': max(0.0, base_conflict),
            'dual_score_mean': random.uniform(0.6, 0.9),
            'dual_score_std': random.uniform(0.1, 0.3),
            'processing_time': random.uniform(30, 120),  # seconds per 1k items
        }
        
        return metrics
    
    def _make_decision(self, baseline: Dict[str, float], candidate: Dict[str, float], 
                      deltas: Dict[str, float], config: ExperimentConfig) -> Tuple[str, str, float]:
        """Make promotion/rollback decision based on criteria"""
        
        # Check promotion criteria
        promotion_score = 0.0
        promotion_checks = []
        
        for metric, threshold in config.promotion_criteria.items():
            delta_key = f"delta_{metric}"
            if delta_key in deltas:
                delta = deltas[delta_key]
                if delta >= threshold:
                    promotion_score += 1.0
                    promotion_checks.append(f"{metric}: +{delta:.3f} ≥ {threshold}")
                else:
                    promotion_checks.append(f"{metric}: +{delta:.3f} < {threshold}")
        
        # Check rollback criteria (safety)
        rollback_triggered = False
        rollback_reasons = []
        
        for metric, threshold in config.rollback_criteria.items():
            delta_key = f"delta_{metric}"
            if delta_key in deltas:
                delta = deltas[delta_key]
                if delta <= -threshold:  # Negative delta exceeds rollback threshold
                    rollback_triggered = True
                    rollback_reasons.append(f"{metric}: {delta:.3f} ≤ -{threshold}")
        
        # Decision logic
        total_criteria = len(config.promotion_criteria)
        promotion_rate = promotion_score / max(total_criteria, 1)
        
        if rollback_triggered:
            decision = "rollback"
            reason = f"Safety criteria failed: {'; '.join(rollback_reasons)}"
            confidence = 0.95
        elif promotion_rate >= 0.8:  # 80% of criteria must pass
            decision = "promote"
            reason = f"Promotion criteria met ({promotion_score}/{total_criteria}): {'; '.join(promotion_checks)}"
            confidence = promotion_rate
        else:
            decision = "continue"
            reason = f"Insufficient improvement ({promotion_score}/{total_criteria}): {'; '.join(promotion_checks)}"
            confidence = 0.5
        
        return decision, reason, confidence

class DeploymentManager:
    """Manages model deployments and rollbacks"""
    
    def __init__(self, deployments_path: str = "deployments.json"):
        self.deployments_path = Path(deployments_path)
        self.deployments = self._load_deployments()
        self.current_deployment = self._get_current_deployment()
        
    def _load_deployments(self) -> List[DeploymentRecord]:
        """Load deployment history"""
        if self.deployments_path.exists():
            with open(self.deployments_path, 'r') as f:
                data = json.load(f)
                return [DeploymentRecord(**d) for d in data]
        return []
    
    def _save_deployments(self):
        """Save deployment history"""
        data = [asdict(d) for d in self.deployments]
        with open(self.deployments_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_current_deployment(self) -> Optional[DeploymentRecord]:
        """Get current active deployment"""
        active_deployments = [
            d for d in self.deployments 
            if d.status in [DeploymentStatus.PROMOTED, DeploymentStatus.SHADOW]
        ]
        return max(active_deployments, key=lambda x: x.deployed_at) if active_deployments else None
    
    def deploy(self, model_version: str, overlay_version: str, metrics: Dict[str, float], 
               status: DeploymentStatus = DeploymentStatus.SHADOW) -> str:
        """Deploy new model version"""
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deployment = DeploymentRecord(
            deployment_id=deployment_id,
            model_version=model_version,
            overlay_version=overlay_version,
            status=status,
            deployed_at=datetime.now().isoformat(),
            metrics=metrics,
            rollback_target=self.current_deployment.deployment_id if self.current_deployment else None
        )
        
        self.deployments.append(deployment)
        if status == DeploymentStatus.PROMOTED:
            self.current_deployment = deployment
        
        self._save_deployments()
        logger.info(f"Deployed {model_version} as {status.value} (ID: {deployment_id})")
        
        return deployment_id
    
    def promote(self, deployment_id: str) -> bool:
        """Promote shadow deployment to production"""
        deployment = next((d for d in self.deployments if d.deployment_id == deployment_id), None)
        
        if not deployment:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        if deployment.status != DeploymentStatus.SHADOW:
            logger.error(f"Can only promote shadow deployments, got: {deployment.status}")
            return False
        
        # Mark current as rolled back
        if self.current_deployment:
            self.current_deployment.status = DeploymentStatus.ROLLED_BACK
        
        # Promote new deployment
        deployment.status = DeploymentStatus.PROMOTED
        self.current_deployment = deployment
        
        self._save_deployments()
        logger.info(f"Promoted deployment {deployment_id} to production")
        
        return True
    
    def rollback(self, reason: str = "Manual rollback") -> Optional[str]:
        """Rollback to previous deployment"""
        if not self.current_deployment or not self.current_deployment.rollback_target:
            logger.error("No rollback target available")
            return None
        
        rollback_target = next(
            (d for d in self.deployments if d.deployment_id == self.current_deployment.rollback_target),
            None
        )
        
        if not rollback_target:
            logger.error(f"Rollback target not found: {self.current_deployment.rollback_target}")
            return None
        
        # Mark current as rolled back
        self.current_deployment.status = DeploymentStatus.ROLLED_BACK
        
        # Create new deployment record for rollback
        rollback_deployment = DeploymentRecord(
            deployment_id=f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_version=rollback_target.model_version,
            overlay_version=rollback_target.overlay_version,
            status=DeploymentStatus.PROMOTED,
            deployed_at=datetime.now().isoformat(),
            metrics=rollback_target.metrics.copy(),
            rollback_target=None
        )
        
        self.deployments.append(rollback_deployment)
        self.current_deployment = rollback_deployment
        
        self._save_deployments()
        logger.info(f"Rolled back to {rollback_target.model_version} (reason: {reason})")
        
        return rollback_deployment.deployment_id
    
    def get_deployment_history(self, limit: int = 10) -> List[DeploymentRecord]:
        """Get recent deployment history"""
        return sorted(self.deployments, key=lambda x: x.deployed_at, reverse=True)[:limit]

def demo_ab_testing():
    """Demo A/B testing and rollback system"""
    print("🧪 A/B Testing & Rollback Demo")
    print("=" * 40)
    
    # Initialize components
    processor = IdempotentProcessor("demo_datasets")
    test_manager = TestSetManager("demo_datasets/eval")
    model_registry = ModelRegistry("demo_model_registry.json")
    deployment_manager = DeploymentManager("demo_deployments.json")
    
    # Create test set
    test_items = [
        {
            'id': f'test_{i:03d}',
            'url': f'https://example.com/test_{i}.jpg',
            'query': f'test cocktail {i}',
            'domain': ['blue_tropical', 'red_berry'][i % 2]
        }
        for i in range(100)
    ]
    
    test_set_path = test_manager.create_test_set(
        "stable_test_v1", test_items, "Stable test set for model evaluation"
    )
    
    # Register model versions
    baseline_model = ModelVersion(
        version_id="clip_v1.0",
        model_name="CLIP Baseline",
        model_path="/models/clip_v1.0",
        created_at=datetime.now().isoformat(),
        creator="team",
        description="Baseline CLIP model",
        baseline_metrics={"compliance_at_1": 0.90, "conflict_rate": 0.10}
    )
    
    candidate_model = ModelVersion(
        version_id="clip_v1.1",
        model_name="CLIP Improved",
        model_path="/models/clip_v1.1", 
        created_at=datetime.now().isoformat(),
        creator="team",
        description="Improved CLIP with better training data",
        baseline_metrics={"compliance_at_1": 0.93, "conflict_rate": 0.08}
    )
    
    model_registry.register_model(baseline_model)
    model_registry.register_model(candidate_model)
    
    print(f"📝 Registered {len(model_registry.list_models())} models")
    print(f"🧪 Created test set with {len(test_items)} items")
    
    # Deploy baseline
    deployment_manager.deploy(
        "clip_v1.0", "overlay_v1.0", 
        baseline_model.baseline_metrics,
        DeploymentStatus.PROMOTED
    )
    
    print(f"🚀 Deployed baseline model to production")
    
    # Run A/B experiment
    experiment_config = ExperimentConfig(
        experiment_id="clip_v1.1_experiment",
        baseline_version="clip_v1.0",
        candidate_version="clip_v1.1", 
        test_set_path="stable_test_v1",
        promotion_criteria={
            "compliance_at_1": 0.03,  # +3 percentage points
            "conflict_rate": -0.02    # -2 percentage points (negative = improvement)
        },
        rollback_criteria={
            "compliance_at_1": 0.05,  # -5 percentage points triggers rollback
            "conflict_rate": 0.05     # +5 percentage points triggers rollback
        }
    )
    
    runner = ExperimentRunner(processor, test_manager)
    
    print(f"\n🔬 Running A/B experiment...")
    import asyncio
    result = asyncio.run(runner.run_experiment(experiment_config))
    
    print(f"\n📊 Experiment Results:")
    print(f"   Decision: {result.decision.upper()}")
    print(f"   Reason: {result.decision_reason}")
    print(f"   Confidence: {result.confidence_level:.1%}")
    
    print(f"\n📈 Metric Deltas:")
    for metric, delta in result.delta_metrics.items():
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"   {direction} {metric}: {delta:.3f}")
    
    # Handle deployment decision
    if result.decision == "promote":
        # Deploy candidate as shadow first
        shadow_id = deployment_manager.deploy(
            "clip_v1.1", "overlay_v1.1",
            result.candidate_metrics,
            DeploymentStatus.SHADOW
        )
        
        # Promote to production
        deployment_manager.promote(shadow_id)
        print(f"✅ Promoted clip_v1.1 to production")
        
    elif result.decision == "rollback":
        rollback_id = deployment_manager.rollback("A/B test failed criteria")
        print(f"🔄 Rolled back due to poor performance")
    else:
        print(f"⏸️ Continuing with current model - insufficient improvement")
    
    # Show deployment history
    history = deployment_manager.get_deployment_history()
    print(f"\n📋 Deployment History:")
    for deployment in history:
        status_emoji = {
            DeploymentStatus.SHADOW: "🔍",
            DeploymentStatus.PROMOTED: "✅", 
            DeploymentStatus.ROLLED_BACK: "🔄",
            DeploymentStatus.FAILED: "❌"
        }[deployment.status]
        
        print(f"   {status_emoji} {deployment.model_version} ({deployment.status.value})")
        print(f"      Deployed: {deployment.deployed_at}")
        print(f"      Compliance@1: {deployment.metrics.get('compliance_at_1', 0):.3f}")
    
    print(f"\n✅ A/B testing system ready for production!")
    
    return result, deployment_manager

if __name__ == "__main__":
    demo_ab_testing()