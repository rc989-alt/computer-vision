#!/usr/bin/env python3
"""
CoTRR-lite Reranker Training

PyTorch implementation of pairwise RankNet for cocktail image reranking.
Optimizes dual objective: λ·Compliance + (1−λ)·(1−p_conflict)

Supports ablation studies:
- CLIP-only baseline
- +Subject-Object features
- +Conflict features  
- Full reranker

Integrates with Step 4 A/B testing for safe deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.metrics import roc_auc_score, ndcg_score
import time
from datetime import datetime

from feature_extractor import FeatureExtractor, FeatureConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """CoTRR-lite model configuration"""
    # Architecture
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    activation: str = "relu"  # relu, gelu, swish
    batch_norm: bool = True
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    
    # Loss function
    margin: float = 0.1  # RankNet margin
    loss_type: str = "ranknet"  # ranknet, listnet, lambdarank
    
    # Ablation settings
    use_clip: bool = True
    use_visual: bool = True
    use_conflict: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 1]

class PairwiseDataset(Dataset):
    """Dataset for pairwise training"""
    
    def __init__(self, pair_features: np.ndarray, pair_labels: np.ndarray):
        self.pair_features = torch.FloatTensor(pair_features)
        self.pair_labels = torch.FloatTensor(pair_labels)
    
    def __len__(self):
        return len(self.pair_features)
    
    def __getitem__(self, idx):
        return self.pair_features[idx], self.pair_labels[idx]

class CoTRRLiteReranker(nn.Module):
    """CoTRR-lite reranker network"""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if hidden_dim != config.hidden_dims[-1]:  # Don't add activation/dropout to output
                if config.batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "gelu":
                    layers.append(nn.GELU())
                elif config.activation == "swish":
                    layers.append(nn.SiLU())
                
                if config.dropout_rate > 0:
                    layers.append(nn.Dropout(config.dropout_rate))
            
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass - returns ranking score"""
        return self.network(x).squeeze(-1)
    
    def predict_batch(self, features: torch.Tensor) -> torch.Tensor:
        """Predict ranking scores for a batch"""
        self.eval()
        with torch.no_grad():
            scores = self.forward(features)
        return scores

class RankNetLoss(nn.Module):
    """RankNet pairwise ranking loss"""
    
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, pair_diff_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        pair_diff_scores: score_better - score_worse for each pair
        targets: 1.0 if first item should rank higher, 0.0 otherwise
        """
        # RankNet loss: -log(sigmoid(score_diff))
        loss = -torch.log(torch.sigmoid(pair_diff_scores + self.margin))
        return loss.mean()

class CoTRRTrainer:
    """Trainer for CoTRR-lite reranker"""
    
    def __init__(self, model: CoTRRLiteReranker, config: ModelConfig, device: str = "auto"):
        self.model = model
        self.config = config
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.loss_fn = RankNetLoss(margin=config.margin)
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            scores = self.model(batch_features)
            
            # RankNet expects pair differences
            loss = self.loss_fn(scores, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'train_loss': total_loss / num_batches}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                scores = self.model(batch_features)
                loss = self.loss_fn(scores, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / num_batches}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Full training loop with early stopping"""
        logger.info(f"Starting training on {self.device}")
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            self.training_history.append(epoch_metrics)
            
            # Early stopping check
            val_loss = val_metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                          f"val_loss={val_loss:.4f}, patience={self.patience_counter}")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f} seconds")
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1,
            'training_time': training_time,
            'history': self.training_history
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])

def create_ablation_configs() -> Dict[str, ModelConfig]:
    """Create configurations for ablation study"""
    base_config = ModelConfig()
    
    configs = {
        'clip_only': ModelConfig(
            use_clip=True, use_visual=False, use_conflict=False,
            hidden_dims=[256, 128, 1]
        ),
        'clip_visual': ModelConfig(
            use_clip=True, use_visual=True, use_conflict=False,
            hidden_dims=[384, 192, 1]
        ),
        'clip_conflict': ModelConfig(
            use_clip=True, use_visual=False, use_conflict=True,
            hidden_dims=[384, 192, 1]
        ),
        'full_reranker': ModelConfig(
            use_clip=True, use_visual=True, use_conflict=True,
            hidden_dims=[512, 256, 128, 1]
        )
    }
    
    return configs

def filter_features_for_ablation(features: np.ndarray, config: ModelConfig, 
                                 feature_config: FeatureConfig) -> np.ndarray:
    """Filter features based on ablation configuration"""
    feature_slices = []
    start_idx = 0
    
    # CLIP features (first 1024 dims)
    clip_dim = 1024  # 512 image + 512 text
    if config.use_clip:
        feature_slices.append(slice(start_idx, start_idx + clip_dim))
    start_idx += clip_dim
    
    # Visual features
    visual_dim = len(feature_config.visual_features)
    if config.use_visual:
        feature_slices.append(slice(start_idx, start_idx + visual_dim))
    start_idx += visual_dim
    
    # Conflict features
    conflict_dim = len(feature_config.conflict_features)
    if config.use_conflict:
        feature_slices.append(slice(start_idx, start_idx + conflict_dim))
    
    # Concatenate selected feature slices
    if feature_slices:
        filtered_features = np.concatenate([features[:, s] for s in feature_slices], axis=1)
    else:
        raise ValueError("At least one feature type must be enabled")
    
    return filtered_features

def train_reranker(feature_data: Dict[str, Any], model_config: ModelConfig, 
                  output_dir: str) -> Dict[str, Any]:
    """Train CoTRR-lite reranker"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter features based on ablation config
    features = filter_features_for_ablation(
        feature_data['features'], model_config, feature_data['config']
    )
    
    pair_features = filter_features_for_ablation(
        feature_data['pair_features'], model_config, feature_data['config']
    )
    
    # Create datasets
    train_dataset = PairwiseDataset(pair_features, feature_data['pair_labels'])
    
    # For validation, we'll use a subset of training data (in real scenario, use separate validation set)
    val_size = len(train_dataset) // 5
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=model_config.batch_size, shuffle=False
    )
    
    # Initialize model
    input_dim = features.shape[1]
    model = CoTRRLiteReranker(input_dim, model_config)
    
    # Initialize trainer
    trainer = CoTRRTrainer(model, model_config)
    
    # Train model
    logger.info(f"Training {model.__class__.__name__} with {input_dim} input features")
    training_results = trainer.train(train_loader, val_loader)
    
    # Save final model
    model_path = output_path / "final_model.pt"
    trainer.save_checkpoint(str(model_path))
    
    # Save configuration
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': asdict(model_config),
            'feature_config': asdict(feature_data['config']),
            'input_dim': input_dim,
            'training_results': training_results
        }, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    
    return {
        'model': model,
        'trainer': trainer,
        'training_results': training_results,
        'model_path': str(model_path),
        'input_dim': input_dim
    }

def demo_reranker_training():
    """Demo reranker training on mock data"""
    print("🏋️ CoTRR-lite Reranker Training Demo")
    print("=" * 40)
    
    # Generate mock data using feature extractor
    from feature_extractor import demo_feature_extraction
    feature_data = demo_feature_extraction()
    
    # Create ablation configurations
    ablation_configs = create_ablation_configs()
    
    results = {}
    
    for name, config in ablation_configs.items():
        print(f"\n🔬 Training {name} configuration...")
        
        output_dir = f"research/models/{name}"
        
        try:
            result = train_reranker(feature_data, config, output_dir)
            results[name] = result
            
            print(f"✅ {name} training completed")
            print(f"   Best val loss: {result['training_results']['best_val_loss']:.4f}")
            print(f"   Total epochs: {result['training_results']['total_epochs']}")
            print(f"   Training time: {result['training_results']['training_time']:.1f}s")
            
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    print(f"\n🎯 Ablation Study Results:")
    print("-" * 40)
    print(f"{'Configuration':<15} {'Val Loss':<10} {'Epochs':<8} {'Time (s)':<10}")
    print("-" * 40)
    
    for name, result in results.items():
        val_loss = result['training_results']['best_val_loss']
        epochs = result['training_results']['total_epochs']
        time_s = result['training_results']['training_time']
        print(f"{name:<15} {val_loss:<10.4f} {epochs:<8} {time_s:<10.1f}")
    
    print(f"\n✅ CoTRR-lite reranker training demo complete!")
    
    return results

if __name__ == "__main__":
    demo_reranker_training()