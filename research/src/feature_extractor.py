#!/usr/bin/env python3
"""
Feature Extractor for CoTRR-lite Reranker

Extracts multi-modal features from Step 4 pipeline outputs:
- CLIP image/text embeddings (512-dim each)
- Visual attributes: region stats, color ΔE, subject_ratio  
- Conflict scores and probabilities
- Domain normalization and feature scaling

Integrates with Step 4 batched scoring pipeline.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    clip_dim: int = 512
    visual_features: List[str] = None
    conflict_features: List[str] = None
    normalize_per_domain: bool = True
    feature_scaling: str = "standard"  # standard, minmax, none
    
    def __post_init__(self):
        if self.visual_features is None:
            self.visual_features = [
                'subject_ratio', 'glass_ratio', 'garnish_ratio', 'ice_ratio',
                'color_delta_e', 'brightness', 'contrast', 'saturation'
            ]
        if self.conflict_features is None:
            self.conflict_features = [
                'conflict_score', 'conflict_prob', 'conflict_calibrated',
                'strong_conflict_count', 'soft_conflict_count'
            ]

class FeatureExtractor:
    """Extract features from Step 4 pipeline outputs"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scalers = {}
        self.domain_stats = {}
        
    def load_scored_jsonl(self, filepath: str) -> List[Dict]:
        """Load scored JSONL from Step 4 pipeline"""
        items = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(items)} scored items from {filepath}")
        return items
    
    def extract_clip_features(self, item: Dict) -> np.ndarray:
        """Extract CLIP image and text embeddings"""
        # Image embedding (from Step 4 batched scoring)
        img_embedding = item.get('image_embedding', [0.0] * self.config.clip_dim)
        
        # Text embedding (from query)
        text_embedding = item.get('text_embedding', [0.0] * self.config.clip_dim)
        
        # Concatenate image + text embeddings
        clip_features = np.concatenate([
            np.array(img_embedding, dtype=np.float32),
            np.array(text_embedding, dtype=np.float32)
        ])
        
        return clip_features
    
    def extract_visual_features(self, item: Dict) -> np.ndarray:
        """Extract visual attribute features"""
        features = []
        
        for feature_name in self.config.visual_features:
            if feature_name == 'subject_ratio':
                # From subject_object.py analysis
                features.append(item.get('subject_ratio', 0.5))
            elif feature_name == 'glass_ratio':
                # Glass detection ratio
                features.append(item.get('glass_ratio', 0.0))
            elif feature_name == 'garnish_ratio':
                # Garnish/decoration ratio
                features.append(item.get('garnish_ratio', 0.0))
            elif feature_name == 'ice_ratio':
                # Ice detection ratio  
                features.append(item.get('ice_ratio', 0.0))
            elif feature_name == 'color_delta_e':
                # Color difference from domain prototype
                features.append(item.get('color_delta_e', 50.0))
            elif feature_name == 'brightness':
                # Image brightness
                features.append(item.get('brightness', 0.5))
            elif feature_name == 'contrast':
                # Image contrast
                features.append(item.get('contrast', 0.5))
            elif feature_name == 'saturation':
                # Color saturation
                features.append(item.get('saturation', 0.5))
            else:
                # Default to 0 for unknown features
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_conflict_features(self, item: Dict) -> np.ndarray:
        """Extract conflict-related features"""
        features = []
        
        for feature_name in self.config.conflict_features:
            if feature_name == 'conflict_score':
                # Raw conflict score from conflict_penalty.py
                features.append(item.get('conflict_score', 0.0))
            elif feature_name == 'conflict_prob':
                # Calibrated conflict probability
                features.append(item.get('conflict_prob', 0.0))
            elif feature_name == 'conflict_calibrated':
                # Temperature-scaled conflict probability
                features.append(item.get('conflict_calibrated', 0.0))
            elif feature_name == 'strong_conflict_count':
                # Number of strong conflicts detected
                conflicts = item.get('conflicts', [])
                strong_count = sum(1 for c in conflicts if c.get('severity', 'soft') == 'strong')
                features.append(float(strong_count))
            elif feature_name == 'soft_conflict_count':
                # Number of soft conflicts detected
                conflicts = item.get('conflicts', [])
                soft_count = sum(1 for c in conflicts if c.get('severity', 'soft') == 'soft')
                features.append(float(soft_count))
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_all_features(self, item: Dict) -> np.ndarray:
        """Extract all features for an item"""
        # CLIP features (1024-dim: 512 image + 512 text)
        clip_features = self.extract_clip_features(item)
        
        # Visual features
        visual_features = self.extract_visual_features(item)
        
        # Conflict features
        conflict_features = self.extract_conflict_features(item)
        
        # Concatenate all features
        all_features = np.concatenate([
            clip_features,
            visual_features, 
            conflict_features
        ])
        
        return all_features
    
    def compute_dual_score_label(self, item: Dict, lambda_param: float = 0.7) -> float:
        """Compute dual score label: λ·Compliance + (1−λ)·(1−p_conflict)"""
        compliance_score = item.get('compliance_score', 0.5)
        conflict_prob = item.get('conflict_prob', 0.5)
        
        # Dual score combines compliance and anti-conflict
        dual_score = lambda_param * compliance_score + (1 - lambda_param) * (1 - conflict_prob)
        
        return np.clip(dual_score, 0.0, 1.0)
    
    def normalize_features_per_domain(self, items: List[Dict], features: np.ndarray) -> np.ndarray:
        """Normalize features per domain to handle domain shift"""
        if not self.config.normalize_per_domain:
            return features
        
        normalized_features = features.copy()
        domains = [item.get('domain', 'unknown') for item in items]
        unique_domains = list(set(domains))
        
        for domain in unique_domains:
            domain_mask = np.array([d == domain for d in domains])
            domain_indices = np.where(domain_mask)[0]
            
            if len(domain_indices) > 1:  # Need at least 2 samples
                domain_features = features[domain_indices]
                
                # Standardize within domain
                domain_mean = np.mean(domain_features, axis=0)
                domain_std = np.std(domain_features, axis=0) + 1e-8  # Avoid division by zero
                
                normalized_features[domain_indices] = (domain_features - domain_mean) / domain_std
                
                # Store stats for inference
                self.domain_stats[domain] = {
                    'mean': domain_mean,
                    'std': domain_std
                }
        
        return normalized_features
    
    def scale_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Apply feature scaling"""
        if self.config.feature_scaling == "none":
            return features
        
        scaler_key = "main_scaler"
        
        if fit or scaler_key not in self.scalers:
            if self.config.feature_scaling == "standard":
                scaler = StandardScaler()
            elif self.config.feature_scaling == "minmax":
                scaler = MinMaxScaler()
            else:
                return features
            
            scaled_features = scaler.fit_transform(features)
            self.scalers[scaler_key] = scaler
        else:
            scaled_features = self.scalers[scaler_key].transform(features)
        
        return scaled_features
    
    def create_training_pairs(self, items: List[Dict], features: np.ndarray, 
                            labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create pairwise training data for RankNet"""
        # Group items by query
        query_groups = {}
        for i, item in enumerate(items):
            query = item.get('query', 'unknown')
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((i, labels[i]))
        
        # Create pairs within each query group
        pair_features = []
        pair_labels = []
        pair_indices = []
        
        for query, group_items in query_groups.items():
            if len(group_items) < 2:
                continue  # Need at least 2 items to create pairs
            
            # Sort by label (descending)
            group_items.sort(key=lambda x: x[1], reverse=True)
            
            # Create pairs: better item vs worse item
            for i in range(len(group_items)):
                for j in range(i + 1, len(group_items)):
                    idx_better, label_better = group_items[i]
                    idx_worse, label_worse = group_items[j]
                    
                    if label_better > label_worse:
                        # Feature difference: better - worse
                        feature_diff = features[idx_better] - features[idx_worse]
                        pair_features.append(feature_diff)
                        pair_labels.append(1.0)  # Better item should rank higher
                        pair_indices.append((idx_better, idx_worse))
        
        return (
            np.array(pair_features, dtype=np.float32),
            np.array(pair_labels, dtype=np.float32),
            pair_indices
        )
    
    def process_pipeline_output(self, filepath: str, lambda_param: float = 0.7) -> Dict[str, Any]:
        """Process Step 4 pipeline output into training data"""
        # Load scored items
        items = self.load_scored_jsonl(filepath)
        
        if not items:
            raise ValueError(f"No items loaded from {filepath}")
        
        # Extract features
        logger.info("Extracting features...")
        features_list = []
        labels_list = []
        
        for item in items:
            features = self.extract_all_features(item)
            label = self.compute_dual_score_label(item, lambda_param)
            
            features_list.append(features)
            labels_list.append(label)
        
        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.float32)
        
        # Domain normalization
        if self.config.normalize_per_domain:
            features = self.normalize_features_per_domain(items, features)
        
        # Feature scaling
        features = self.scale_features(features, fit=True)
        
        # Create pairwise training data
        logger.info("Creating pairwise training data...")
        pair_features, pair_labels, pair_indices = self.create_training_pairs(items, features, labels)
        
        # Split by canonical_id to avoid leakage
        canonical_ids = [self._get_canonical_id(item) for item in items]
        train_indices, val_indices, test_indices = self._split_by_canonical_id(canonical_ids)
        
        logger.info(f"Feature extraction complete:")
        logger.info(f"  Total items: {len(items)}")
        logger.info(f"  Feature dimension: {features.shape[1]}")
        logger.info(f"  Training pairs: {len(pair_features)}")
        logger.info(f"  Train/Val/Test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
        
        return {
            'items': items,
            'features': features,
            'labels': labels,
            'pair_features': pair_features,
            'pair_labels': pair_labels,
            'pair_indices': pair_indices,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'feature_dim': features.shape[1],
            'config': self.config
        }
    
    def _get_canonical_id(self, item: Dict) -> str:
        """Get canonical ID for train/test splitting"""
        # Use URL as canonical ID to avoid leakage
        url = item.get('url', item.get('id', 'unknown'))
        return hashlib.md5(url.encode()).hexdigest()[:8]
    
    def _split_by_canonical_id(self, canonical_ids: List[str], 
                              train_ratio: float = 0.7, 
                              val_ratio: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
        """Split indices by canonical ID to avoid leakage"""
        unique_ids = list(set(canonical_ids))
        unique_ids.sort()  # Deterministic ordering
        
        n_ids = len(unique_ids)
        n_train = int(n_ids * train_ratio)
        n_val = int(n_ids * val_ratio)
        
        train_ids = set(unique_ids[:n_train])
        val_ids = set(unique_ids[n_train:n_train + n_val])
        test_ids = set(unique_ids[n_train + n_val:])
        
        train_indices = [i for i, cid in enumerate(canonical_ids) if cid in train_ids]
        val_indices = [i for i, cid in enumerate(canonical_ids) if cid in val_ids]
        test_indices = [i for i, cid in enumerate(canonical_ids) if cid in test_ids]
        
        return train_indices, val_indices, test_indices

def demo_feature_extraction():
    """Demo feature extraction on mock data"""
    print("🔬 Feature Extraction Demo")
    print("=" * 30)
    
    # Create mock scored data (simulating Step 4 pipeline output)
    mock_items = []
    for i in range(100):
        item = {
            'id': f'item_{i:03d}',
            'url': f'https://example.com/img_{i}.jpg',
            'query': f'cocktail query {i % 5}',
            'domain': ['blue_tropical', 'red_berry', 'green_citrus'][i % 3],
            
            # CLIP embeddings (mock)
            'image_embedding': np.random.normal(0, 1, 512).tolist(),
            'text_embedding': np.random.normal(0, 1, 512).tolist(),
            
            # Visual features
            'subject_ratio': np.random.uniform(0.2, 0.8),
            'glass_ratio': np.random.uniform(0.0, 0.3),
            'garnish_ratio': np.random.uniform(0.0, 0.2),
            'ice_ratio': np.random.uniform(0.0, 0.4),
            'color_delta_e': np.random.uniform(10, 100),
            'brightness': np.random.uniform(0.2, 0.8),
            'contrast': np.random.uniform(0.3, 0.7),
            'saturation': np.random.uniform(0.2, 0.8),
            
            # Conflict features
            'conflict_score': np.random.uniform(0.0, 1.0),
            'conflict_prob': np.random.uniform(0.0, 0.5),
            'conflict_calibrated': np.random.uniform(0.0, 0.5),
            'conflicts': [
                {'type': 'color_mismatch', 'severity': 'soft'},
                {'type': 'attribute_conflict', 'severity': 'strong'}
            ] if i % 4 == 0 else [],
            
            # Ground truth scores
            'compliance_score': np.random.uniform(0.5, 1.0),
        }
        mock_items.append(item)
    
    # Save mock data
    mock_file = "research/data/mock_scored.jsonl"
    Path("research/data").mkdir(parents=True, exist_ok=True)
    
    with open(mock_file, 'w') as f:
        for item in mock_items:
            f.write(json.dumps(item) + '\n')
    
    print(f"📝 Created mock scored data: {mock_file}")
    
    # Initialize feature extractor
    config = FeatureConfig(
        normalize_per_domain=True,
        feature_scaling="standard"
    )
    extractor = FeatureExtractor(config)
    
    # Process data
    result = extractor.process_pipeline_output(mock_file, lambda_param=0.7)
    
    print(f"\n📊 Feature Extraction Results:")
    print(f"   Items: {len(result['items'])}")
    print(f"   Feature dimension: {result['feature_dim']}")
    print(f"   Training pairs: {len(result['pair_features'])}")
    print(f"   Train/Val/Test: {len(result['train_indices'])}/{len(result['val_indices'])}/{len(result['test_indices'])}")
    
    # Feature breakdown
    clip_dim = 1024  # 512 image + 512 text
    visual_dim = len(config.visual_features)
    conflict_dim = len(config.conflict_features) 
    
    print(f"\n🧠 Feature Breakdown:")
    print(f"   CLIP embeddings: {clip_dim} dims")
    print(f"   Visual features: {visual_dim} dims")
    print(f"   Conflict features: {conflict_dim} dims")
    print(f"   Total: {clip_dim + visual_dim + conflict_dim} dims")
    
    # Show sample feature statistics
    features = result['features']
    print(f"\n📈 Feature Statistics:")
    print(f"   Mean: {np.mean(features):.3f}")
    print(f"   Std: {np.std(features):.3f}")
    print(f"   Min: {np.min(features):.3f}")
    print(f"   Max: {np.max(features):.3f}")
    
    return result

if __name__ == "__main__":
    demo_feature_extraction()