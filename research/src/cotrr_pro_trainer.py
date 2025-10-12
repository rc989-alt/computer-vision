#!/usr/bin/env python3
"""
CoTRR-Pro Training Pipeline
基于CVPR最佳实践的三阶段训练策略

Stage 1: Contrastive Pre-training (3-4 days)
Stage 2: Ranking Fine-tuning (2-3 days)  
Stage 3: Calibration Optimization (1 day)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 导入我们的模型
from cotrr_pro_transformer import (
    CoTRRProTransformer, ModelConfig, 
    ContrastiveLoss, ListMLELoss, create_model
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    batch_size: int = 32
    num_workers: int = 4
    
    # Stage 1: Contrastive Pre-training
    stage1_epochs: int = 20
    stage1_lr: float = 1e-4
    stage1_weight_decay: float = 1e-4
    contrastive_temperature: float = 0.07
    
    # Stage 2: Ranking Fine-tuning
    stage2_epochs: int = 15
    stage2_lr: float = 5e-5
    stage2_weight_decay: float = 1e-4
    dual_score_lambda: float = 0.7  # λ·Compliance + (1−λ)·(1−p_conflict)
    
    # Stage 3: Calibration
    stage3_epochs: int = 5
    stage3_lr: float = 1e-5
    mc_samples: int = 10
    
    # 通用
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "research/models"
    log_interval: int = 50
    eval_interval: int = 500
    early_stopping_patience: int = 3

class CoTRRDataset(Dataset):
    """CoTRR数据集"""
    
    def __init__(self, data_path: str, stage: str = "ranking"):
        """
        Args:
            data_path: JSONL文件路径
            stage: "contrastive" 或 "ranking"
        """
        self.stage = stage
        self.data = []
        
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} items for {stage} stage")
        
        if stage == "contrastive":
            self._prepare_contrastive_pairs()
        elif stage == "ranking":
            self._prepare_ranking_data()
    
    def _prepare_contrastive_pairs(self):
        """准备对比学习的positive/negative pairs"""
        # 按query和cocktail类型分组
        query_groups = {}
        cocktail_groups = {}
        
        for idx, item in enumerate(self.data):
            query = item.get('query', '')
            cocktail_type = item.get('cocktail_type', 'unknown')
            
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append(idx)
            
            if cocktail_type not in cocktail_groups:
                cocktail_groups[cocktail_type] = []
            cocktail_groups[cocktail_type].append(idx)
        
        self.query_groups = query_groups
        self.cocktail_groups = cocktail_groups
        
    def _prepare_ranking_data(self):
        """准备ranking数据"""
        # 按query分组，每组内按dual_score排序
        query_groups = {}
        for idx, item in enumerate(self.data):
            query = item.get('query', '')
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((idx, item.get('dual_score', 0.0)))
        
        # 每个query至少要有2个items才能做ranking
        self.ranking_groups = {
            query: sorted(items, key=lambda x: x[1], reverse=True) 
            for query, items in query_groups.items() 
            if len(items) >= 2
        }
        
        logger.info(f"Created {len(self.ranking_groups)} ranking groups")
    
    def __len__(self):
        if self.stage == "contrastive":
            return len(self.data)
        elif self.stage == "ranking":
            return len(self.ranking_groups)
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.stage == "contrastive":
            return self._get_contrastive_item(idx)
        elif self.stage == "ranking":
            return self._get_ranking_item(idx)
        else:
            return self._get_basic_item(idx)
    
    def _get_contrastive_item(self, idx):
        """获取对比学习样本"""
        anchor_item = self.data[idx]
        anchor_query = anchor_item.get('query', '')
        
        # Positive: 同一query的其他item
        positive_candidates = [i for i in self.query_groups.get(anchor_query, []) if i != idx]
        if positive_candidates:
            pos_idx = np.random.choice(positive_candidates)
            positive_item = self.data[pos_idx]
        else:
            positive_item = anchor_item  # fallback
        
        # Negative: 不同query的随机item
        all_other_queries = [q for q in self.query_groups.keys() if q != anchor_query]
        if all_other_queries:
            neg_query = np.random.choice(all_other_queries)
            neg_idx = np.random.choice(self.query_groups[neg_query])
            negative_item = self.data[neg_idx]
        else:
            # 如果没有其他query，用不同cocktail_type
            anchor_cocktail = anchor_item.get('cocktail_type', 'unknown')
            other_cocktails = [c for c in self.cocktail_groups.keys() if c != anchor_cocktail]
            if other_cocktails:
                neg_cocktail = np.random.choice(other_cocktails)
                neg_idx = np.random.choice(self.cocktail_groups[neg_cocktail])
                negative_item = self.data[neg_idx]
            else:
                negative_item = anchor_item  # fallback
        
        return {
            'anchor': self._extract_features(anchor_item),
            'positive': self._extract_features(positive_item),
            'negative': self._extract_features(negative_item),
            'query_id': hash(anchor_query) % 10000  # 用于contrastive loss
        }
    
    def _get_ranking_item(self, idx):
        """获取ranking样本"""
        query = list(self.ranking_groups.keys())[idx]
        items_with_scores = self.ranking_groups[query]
        
        # 取前10个items (如果有的话)
        top_items = items_with_scores[:10]
        
        features_list = []
        scores_list = []
        
        for item_idx, dual_score in top_items:
            item = self.data[item_idx]
            features_list.append(self._extract_features(item))
            scores_list.append(dual_score)
        
        # Padding到固定长度
        max_items = 10
        while len(features_list) < max_items:
            # 用最后一个item padding
            if features_list:
                features_list.append(features_list[-1])
                scores_list.append(0.0)  # padding score为0
        
        return {
            'features': features_list,
            'scores': scores_list,
            'query': query,
            'num_items': len(top_items)
        }
    
    def _get_basic_item(self, idx):
        """获取基础样本"""
        item = self.data[idx]
        return self._extract_features(item)
    
    def _extract_features(self, item: Dict) -> Dict[str, torch.Tensor]:
        """提取特征"""
        # 模拟特征提取（实际应该从item中读取真实特征）
        return {
            'clip_img': torch.randn(1024),  # 实际应该是item['clip_img_features']
            'clip_text': torch.randn(1024),  # 实际应该是item['clip_text_features']
            'visual': torch.randn(8),  # 实际应该是item['visual_features']
            'conflict': torch.randn(5),  # 实际应该是item['conflict_features']
            'compliance_score': item.get('compliance_score', 0.0),
            'conflict_prob': item.get('conflict_probability', 0.0),
            'dual_score': item.get('dual_score', 0.0)
        }

class CoTRRTrainer:
    """CoTRR-Pro训练器"""
    
    def __init__(self, config: TrainingConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        
        # 创建模型
        self.model = create_model(model_config).to(config.device)
        
        # 损失函数
        self.contrastive_loss = ContrastiveLoss(config.contrastive_temperature)
        self.ranking_loss = ListMLELoss()
        self.calibration_loss = nn.BCEWithLogitsLoss()
        
        # 创建保存目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'stage1_loss': [], 'stage2_loss': [], 'stage3_loss': [],
            'val_compliance': [], 'val_ndcg': [], 'val_auc': [], 'val_ece': []
        }
    
    def stage1_contrastive_pretraining(self, train_dataset: CoTRRDataset, 
                                       val_dataset: Optional[CoTRRDataset] = None):
        """Stage 1: 对比学习预训练"""
        logger.info("🚀 Stage 1: Contrastive Pre-training")
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers
        )
        
        # 优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.stage1_lr,
            weight_decay=self.config.stage1_weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.stage1_epochs
        )
        
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(self.config.stage1_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                # 提取anchor, positive, negative features
                anchor_features = self._extract_batch_features(batch, 'anchor')
                pos_features = self._extract_batch_features(batch, 'positive') 
                neg_features = self._extract_batch_features(batch, 'negative')
                
                # 前向传播获取表示
                anchor_out = self.model(**anchor_features)
                pos_out = self.model(**pos_features)
                neg_out = self.model(**neg_features)
                
                # 使用多模态特征做对比学习
                anchor_repr = anchor_out['multimodal_features'].mean(dim=1)  # [batch, hidden]
                pos_repr = pos_out['multimodal_features'].mean(dim=1)
                neg_repr = neg_out['multimodal_features'].mean(dim=1)
                
                # 对比学习损失
                # InfoNCE loss
                batch_size = anchor_repr.size(0)
                
                # 计算相似度
                pos_sim = F.cosine_similarity(anchor_repr, pos_repr)
                neg_sim = F.cosine_similarity(anchor_repr, neg_repr)
                
                # InfoNCE
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
                labels = torch.zeros(batch_size, dtype=torch.long, device=self.config.device)
                
                loss = F.cross_entropy(logits / self.config.contrastive_temperature, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if batch_idx % self.config.log_interval == 0:
                    logger.info(f"Stage1 Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            scheduler.step()
            avg_loss = epoch_loss / num_batches
            self.history['stage1_loss'].append(avg_loss)
            
            logger.info(f"Stage1 Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(f'stage1_best.pt', epoch, avg_loss)
        
        logger.info("✅ Stage 1 completed!")
    
    def stage2_ranking_finetuning(self, train_dataset: CoTRRDataset,
                                  val_dataset: Optional[CoTRRDataset] = None):
        """Stage 2: Ranking微调"""
        logger.info("🚀 Stage 2: Ranking Fine-tuning")
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size // 2,  # ranking需要更多内存
            shuffle=True, num_workers=self.config.num_workers
        )
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.stage2_lr,
            weight_decay=self.config.stage2_weight_decay
        )
        
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(self.config.stage2_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                # 处理ranking batch
                batch_loss = 0.0
                batch_count = 0
                
                for item in batch:
                    features_list = item['features']
                    scores_list = item['scores']
                    num_items = item['num_items']
                    
                    if num_items < 2:
                        continue
                    
                    # 只取有效的items
                    valid_features = features_list[:num_items]
                    valid_scores = scores_list[:num_items]
                    
                    # 批量前向传播
                    logits_list = []
                    for features in valid_features:
                        features_batch = {k: v.unsqueeze(0).to(self.config.device) 
                                        for k, v in features.items() 
                                        if k in ['clip_img', 'clip_text', 'visual', 'conflict']}
                        
                        output = self.model(**features_batch)
                        logits_list.append(output['logits'].squeeze())
                    
                    if len(logits_list) < 2:
                        continue
                    
                    # 转换为tensor
                    logits_tensor = torch.stack(logits_list).unsqueeze(0)  # [1, num_items]
                    scores_tensor = torch.tensor(valid_scores, device=self.config.device).unsqueeze(0)
                    
                    # ListMLE loss
                    loss = self.ranking_loss(logits_tensor, scores_tensor)
                    batch_loss += loss
                    batch_count += 1
                
                if batch_count > 0:
                    final_loss = batch_loss / batch_count
                    final_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += final_loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({'loss': f'{final_loss.item():.4f}'})
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.history['stage2_loss'].append(avg_loss)
                
                logger.info(f"Stage2 Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_checkpoint(f'stage2_best.pt', epoch, avg_loss)
        
        logger.info("✅ Stage 2 completed!")
    
    def stage3_calibration(self, val_dataset: CoTRRDataset):
        """Stage 3: 校准优化"""
        logger.info("🚀 Stage 3: Calibration Optimization")
        
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # 只优化temperature参数
        optimizer = optim.LBFGS([self.model.temperature], lr=self.config.stage3_lr)
        
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            num_batches = 0
            
            for batch in val_loader:
                features = self._extract_batch_features(batch, None)
                
                # 使用Monte Carlo Dropout
                outputs = self.model(**features, mc_samples=self.config.mc_samples)
                logits = outputs['logits']
                
                # 获取真实conflict labels
                conflict_labels = torch.tensor([
                    item.get('conflict_prob', 0.0) > 0.5 
                    for item in batch
                ], device=self.config.device, dtype=torch.float)
                
                # 校准损失（ECE-based）
                probs = torch.sigmoid(logits.squeeze())
                loss = F.binary_cross_entropy(probs, conflict_labels)
                
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else total_loss
            avg_loss.backward()
            return avg_loss
        
        for epoch in range(self.config.stage3_epochs):
            loss = optimizer.step(closure)
            logger.info(f"Stage3 Epoch {epoch+1}, Temperature: {self.model.temperature.item():.4f}, Loss: {loss.item():.4f}")
            self.history['stage3_loss'].append(loss.item())
        
        self._save_checkpoint(f'stage3_final.pt', self.config.stage3_epochs, loss.item())
        logger.info("✅ Stage 3 completed!")
    
    def _extract_batch_features(self, batch, key=None):
        """从batch中提取特征"""
        if key is not None:
            # 对比学习模式
            items = [item[key] for item in batch]
        else:
            # 普通模式
            items = batch
        
        features = {
            'clip_img': torch.stack([item['clip_img'] for item in items]).to(self.config.device),
            'clip_text': torch.stack([item['clip_text'] for item in items]).to(self.config.device),
            'visual_features': torch.stack([item['visual'] for item in items]).to(self.config.device),
            'conflict_features': torch.stack([item['conflict'] for item in items]).to(self.config.device)
        }
        
        return features
    
    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': asdict(self.config),
            'model_config': asdict(self.model_config),
            'history': self.history
        }
        
        torch.save(checkpoint, self.save_dir / filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def train_full_pipeline(self, 
                           contrastive_data_path: str,
                           ranking_data_path: str,
                           val_data_path: str):
        """完整的三阶段训练流程"""
        logger.info("🚀 Starting CoTRR-Pro Full Training Pipeline")
        
        # Stage 1: Contrastive Pre-training
        contrastive_dataset = CoTRRDataset(contrastive_data_path, stage="contrastive")
        self.stage1_contrastive_pretraining(contrastive_dataset)
        
        # Stage 2: Ranking Fine-tuning  
        ranking_dataset = CoTRRDataset(ranking_data_path, stage="ranking")
        self.stage2_ranking_finetuning(ranking_dataset)
        
        # Stage 3: Calibration
        val_dataset = CoTRRDataset(val_data_path, stage="basic")
        self.stage3_calibration(val_dataset)
        
        # 保存最终模型和训练历史
        self._save_checkpoint('final_model.pt', 0, 0.0)
        self._save_training_plots()
        
        logger.info("🎉 CoTRR-Pro Training Pipeline Completed!")
    
    def _save_training_plots(self):
        """保存训练曲线图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Stage losses
        if self.history['stage1_loss']:
            axes[0, 0].plot(self.history['stage1_loss'], label='Contrastive Loss')
            axes[0, 0].set_title('Stage 1: Contrastive Pre-training')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        if self.history['stage2_loss']:
            axes[0, 1].plot(self.history['stage2_loss'], label='Ranking Loss') 
            axes[0, 1].set_title('Stage 2: Ranking Fine-tuning')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
        
        if self.history['stage3_loss']:
            axes[1, 0].plot(self.history['stage3_loss'], label='Calibration Loss')
            axes[1, 0].set_title('Stage 3: Calibration')
            axes[1, 0].set_xlabel('Epoch') 
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
        
        # 温度参数变化
        axes[1, 1].axhline(y=self.model.temperature.item(), color='r', linestyle='--', 
                          label=f'Final Temperature: {self.model.temperature.item():.4f}')
        axes[1, 1].set_title('Temperature Parameter')
        axes[1, 1].set_xlabel('Training Progress')
        axes[1, 1].set_ylabel('Temperature')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved: {self.save_dir / 'training_curves.png'}")

def main():
    """主训练函数"""
    # 配置
    model_config = ModelConfig(
        hidden_dim=512,
        num_attention_heads=8,
        num_layers=3,
        dropout=0.1
    )
    
    training_config = TrainingConfig(
        batch_size=16,  # 降低batch size适应复杂模型
        stage1_epochs=10,  # 缩短训练轮数用于演示
        stage2_epochs=8,
        stage3_epochs=3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 创建训练器
    trainer = CoTRRTrainer(training_config, model_config)
    
    print(f"🚀 CoTRR-Pro Trainer 创建成功!")
    print(f"📱 Device: {training_config.device}")
    print(f"🏗️ Model params: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"💾 Save dir: {training_config.save_dir}")
    
    # 如果有真实数据，可以运行完整训练
    # trainer.train_full_pipeline(
    #     contrastive_data_path="data/contrastive_train.jsonl",
    #     ranking_data_path="data/ranking_train.jsonl", 
    #     val_data_path="data/val.jsonl"
    # )
    
    print("✅ CoTRR-Pro Training Pipeline 准备就绪!")
    print("📝 使用 trainer.train_full_pipeline() 开始训练")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()