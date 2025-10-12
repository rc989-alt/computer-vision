#!/usr/bin/env python3
"""
CoTRR-Stable: 稳健的两阶段实现方案
阶段1: Cross-Attention + ListMLE + Calibration (2周目标)

核心设计原则:
1. 与现有Step4/5无缝集成
2. 稳健的性能提升: C@1 +4-6pts, nDCG@10 +8-12pts  
3. 可控的推理成本: 仅Top-M使用复杂模型
4. 严格的A/B测试准入门槛
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class StableConfig:
    """稳健配置"""
    # 模型架构 - 保守参数
    hidden_dim: int = 256  # 降低至256，平衡性能与成本
    num_attention_heads: int = 8
    num_layers: int = 2    # 只用2层，避免过拟合
    dropout: float = 0.15  # 稍高dropout增加鲁棒性
    
    # 特征维度
    clip_img_dim: int = 1024
    clip_text_dim: int = 1024  
    visual_dim: int = 8
    conflict_dim: int = 5
    
    # 训练策略
    warmup_epochs: int = 3      # Pairwise warmup
    listwise_epochs: int = 8    # ListMLE fine-tune
    total_epochs: int = 12      # 总共12轮，保守
    
    # Top-M策略 - 控制推理成本
    top_m_candidates: int = 20  # 只对Top-20做复杂推理
    
    # 校准参数
    mc_samples: int = 5         # 保守的MC采样数
    ensemble_size: int = 3      # 小型ensemble
    
    # 性能门槛
    min_compliance_gain: float = 4.0  # C@1最低+4pts
    min_ndcg_gain: float = 8.0        # nDCG@10最低+8pts
    max_latency_p95: float = 30.0     # p95延迟≤30min/10k

class TokenizedMultiModalEncoder(nn.Module):
    """轻量级Token化多模态编码器"""
    
    def __init__(self, config: StableConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # 特征投影层 - 统一到hidden_dim
        self.img_proj = nn.Linear(config.clip_img_dim, self.hidden_dim)
        self.text_proj = nn.Linear(config.clip_text_dim, self.hidden_dim)
        self.visual_proj = nn.Linear(config.visual_dim, self.hidden_dim)
        self.conflict_proj = nn.Linear(config.conflict_dim, self.hidden_dim)
        
        # 位置编码 - 简单的学习式
        self.position_embeddings = nn.Parameter(torch.randn(4, self.hidden_dim))
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, clip_img, clip_text, visual_features, conflict_features):
        """
        将多模态特征转换为token序列
        Returns: [batch, 4, hidden_dim] - [img, text, visual, conflict]
        """
        batch_size = clip_img.size(0)
        
        # 特征投影
        img_token = self.img_proj(clip_img)      # [batch, hidden]
        text_token = self.text_proj(clip_text)   # [batch, hidden]  
        visual_token = self.visual_proj(visual_features)  # [batch, hidden]
        conflict_token = self.conflict_proj(conflict_features)  # [batch, hidden]
        
        # 堆叠为序列 + 位置编码
        tokens = torch.stack([img_token, text_token, visual_token, conflict_token], dim=1)
        # [batch, 4, hidden]
        
        # 添加位置编码
        tokens = tokens + self.position_embeddings.unsqueeze(0)
        
        # Layer norm
        tokens = self.layer_norm(tokens)
        
        return tokens

class LightweightCrossAttention(nn.Module):
    """轻量级Cross-Attention模块"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 单一线性层 - 减少参数
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Self-attention within multimodal tokens
        x: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # QKV projection
        qkv = self.qkv(x)  # [batch, seq_len, hidden*3]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        # Output projection + residual
        output = self.out_proj(attn_output) + residual
        output = self.layer_norm(output)
        
        return output, attn_weights

class StableCrossAttnReranker(nn.Module):
    """稳健的Cross-Attention重排器"""
    
    def __init__(self, config: StableConfig):
        super().__init__()
        self.config = config
        
        # Token编码器
        self.token_encoder = TokenizedMultiModalEncoder(config)
        
        # Cross-attention层
        self.attention_layers = nn.ModuleList([
            LightweightCrossAttention(
                config.hidden_dim, 
                config.num_attention_heads, 
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # 输出头 - 简化版
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),  # 拼接4个token
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)  # 单分数输出
        )
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        
        # MC Dropout层
        self.mc_dropout = nn.Dropout(0.1)
        
    def forward(self, clip_img, clip_text, visual_features, conflict_features, 
                mc_samples=1, training=False):
        """
        前向传播
        Returns: Dict with 'logits', 'calibrated_logits', 'uncertainty'
        """
        if mc_samples > 1:
            # Monte Carlo inference
            return self._mc_forward(clip_img, clip_text, visual_features, 
                                  conflict_features, mc_samples)
        else:
            return self._single_forward(clip_img, clip_text, visual_features, 
                                      conflict_features, training)
    
    def _single_forward(self, clip_img, clip_text, visual_features, conflict_features, training=False):
        """单次前向传播"""
        # 1. Token化编码
        tokens = self.token_encoder(clip_img, clip_text, visual_features, conflict_features)
        # [batch, 4, hidden_dim]
        
        # 2. Cross-attention
        attention_weights = []
        for attn_layer in self.attention_layers:
            tokens, weights = attn_layer(tokens)
            attention_weights.append(weights)
            
            if training:
                tokens = self.mc_dropout(tokens)
        
        # 3. 聚合token特征
        # Global average pooling + 原始拼接
        pooled = tokens.mean(dim=1)  # [batch, hidden_dim]
        flattened = tokens.flatten(start_dim=1)  # [batch, hidden_dim*4]
        
        # 4. 输出预测
        logits = self.output_head(flattened)  # [batch, 1]
        
        # 5. 温度校准
        calibrated_logits = logits / (self.temperature + 1e-8)
        
        return {
            'logits': logits,
            'calibrated_logits': calibrated_logits,
            'temperature': self.temperature,
            'tokens': tokens,
            'attention_weights': attention_weights,
            'uncertainty': torch.zeros_like(logits)  # 单次推理无uncertainty
        }
    
    def _mc_forward(self, clip_img, clip_text, visual_features, conflict_features, mc_samples):
        """Monte Carlo前向传播"""
        self.train()  # 开启dropout
        
        mc_outputs = []
        for _ in range(mc_samples):
            output = self._single_forward(clip_img, clip_text, visual_features, 
                                        conflict_features, training=True)
            mc_outputs.append(output['calibrated_logits'])
        
        self.eval()  # 恢复eval模式
        
        # 聚合MC结果
        mc_logits = torch.stack(mc_outputs, dim=0)  # [mc_samples, batch, 1]
        mean_logits = mc_logits.mean(dim=0)
        std_logits = mc_logits.std(dim=0)
        
        return {
            'logits': mean_logits,
            'calibrated_logits': mean_logits,
            'uncertainty': std_logits,
            'mc_samples': mc_logits,
            'temperature': self.temperature
        }

class ListMLEWithFocalLoss(nn.Module):
    """ListMLE + Focal Loss组合"""
    
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # ListMLE权重
        self.gamma = gamma  # Focal Loss参数
        
    def forward(self, logits, labels, valid_mask=None):
        """
        Args:
            logits: [batch, max_list_size] - 预测分数
            labels: [batch, max_list_size] - 真实相关性分数
            valid_mask: [batch, max_list_size] - 有效位置mask
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
        
        # ListMLE loss
        listmle_loss = self._listmle_loss(logits, labels, valid_mask)
        
        # Focal loss (在二分类任务上)
        binary_labels = (labels > 0.5).float()
        focal_loss = self._focal_loss(logits, binary_labels, valid_mask)
        
        # 组合损失
        total_loss = self.alpha * listmle_loss + (1 - self.alpha) * focal_loss
        
        return total_loss
    
    def _listmle_loss(self, logits, labels, valid_mask):
        """ListMLE损失实现"""
        batch_size = logits.size(0)
        
        # 按真实标签排序
        sorted_labels, sort_indices = torch.sort(labels, dim=1, descending=True)
        sorted_logits = torch.gather(logits, 1, sort_indices)
        sorted_mask = torch.gather(valid_mask.float(), 1, sort_indices)
        
        # ListMLE计算
        losses = []
        for i in range(batch_size):
            valid_len = int(sorted_mask[i].sum().item())
            if valid_len <= 1:
                continue
                
            item_logits = sorted_logits[i, :valid_len]
            
            # 计算ListMLE
            max_logit = item_logits.max()
            exp_logits = torch.exp(item_logits - max_logit)
            
            cumsum_exp = torch.cumsum(exp_logits.flip(0), dim=0).flip(0)
            log_prob = torch.log(exp_logits / (cumsum_exp + 1e-8))
            
            losses.append(-log_prob.sum())
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)
    
    def _focal_loss(self, logits, binary_labels, valid_mask):
        """Focal Loss实现"""
        # Sigmoid + BCE with focal weighting
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy(probs, binary_labels, reduction='none')
        
        # Focal weighting
        pt = torch.where(binary_labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        # 应用mask并平均
        masked_loss = focal_loss * valid_mask.float()
        return masked_loss.sum() / (valid_mask.sum() + 1e-8)

class IsotonicCalibrator:
    """等渗回归校准器（可替代温度标定）"""
    
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.isotonic_reg = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False
    
    def fit(self, confidences, accuracies):
        """
        训练校准器
        Args:
            confidences: 模型输出概率
            accuracies: 真实二分类标签
        """
        self.isotonic_reg.fit(confidences, accuracies)
        self.fitted = True
    
    def calibrate(self, confidences):
        """校准概率"""
        if not self.fitted:
            return confidences
        return self.isotonic_reg.predict(confidences)

def create_stable_model(config: Optional[StableConfig] = None) -> StableCrossAttnReranker:
    """创建稳健模型"""
    if config is None:
        config = StableConfig()
    
    model = StableCrossAttnReranker(config)
    
    # 初始化权重
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # 小初始化
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

class StableTrainingPipeline:
    """稳健训练流程"""
    
    def __init__(self, config: StableConfig):
        self.config = config
        self.model = create_stable_model(config)
        
        # 损失函数
        self.pairwise_loss = nn.BCEWithLogitsLoss()
        self.listwise_loss = ListMLEWithFocalLoss()
        
        # 校准器
        self.calibrator = IsotonicCalibrator()
        
        # 训练历史
        self.history = {'pairwise_loss': [], 'listwise_loss': [], 'val_metrics': []}
    
    def train_stable_pipeline(self, train_data, val_data, save_dir):
        """两阶段训练：Pairwise Warmup → ListMLE Fine-tune"""
        logger.info("🚀 开始稳健训练流程")
        
        # Stage 1: Pairwise Warmup
        logger.info("Stage 1: Pairwise Warmup")
        self._train_pairwise_warmup(train_data, self.config.warmup_epochs)
        
        # Stage 2: ListMLE Fine-tune
        logger.info("Stage 2: ListMLE Fine-tune")
        self._train_listwise_finetune(train_data, self.config.listwise_epochs)
        
        # Stage 3: Calibration
        logger.info("Stage 3: Calibration")
        self._calibrate_model(val_data)
        
        # 保存模型
        self._save_stable_model(save_dir)
        
        logger.info("✅ 稳健训练完成")
    
    def _train_pairwise_warmup(self, train_data, epochs):
        """Pairwise预热训练"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_data:
                optimizer.zero_grad()
                
                # 提取特征
                features = self._extract_features(batch)
                
                # 前向传播
                outputs = self.model(**features, training=True)
                
                # Pairwise loss
                labels = torch.tensor([item['compliance_score'] > 0.8 for item in batch], 
                                    dtype=torch.float, device=outputs['logits'].device)
                loss = self.pairwise_loss(outputs['logits'].squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.history['pairwise_loss'].append(avg_loss)
            logger.info(f"Pairwise Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    def _train_listwise_finetune(self, train_data, epochs):
        """ListMLE微调训练"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 这里应该按query分组做listwise训练
            # 简化版本，实际需要实现query-level的list采样
            for batch in train_data:
                optimizer.zero_grad()
                
                features = self._extract_features(batch)
                outputs = self.model(**features, training=True)
                
                # 构造list标签（简化版）
                labels = torch.tensor([item.get('dual_score', 0.0) for item in batch], 
                                    dtype=torch.float, device=outputs['logits'].device)
                
                # ListMLE + Focal loss
                loss = self.listwise_loss(outputs['logits'].squeeze().unsqueeze(0), 
                                        labels.unsqueeze(0))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.history['listwise_loss'].append(avg_loss)
            logger.info(f"ListMLE Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    def _calibrate_model(self, val_data):
        """校准模型输出"""
        confidences = []
        accuracies = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in val_data:
                features = self._extract_features(batch)
                outputs = self.model(**features)
                
                probs = torch.sigmoid(outputs['calibrated_logits']).cpu().numpy()
                labels = np.array([item.get('compliance_score', 0) > 0.8 for item in batch])
                
                confidences.extend(probs.flatten())
                accuracies.extend(labels)
        
        # 训练等渗校准器
        self.calibrator.fit(np.array(confidences), np.array(accuracies))
        logger.info("校准器训练完成")
    
    def _extract_features(self, batch):
        """提取特征（与Step5集成）"""
        # 这里应该从实际的Step5输出中提取特征
        # 当前为演示用的mock数据
        return {
            'clip_img': torch.randn(len(batch), 1024),
            'clip_text': torch.randn(len(batch), 1024), 
            'visual_features': torch.randn(len(batch), 8),
            'conflict_features': torch.randn(len(batch), 5)
        }
    
    def _save_stable_model(self, save_dir):
        """保存稳健模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'calibrator': self.calibrator,
            'history': self.history
        }, save_path / 'stable_model.pt')
        
        logger.info(f"模型已保存: {save_path / 'stable_model.pt'}")

# 演示使用
if __name__ == "__main__":
    config = StableConfig(
        hidden_dim=256,
        num_layers=2, 
        total_epochs=12,
        top_m_candidates=20
    )
    
    model = create_stable_model(config)
    
    print("🚀 CoTRR-Stable 稳健模型创建成功!")
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"💾 Hidden dim: {config.hidden_dim}")
    print(f"🎯 性能目标: C@1 +{config.min_compliance_gain}pts, nDCG@10 +{config.min_ndcg_gain}pts")
    print(f"⚡ Top-M策略: 仅前{config.top_m_candidates}候选使用复杂推理")
    print(f"🔬 MC采样: {config.mc_samples}次")
    
    # 测试前向传播
    batch_size = 16
    test_input = {
        'clip_img': torch.randn(batch_size, 1024),
        'clip_text': torch.randn(batch_size, 1024),
        'visual_features': torch.randn(batch_size, 8), 
        'conflict_features': torch.randn(batch_size, 5)
    }
    
    with torch.no_grad():
        output = model(**test_input)
        print(f"✅ 模型输出形状: {output['logits'].shape}")
        print(f"🌡️ 初始温度: {output['temperature'].item():.3f}")
        
        # 测试MC采样
        mc_output = model(**test_input, mc_samples=config.mc_samples)
        print(f"🎲 MC不确定性形状: {mc_output['uncertainty'].shape}")
    
    print("\n📝 使用方式:")
    print("1. pipeline = StableTrainingPipeline(config)")
    print("2. pipeline.train_stable_pipeline(train_data, val_data, save_dir)")
    print("3. A/B测试准入检查")
    
    print("✅ CoTRR-Stable 准备就绪，等待真实数据！")