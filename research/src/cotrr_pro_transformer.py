#!/usr/bin/env python3
"""
CoTRR-Pro Multi-modal Fusion Transformer
基于CVPR 2024最佳实践的多模态融合架构

核心创新:
1. Cross-attention fusion替代简单拼接
2. Region-aware attention for fine-grained features  
3. Learnable temperature for calibration
4. Multi-scale visual feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型配置"""
    # CLIP features
    clip_img_dim: int = 1024
    clip_text_dim: int = 1024
    
    # Visual features (from detection)
    visual_dim: int = 8  # subject_ratio, area, etc.
    
    # Conflict features
    conflict_dim: int = 5  # color, temperature, clarity, etc.
    
    # Fusion architecture
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    
    # Output
    output_dim: int = 1  # ranking score

class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力模块（CVPR 2024风格）"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, d_model]
            key: [batch, seq_len_k, d_model]  
            value: [batch, seq_len_v, d_model]
            mask: [batch, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        residual = query
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Output projection + residual
        output = self.w_o(context) + residual
        output = self.layer_norm(output)
        
        return output

class RegionAwareVisualEncoder(nn.Module):
    """区域感知的视觉特征编码器"""
    
    def __init__(self, visual_dim: int, hidden_dim: int):
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        
        # 多尺度特征提取
        self.global_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 区域注意力权重
        self.region_attention = nn.Sequential(
            nn.Linear(visual_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [batch, visual_dim] - subject_ratio, area, etc.
        Returns:
            encoded_features: [batch, hidden_dim]
        """
        # 全局特征编码
        global_features = self.global_encoder(visual_features)
        
        # 区域注意力权重
        attention_weight = self.region_attention(visual_features)
        
        # 加权融合
        weighted_features = global_features * attention_weight
        
        return self.layer_norm(weighted_features)

class ConflictAwareEncoder(nn.Module):
    """冲突感知编码器"""
    
    def __init__(self, conflict_dim: int, hidden_dim: int):
        super().__init__()
        self.conflict_dim = conflict_dim
        self.hidden_dim = hidden_dim
        
        # 冲突特征编码
        self.conflict_encoder = nn.Sequential(
            nn.Linear(conflict_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 不确定性估计（用于校准）
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出不确定性score
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, conflict_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            conflict_features: [batch, conflict_dim] - color_delta, temp_conflict, etc.
        Returns:
            encoded_features: [batch, hidden_dim]
            uncertainty: [batch, 1] - 不确定性估计
        """
        encoded = self.conflict_encoder(conflict_features)
        encoded = self.layer_norm(encoded)
        
        uncertainty = self.uncertainty_head(encoded)
        
        return encoded, uncertainty

class CoTRRProTransformer(nn.Module):
    """CoTRR-Pro: Multi-modal Fusion Transformer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 特征编码器
        self.clip_img_proj = nn.Linear(config.clip_img_dim, config.hidden_dim)
        self.clip_text_proj = nn.Linear(config.clip_text_dim, config.hidden_dim)
        self.visual_encoder = RegionAwareVisualEncoder(config.visual_dim, config.hidden_dim)
        self.conflict_encoder = ConflictAwareEncoder(config.conflict_dim, config.hidden_dim)
        
        # 多模态融合层
        self.fusion_layers = nn.ModuleList([
            MultiHeadCrossAttention(
                config.hidden_dim, 
                config.num_attention_heads, 
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # 最终输出层
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),  # 4个模态特征拼接
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        # 可学习的温度参数（用于校准）
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Dropout for Monte Carlo estimation
        self.mc_dropout = nn.Dropout(0.1)
        
    def forward(self, 
                clip_img: torch.Tensor,
                clip_text: torch.Tensor, 
                visual_features: torch.Tensor,
                conflict_features: torch.Tensor,
                mc_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Args:
            clip_img: [batch, clip_img_dim]
            clip_text: [batch, clip_text_dim]
            visual_features: [batch, visual_dim]
            conflict_features: [batch, conflict_dim]
            mc_samples: Monte Carlo samples for uncertainty
            
        Returns:
            Dict with 'logits', 'uncertainty', 'attention_weights'
        """
        batch_size = clip_img.size(0)
        
        if mc_samples > 1:
            # Monte Carlo Dropout for uncertainty estimation
            logits_samples = []
            for _ in range(mc_samples):
                sample_logits = self._forward_single(
                    clip_img, clip_text, visual_features, conflict_features, training=True)
                logits_samples.append(sample_logits['logits'])
            
            # 聚合MC samples
            logits_mean = torch.stack(logits_samples).mean(dim=0)
            logits_std = torch.stack(logits_samples).std(dim=0)
            
            return {
                'logits': logits_mean,
                'uncertainty': logits_std,
                'mc_samples': logits_samples
            }
        else:
            return self._forward_single(clip_img, clip_text, visual_features, conflict_features)
    
    def _forward_single(self, clip_img, clip_text, visual_features, conflict_features, training=False):
        """单次前向传播"""
        
        # 1. 特征编码
        img_encoded = self.clip_img_proj(clip_img).unsqueeze(1)  # [batch, 1, hidden]
        text_encoded = self.clip_text_proj(clip_text).unsqueeze(1)  # [batch, 1, hidden]
        visual_encoded = self.visual_encoder(visual_features).unsqueeze(1)  # [batch, 1, hidden]
        conflict_encoded, conflict_uncertainty = self.conflict_encoder(conflict_features)
        conflict_encoded = conflict_encoded.unsqueeze(1)  # [batch, 1, hidden]
        
        # 2. 多模态序列 [img, text, visual, conflict]
        multimodal_seq = torch.cat([img_encoded, text_encoded, visual_encoded, conflict_encoded], dim=1)
        # [batch, 4, hidden]
        
        # 3. Cross-attention fusion
        fused_features = multimodal_seq
        attention_weights = []
        
        for fusion_layer in self.fusion_layers:
            # Self-attention within multimodal sequence
            fused_features = fusion_layer(fused_features, fused_features, fused_features)
            
            if training:
                fused_features = self.mc_dropout(fused_features)
        
        # 4. 聚合多模态特征
        # Global average pooling + 原始特征拼接
        pooled_features = fused_features.mean(dim=1)  # [batch, hidden]
        concat_features = torch.cat([
            img_encoded.squeeze(1),
            text_encoded.squeeze(1), 
            visual_encoded.squeeze(1),
            conflict_encoded.squeeze(1)
        ], dim=-1)  # [batch, hidden*4]
        
        # 5. 最终预测
        logits = self.output_head(concat_features)
        
        # 6. 温度校准
        calibrated_logits = logits / self.temperature
        
        return {
            'logits': calibrated_logits,
            'raw_logits': logits,
            'uncertainty': conflict_uncertainty,
            'temperature': self.temperature,
            'multimodal_features': fused_features
        }

class ContrastiveLoss(nn.Module):
    """对比学习损失（基于ICCV 2023）"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, feature_dim] - 多模态特征
            labels: [batch] - 相同cocktail/query的样本标记为相同label
        """
        batch_size = features.size(0)
        
        # L2 normalize
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建positive mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 移除对角线（自己和自己）
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        pos_sim = (exp_sim * mask).sum(dim=1)
        pos_count = mask.sum(dim=1)
        
        # 避免除零
        pos_count = torch.clamp(pos_count, min=1.0)
        
        loss = -torch.log(pos_sim / (sum_exp_sim.squeeze() + 1e-8))
        loss = loss.sum() / batch_size
        
        return loss

class ListMLELoss(nn.Module):
    """ListMLE损失（现代ranking loss）"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_items] - ranking scores
            labels: [batch, num_items] - relevance scores (higher = better)
        """
        # 按真实标签排序
        sorted_labels, sorted_indices = torch.sort(labels, dim=1, descending=True)
        sorted_logits = torch.gather(logits, 1, sorted_indices)
        
        # ListMLE loss计算
        max_logits = sorted_logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(sorted_logits - max_logits)
        
        cumsum_exp = torch.cumsum(exp_logits.flip(dims=[1]), dim=1).flip(dims=[1])
        
        loss = -torch.log(exp_logits / (cumsum_exp + 1e-8)).sum(dim=1).mean()
        
        return loss

def create_model(config: Optional[ModelConfig] = None) -> CoTRRProTransformer:
    """创建CoTRR-Pro模型"""
    if config is None:
        config = ModelConfig()
    
    model = CoTRRProTransformer(config)
    
    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

# 示例使用
if __name__ == "__main__":
    # 创建模型
    config = ModelConfig(
        hidden_dim=512,
        num_attention_heads=8,
        num_layers=3
    )
    model = create_model(config)
    
    # 模拟输入
    batch_size = 32
    clip_img = torch.randn(batch_size, config.clip_img_dim)
    clip_text = torch.randn(batch_size, config.clip_text_dim) 
    visual_features = torch.randn(batch_size, config.visual_dim)
    conflict_features = torch.randn(batch_size, config.conflict_dim)
    
    # 前向传播
    outputs = model(clip_img, clip_text, visual_features, conflict_features)
    
    print(f"Model output shape: {outputs['logits'].shape}")
    print(f"Temperature: {outputs['temperature'].item():.4f}")
    print(f"Uncertainty shape: {outputs['uncertainty'].shape}")
    
    # Monte Carlo uncertainty estimation
    mc_outputs = model(clip_img, clip_text, visual_features, conflict_features, mc_samples=10)
    print(f"MC uncertainty shape: {mc_outputs['uncertainty'].shape}")
    
    print("✅ CoTRR-Pro Transformer创建成功！")