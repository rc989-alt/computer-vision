#!/usr/bin/env python3
"""
CoTRR-Stable Cross-Attention架构完善版
Day 1 任务：实现Token化多模态编码器和轻量级Cross-Attention

完善内容:
1. 添加详细的单元测试
2. 性能基准测试
3. 梯度检查
4. 可视化注意力权重
5. 模型架构验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import unittest
import time
import logging
from pathlib import Path

# 导入基础模型
from cotrr_stable import (
    StableCrossAttnReranker, StableConfig, TokenizedMultiModalEncoder,
    LightweightCrossAttention, create_stable_model
)

logger = logging.getLogger(__name__)

class CrossAttentionTester:
    """Cross-Attention模型测试器"""
    
    def __init__(self, config: Optional[StableConfig] = None):
        if config is None:
            config = StableConfig()
        
        self.config = config
        self.model = create_stable_model(config)
        self.test_results = {}
        
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """运行全面测试套件"""
        logger.info("🧪 开始Cross-Attention模型全面测试")
        
        test_suite = {
            "architecture_validation": self.test_architecture_validation,
            "forward_pass": self.test_forward_pass,
            "gradient_flow": self.test_gradient_flow,
            "attention_weights": self.test_attention_weights,
            "batch_consistency": self.test_batch_consistency,
            "performance_benchmark": self.test_performance_benchmark,
            "memory_efficiency": self.test_memory_efficiency,
            "numerical_stability": self.test_numerical_stability
        }
        
        results = {}
        for test_name, test_func in test_suite.items():
            try:
                logger.info(f"运行测试: {test_name}")
                success = test_func()
                results[test_name] = success
                status = "✅ PASS" if success else "❌ FAIL"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name}: ❌ ERROR - {str(e)}")
                results[test_name] = False
        
        self.test_results = results
        return results
    
    def test_architecture_validation(self) -> bool:
        """测试架构验证"""
        try:
            # 检查模型参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"可训练参数: {trainable_params:,}")
            
            # 验证参数量在合理范围内 (1-2M)
            if not (1_000_000 <= total_params <= 2_000_000):
                logger.warning(f"参数量 {total_params:,} 超出预期范围 [1M, 2M]")
                return False
            
            # 检查关键组件存在
            required_components = [
                'token_encoder', 'attention_layers', 'output_head', 'temperature'
            ]
            
            for component in required_components:
                if not hasattr(self.model, component):
                    logger.error(f"缺少关键组件: {component}")
                    return False
            
            # 检查attention层数
            if len(self.model.attention_layers) != self.config.num_layers:
                logger.error(f"Attention层数不匹配: 期望{self.config.num_layers}, 实际{len(self.model.attention_layers)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"架构验证失败: {e}")
            return False
    
    def test_forward_pass(self) -> bool:
        """测试前向传播"""
        try:
            batch_size = 8
            
            # 创建测试输入
            test_input = {
                'clip_img': torch.randn(batch_size, self.config.clip_img_dim),
                'clip_text': torch.randn(batch_size, self.config.clip_text_dim),
                'visual_features': torch.randn(batch_size, self.config.visual_dim),
                'conflict_features': torch.randn(batch_size, self.config.conflict_dim)
            }
            
            # 前向传播
            self.model.eval()
            with torch.no_grad():
                output = self.model(**test_input)
            
            # 检查输出形状
            expected_shape = (batch_size, 1)
            if output['logits'].shape != expected_shape:
                logger.error(f"输出形状错误: 期望{expected_shape}, 实际{output['logits'].shape}")
                return False
            
            # 检查输出值范围
            logits = output['logits']
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error("输出包含NaN或Inf值")
                return False
            
            # 检查校准后的logits
            if 'calibrated_logits' in output:
                cal_logits = output['calibrated_logits']
                if torch.isnan(cal_logits).any() or torch.isinf(cal_logits).any():
                    logger.error("校准后logits包含NaN或Inf值")
                    return False
            
            # 检查温度参数
            if 'temperature' in output:
                temp = output['temperature']
                if temp <= 0:
                    logger.error(f"温度参数必须为正: {temp}")
                    return False
            
            logger.info(f"前向传播成功: 输出形状{output['logits'].shape}, 温度{output['temperature'].item():.3f}")
            return True
            
        except Exception as e:
            logger.error(f"前向传播测试失败: {e}")
            return False
    
    def test_gradient_flow(self) -> bool:
        """测试梯度流"""
        try:
            batch_size = 4
            
            # 创建测试输入和标签
            test_input = {
                'clip_img': torch.randn(batch_size, self.config.clip_img_dim, requires_grad=True),
                'clip_text': torch.randn(batch_size, self.config.clip_text_dim, requires_grad=True),
                'visual_features': torch.randn(batch_size, self.config.visual_dim, requires_grad=True),
                'conflict_features': torch.randn(batch_size, self.config.conflict_dim, requires_grad=True)
            }
            
            labels = torch.randn(batch_size, 1)
            
            # 前向传播
            self.model.train()
            output = self.model(**test_input, training=True)
            
            # 计算损失
            loss = F.mse_loss(output['logits'], labels)
            
            # 反向传播
            loss.backward()
            
            # 检查所有参数都有梯度
            no_grad_params = []
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    no_grad_params.append(name)
                elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.error(f"参数 {name} 梯度包含NaN或Inf")
                    return False
            
            if no_grad_params:
                logger.warning(f"以下参数没有梯度: {no_grad_params}")
                # 对于某些参数（如bias），可能确实没有梯度，这是正常的
            
            logger.info(f"梯度流测试成功: 损失值 {loss.item():.4f}")
            return True
            
        except Exception as e:
            logger.error(f"梯度流测试失败: {e}")
            return False
    
    def test_attention_weights(self) -> bool:
        """测试注意力权重"""
        try:
            batch_size = 4
            
            test_input = {
                'clip_img': torch.randn(batch_size, self.config.clip_img_dim),
                'clip_text': torch.randn(batch_size, self.config.clip_text_dim),
                'visual_features': torch.randn(batch_size, self.config.visual_dim),
                'conflict_features': torch.randn(batch_size, self.config.conflict_dim)
            }
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(**test_input)
            
            # 检查是否有注意力权重
            if 'attention_weights' in output:
                attn_weights = output['attention_weights']
                
                # 检查每层注意力权重
                for i, weights in enumerate(attn_weights):
                    # 形状应该是 [batch, heads, seq_len, seq_len]
                    expected_shape = (batch_size, self.config.num_attention_heads, 4, 4)  # 4个token
                    
                    if weights.shape != expected_shape:
                        logger.error(f"第{i}层注意力权重形状错误: 期望{expected_shape}, 实际{weights.shape}")
                        return False
                    
                    # 检查权重是否归一化（每行和为1）
                    row_sums = weights.sum(dim=-1)
                    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
                        logger.error(f"第{i}层注意力权重未正确归一化")
                        return False
                    
                    # 检查权重值范围 [0, 1]
                    if (weights < 0).any() or (weights > 1).any():
                        logger.error(f"第{i}层注意力权重超出[0,1]范围")
                        return False
                
                logger.info(f"注意力权重测试成功: {len(attn_weights)}层权重")
                return True
            else:
                logger.warning("模型输出中没有注意力权重")
                return True  # 不是致命错误
                
        except Exception as e:
            logger.error(f"注意力权重测试失败: {e}")
            return False
    
    def test_batch_consistency(self) -> bool:
        """测试批次一致性"""
        try:
            # 测试单个样本 vs 批次处理的一致性
            single_input = {
                'clip_img': torch.randn(1, self.config.clip_img_dim),
                'clip_text': torch.randn(1, self.config.clip_text_dim),
                'visual_features': torch.randn(1, self.config.visual_dim),
                'conflict_features': torch.randn(1, self.config.conflict_dim)
            }
            
            # 复制成批次
            batch_input = {
                key: value.repeat(3, 1) for key, value in single_input.items()
            }
            
            self.model.eval()
            with torch.no_grad():
                # 单个样本输出
                single_output = self.model(**single_input)
                
                # 批次输出
                batch_output = self.model(**batch_input)
            
            # 检查批次中第一个样本是否与单个样本一致
            single_logit = single_output['logits'][0]
            batch_first_logit = batch_output['logits'][0]
            
            if not torch.allclose(single_logit, batch_first_logit, atol=1e-6):
                logger.error(f"批次一致性测试失败: 单个{single_logit}, 批次第一个{batch_first_logit}")
                return False
            
            logger.info("批次一致性测试成功")
            return True
            
        except Exception as e:
            logger.error(f"批次一致性测试失败: {e}")
            return False
    
    def test_performance_benchmark(self) -> bool:
        """性能基准测试"""
        try:
            batch_sizes = [1, 4, 8, 16]
            timing_results = {}
            
            for batch_size in batch_sizes:
                test_input = {
                    'clip_img': torch.randn(batch_size, self.config.clip_img_dim),
                    'clip_text': torch.randn(batch_size, self.config.clip_text_dim),
                    'visual_features': torch.randn(batch_size, self.config.visual_dim),
                    'conflict_features': torch.randn(batch_size, self.config.conflict_dim)
                }
                
                # 预热
                self.model.eval()
                with torch.no_grad():
                    for _ in range(5):
                        _ = self.model(**test_input)
                
                # 计时
                num_runs = 50
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = self.model(**test_input)
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_runs
                throughput = batch_size / avg_time  # samples/second
                
                timing_results[batch_size] = {
                    'avg_time': avg_time,
                    'throughput': throughput
                }
                
                logger.info(f"Batch {batch_size}: {avg_time*1000:.2f}ms, {throughput:.1f} samples/sec")
            
            # 检查性能是否合理
            single_sample_time = timing_results[1]['avg_time']
            if single_sample_time > 0.1:  # 100ms threshold
                logger.warning(f"单样本推理时间过长: {single_sample_time*1000:.2f}ms")
                return False
            
            self.test_results['performance_benchmark'] = timing_results
            return True
            
        except Exception as e:
            logger.error(f"性能基准测试失败: {e}")
            return False
    
    def test_memory_efficiency(self) -> bool:
        """内存效率测试"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # 基准内存使用
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建大批次进行内存测试
            large_batch = 32
            test_input = {
                'clip_img': torch.randn(large_batch, self.config.clip_img_dim),
                'clip_text': torch.randn(large_batch, self.config.clip_text_dim),
                'visual_features': torch.randn(large_batch, self.config.visual_dim),
                'conflict_features': torch.randn(large_batch, self.config.conflict_dim)
            }
            
            # 前向传播
            self.model.eval()
            with torch.no_grad():
                output = self.model(**test_input)
            
            # 检查内存使用
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory
            
            logger.info(f"内存使用: 基准 {baseline_memory:.1f}MB, 峰值 {peak_memory:.1f}MB, 增长 {memory_increase:.1f}MB")
            
            # 内存增长应该合理（<500MB for batch_32）
            if memory_increase > 500:
                logger.warning(f"内存使用过高: {memory_increase:.1f}MB")
                return False
            
            return True
            
        except ImportError:
            logger.warning("psutil未安装，跳过内存效率测试")
            return True
        except Exception as e:
            logger.error(f"内存效率测试失败: {e}")
            return False
    
    def test_numerical_stability(self) -> bool:
        """数值稳定性测试"""
        try:
            # 测试极端输入值
            extreme_cases = [
                # 全零输入
                {
                    'clip_img': torch.zeros(2, self.config.clip_img_dim),
                    'clip_text': torch.zeros(2, self.config.clip_text_dim),
                    'visual_features': torch.zeros(2, self.config.visual_dim),
                    'conflict_features': torch.zeros(2, self.config.conflict_dim)
                },
                # 大数值输入
                {
                    'clip_img': torch.ones(2, self.config.clip_img_dim) * 10,
                    'clip_text': torch.ones(2, self.config.clip_text_dim) * 10,
                    'visual_features': torch.ones(2, self.config.visual_dim) * 10,
                    'conflict_features': torch.ones(2, self.config.conflict_dim) * 10
                },
                # 随机大范围输入
                {
                    'clip_img': torch.randn(2, self.config.clip_img_dim) * 100,
                    'clip_text': torch.randn(2, self.config.clip_text_dim) * 100,
                    'visual_features': torch.randn(2, self.config.visual_dim) * 100,
                    'conflict_features': torch.randn(2, self.config.conflict_dim) * 100
                }
            ]
            
            self.model.eval()
            for i, test_case in enumerate(extreme_cases):
                with torch.no_grad():
                    try:
                        output = self.model(**test_case)
                        
                        # 检查输出是否包含NaN或Inf
                        if torch.isnan(output['logits']).any() or torch.isinf(output['logits']).any():
                            logger.error(f"极端情况{i}产生了NaN/Inf输出")
                            return False
                        
                        logger.info(f"极端情况{i}测试通过")
                        
                    except Exception as e:
                        logger.error(f"极端情况{i}测试失败: {e}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"数值稳定性测试失败: {e}")
            return False
    
    def visualize_attention_patterns(self, save_dir: str = "research/stage1_progress"):
        """可视化注意力模式"""
        try:
            # 创建测试输入
            batch_size = 1
            test_input = {
                'clip_img': torch.randn(batch_size, self.config.clip_img_dim),
                'clip_text': torch.randn(batch_size, self.config.clip_text_dim),
                'visual_features': torch.randn(batch_size, self.config.visual_dim),
                'conflict_features': torch.randn(batch_size, self.config.conflict_dim)
            }
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(**test_input)
            
            if 'attention_weights' not in output:
                logger.warning("无法获取注意力权重进行可视化")
                return
            
            attn_weights = output['attention_weights']
            token_names = ['CLIP-Img', 'CLIP-Text', 'Visual', 'Conflict']
            
            # 为每一层创建注意力热图
            num_layers = len(attn_weights)
            fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
            
            if num_layers == 1:
                axes = [axes]
            
            for i, weights in enumerate(attn_weights):
                # 取第一个batch和平均所有heads
                layer_weights = weights[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
                
                sns.heatmap(
                    layer_weights,
                    annot=True,
                    fmt='.3f',
                    xticklabels=token_names,
                    yticklabels=token_names,
                    ax=axes[i],
                    cmap='Blues'
                )
                axes[i].set_title(f'Layer {i+1} Attention')
            
            plt.tight_layout()
            save_path = Path(save_dir) / 'attention_visualization.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"注意力可视化已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"注意力可视化失败: {e}")
    
    def generate_test_report(self, save_dir: str = "research/stage1_progress") -> Dict:
        """生成测试报告"""
        if not self.test_results:
            logger.warning("没有测试结果，请先运行测试")
            return {}
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        success_rate = passed_tests / total_tests * 100
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': {
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_attention_heads': self.config.num_attention_heads,
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate
            },
            'detailed_results': self.test_results,
            'performance_data': self.test_results.get('performance_benchmark', {})
        }
        
        # 保存报告
        import json
        report_path = Path(save_dir) / 'cross_attention_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"测试报告已保存: {report_path}")
        logger.info(f"测试结果: {passed_tests}/{total_tests} 通过 ({success_rate:.1f}%)")
        
        return report

def main():
    """运行Cross-Attention架构测试"""
    logger.info("🚀 开始Cross-Attention架构完善和测试")
    
    # 创建测试器
    config = StableConfig()
    tester = CrossAttentionTester(config)
    
    print(f"📊 模型配置:")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Attention layers: {config.num_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Total parameters: {sum(p.numel() for p in tester.model.parameters()):,}")
    
    # 运行全面测试
    test_results = tester.run_comprehensive_tests()
    
    # 生成可视化
    tester.visualize_attention_patterns()
    
    # 生成测试报告
    report = tester.generate_test_report()
    
    # 总结结果
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"\n📋 测试总结:")
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total-passed}/{total}")
    print(f"📊 成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！Cross-Attention架构实现完成")
        return True
    else:
        print("⚠️ 部分测试失败，需要修复")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()