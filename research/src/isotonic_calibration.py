#!/usr/bin/env python3
"""
CoTRR-Stable Stage 1 Task T004: Isotonic校准实现
实现概率校准机制，提高模型输出的可靠性

关键特性:
1. Isotonic Regression: 非参数化概率校准
2. ECE指标: Expected Calibration Error计算
3. 校准器保存/加载: 生产环境持久化
4. 可视化分析: 校准前后效果对比
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsotonicCalibrator:
    """
    Isotonic Regression校准器
    用于将模型输出校准为可靠的概率估计
    """
    
    def __init__(self, out_of_bounds='clip'):
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)
        self.is_fitted = False
        self.calibration_metrics = {}
        
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        拟合校准器
        Args:
            scores: 模型原始分数 [N,] (logits)
            labels: 二值标签 [N,] (0/1)
        Returns:
            校准指标字典
        """
        logger.info(f"🎯 开始训练Isotonic校准器 (样本数: {len(scores)})")
        
        # 将logits转换为概率
        probs = self._logits_to_probs(scores)
        
        # 拟合isotonic regression
        self.calibrator.fit(probs, labels)
        self.is_fitted = True
        
        # 计算校准前后的指标
        calibrated_probs = self.calibrator.predict(probs)
        
        # 计算各种校准指标
        metrics = self._compute_calibration_metrics(probs, calibrated_probs, labels)
        self.calibration_metrics = metrics
        
        logger.info(f"📊 校准效果:")
        logger.info(f"   ECE: {metrics['original_ece']:.4f} → {metrics['calibrated_ece']:.4f}")
        logger.info(f"   Brier Score: {metrics['original_brier']:.4f} → {metrics['calibrated_brier']:.4f}")
        logger.info(f"   改善度: ECE {metrics['ece_improvement']:.4f}, Brier {metrics['brier_improvement']:.4f}")
        
        return metrics
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        校准预测概率
        Args:
            scores: 模型原始分数 (logits)
        Returns:
            校准后的概率 [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("校准器未拟合，请先调用fit()")
        
        probs = self._logits_to_probs(scores)
        return self.calibrator.predict(probs)
    
    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """将logits转换为概率"""
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))  # 防止数值溢出
    
    def _compute_calibration_metrics(self, 
                                   original_probs: np.ndarray, 
                                   calibrated_probs: np.ndarray, 
                                   labels: np.ndarray) -> Dict[str, float]:
        """计算校准指标"""
        
        # ECE (Expected Calibration Error)
        original_ece = self._compute_ece(original_probs, labels)
        calibrated_ece = self._compute_ece(calibrated_probs, labels)
        
        # Brier Score
        original_brier = brier_score_loss(labels, original_probs)
        calibrated_brier = brier_score_loss(labels, calibrated_probs)
        
        # Log Loss
        original_logloss = log_loss(labels, np.clip(original_probs, 1e-15, 1-1e-15))
        calibrated_logloss = log_loss(labels, np.clip(calibrated_probs, 1e-15, 1-1e-15))
        
        # 可靠性指标 (Reliability)
        original_reliability = self._compute_reliability(original_probs, labels)
        calibrated_reliability = self._compute_reliability(calibrated_probs, labels)
        
        return {
            'original_ece': original_ece,
            'calibrated_ece': calibrated_ece,
            'ece_improvement': original_ece - calibrated_ece,
            'original_brier': original_brier,
            'calibrated_brier': calibrated_brier,
            'brier_improvement': original_brier - calibrated_brier,
            'original_logloss': original_logloss,
            'calibrated_logloss': calibrated_logloss,
            'logloss_improvement': original_logloss - calibrated_logloss,
            'original_reliability': original_reliability,
            'calibrated_reliability': calibrated_reliability,
            'reliability_improvement': original_reliability - calibrated_reliability
        }
    
    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        计算Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(probs)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_reliability(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """计算可靠性指标"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability = 0
        total_samples = len(probs)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                reliability += (avg_confidence_in_bin - accuracy_in_bin) ** 2 * prop_in_bin
        
        return reliability
    
    def plot_calibration_curve(self, 
                              scores: np.ndarray, 
                              labels: np.ndarray, 
                              save_path: Optional[str] = None,
                              n_bins: int = 10):
        """绘制校准曲线"""
        if not self.is_fitted:
            raise ValueError("校准器未拟合，请先调用fit()")
        
        original_probs = self._logits_to_probs(scores)
        calibrated_probs = self.predict(scores)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原始校准曲线
        self._plot_single_calibration(original_probs, labels, ax1, "原始模型", n_bins)
        
        # 校准后曲线
        self._plot_single_calibration(calibrated_probs, labels, ax2, "校准后模型", n_bins)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 校准曲线已保存: {save_path}")
        
        plt.show()
    
    def _plot_single_calibration(self, probs, labels, ax, title, n_bins):
        """绘制单个校准曲线"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(labels[in_bin].mean())
                bin_confidences.append(probs[in_bin].mean())
                bin_counts.append(in_bin.sum())
        
        # 校准曲线
        ax.plot(bin_confidences, bin_accuracies, 'o-', label=f'{title}校准曲线')
        ax.plot([0, 1], [0, 1], 'k--', label='完美校准')
        
        # 设置图形
        ax.set_xlabel('平均预测概率')
        ax.set_ylabel('准确率')
        ax.set_title(f'{title}校准曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加ECE数值
        ece = self._compute_ece(np.array(bin_confidences), np.array(bin_accuracies))
        ax.text(0.02, 0.98, f'ECE: {ece:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    def save(self, path: str):
        """保存校准器"""
        calibrator_data = {
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted,
            'metrics': self.calibration_metrics
        }
        
        with open(path, 'wb') as f:
            pickle.dump(calibrator_data, f)
        
        logger.info(f"💾 校准器已保存: {path}")
    
    def load(self, path: str):
        """加载校准器"""
        with open(path, 'rb') as f:
            calibrator_data = pickle.load(f)
        
        self.calibrator = calibrator_data['calibrator']
        self.is_fitted = calibrator_data['is_fitted']
        self.calibration_metrics = calibrator_data.get('metrics', {})
        
        logger.info(f"📂 校准器已加载: {path}")
    
    def get_metrics(self) -> Dict[str, float]:
        """获取校准指标"""
        return self.calibration_metrics.copy()

def test_isotonic_calibrator():
    """全面测试Isotonic校准器"""
    logger.info("🧪 开始全面测试Isotonic校准器")
    
    # 生成更真实的模拟数据
    np.random.seed(42)
    n_samples = 2000
    
    # 模拟真实场景：模型过度自信
    # 真实概率服从Beta分布
    true_probs = np.random.beta(2, 5, n_samples)  # 偏向低概率
    labels = np.random.binomial(1, true_probs)
    
    # 模拟过度自信的模型输出 (logits偏高)
    noise = np.random.normal(0, 0.8, n_samples)
    confidence_bias = 1.5  # 过度自信偏差
    raw_logits = np.log(true_probs / (1 - true_probs + 1e-8)) + confidence_bias + noise
    
    logger.info(f"数据统计: {n_samples}样本, 正例比例: {labels.mean():.3f}")
    
    # 创建并训练校准器
    calibrator = IsotonicCalibrator()
    metrics = calibrator.fit(raw_logits, labels)
    
    # 测试预测功能
    test_logits = np.random.normal(0, 2, 500)
    calibrated_probs = calibrator.predict(test_logits)
    
    logger.info(f"测试预测: 校准概率范围 [{calibrated_probs.min():.3f}, {calibrated_probs.max():.3f}]")
    
    # 测试保存和加载
    save_path = "research/stage1_progress/test_calibrator.pkl"
    calibrator.save(save_path)
    
    # 加载测试
    new_calibrator = IsotonicCalibrator()
    new_calibrator.load(save_path)
    
    # 验证加载后的一致性
    new_probs = new_calibrator.predict(test_logits)
    consistency_check = np.allclose(calibrated_probs, new_probs)
    
    logger.info(f"保存/加载一致性: {'✅ 通过' if consistency_check else '❌ 失败'}")
    
    # 生成校准曲线
    try:
        calibrator.plot_calibration_curve(
            raw_logits, labels, 
            save_path="research/stage1_progress/calibration_curve.png"
        )
    except Exception as e:
        logger.warning(f"校准曲线生成失败: {e}")
    
    # 验证指标改善
    improvements = {
        'ECE改善': metrics['ece_improvement'] > 0,
        'Brier改善': metrics['brier_improvement'] > 0,
        'ECE达标': metrics['calibrated_ece'] < 0.05,  # 目标ECE < 0.05
    }
    
    logger.info("✅ 校准器测试完成")
    logger.info("📊 改善验证:")
    for metric, improved in improvements.items():
        status = "✅ 达标" if improved else "⚠️ 未达标"
        logger.info(f"   {metric}: {status}")
    
    # 清理测试文件
    try:
        Path(save_path).unlink()
        Path("research/stage1_progress/calibration_curve.png").unlink(missing_ok=True)
    except:
        pass
    
    return calibrator, metrics, all(improvements.values())

if __name__ == "__main__":
    logger.info("🚀 开始Task T004: Isotonic校准实现")
    
    # 创建输出目录
    Path("research/stage1_progress").mkdir(parents=True, exist_ok=True)
    
    # 运行全面测试
    calibrator, metrics, all_tests_passed = test_isotonic_calibrator()
    
    if all_tests_passed:
        logger.info("🎉 Task T004实现完成！")
        logger.info("📋 交付内容:")
        logger.info("  - IsotonicCalibrator类: 完整概率校准器")
        logger.info("  - ECE/Brier/LogLoss指标计算")
        logger.info("  - 校准曲线可视化分析")
        logger.info("  - 校准器保存/加载功能")
        logger.info("  - 生产就绪的校准接口")
        logger.info("  - 全面测试验证通过")
        
        logger.info("📊 最终指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.4f}")
    else:
        logger.error("❌ Task T004部分测试未通过，需要进一步优化")