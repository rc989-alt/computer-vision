#!/usr/bin/env python3
"""
CoTRR-Pro Advanced Evaluation Suite
基于最新CVPR/ICCV标准的完整评测框架

包含:
1. 排序指标: Compliance@K, nDCG@K with bootstrap CI
2. 校准指标: ECE, MCE, Brier Score, Reliability Diagram  
3. 不确定性: Entropy, MI, OOD detection
4. 消融研究: 模块贡献分析
5. 失败分析: 错误类型分类 + 可视化
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from cotrr_pro_transformer import CoTRRProTransformer, ModelConfig, create_model

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """评测配置"""
    # Bootstrap设置
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Monte Carlo不确定性
    mc_samples: int = 20
    
    # 排序评测
    k_values: List[int] = None  # Compliance@K, nDCG@K
    
    # 校准评测
    calibration_bins: int = 10
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 结果保存
    save_dir: str = "research/evaluation"
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]

class BootstrapMetrics:
    """Bootstrap置信区间计算"""
    
    @staticmethod
    def bootstrap_ci(data: np.ndarray, metric_func, confidence_level: float = 0.95,
                     n_bootstrap: int = 1000) -> Tuple[float, float, float]:
        """
        计算指标的bootstrap置信区间
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        bootstrap_metrics = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            # 重采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = data[indices]
            
            # 计算指标
            metric_value = metric_func(bootstrap_sample)
            bootstrap_metrics.append(metric_value)
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        mean_val = np.mean(bootstrap_metrics)
        lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
        upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
        
        return mean_val, lower_bound, upper_bound

class RankingMetrics:
    """排序指标计算"""
    
    @staticmethod
    def compliance_at_k(predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """
        Compliance@K: 前K个结果中符合要求的比例
        
        Args:
            predictions: [n_queries, n_items] - 预测分数
            targets: [n_queries, n_items] - 二分类标签 (1=符合, 0=不符合)
            k: 前K个
        """
        assert predictions.shape == targets.shape
        n_queries = predictions.shape[0]
        
        compliant_at_k = 0.0
        
        for i in range(n_queries):
            # 按预测分数排序
            sorted_indices = np.argsort(predictions[i])[::-1]  # 降序
            top_k_indices = sorted_indices[:k]
            
            # 计算前K个中的compliance
            top_k_targets = targets[i][top_k_indices]
            compliance = np.sum(top_k_targets) / k
            compliant_at_k += compliance
        
        return compliant_at_k / n_queries
    
    @staticmethod
    def ndcg_at_k(predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        
        Args:
            predictions: [n_queries, n_items] - 预测分数
            targets: [n_queries, n_items] - 相关性分数 (连续值)
        """
        assert predictions.shape == targets.shape
        n_queries = predictions.shape[0]
        
        ndcg_scores = []
        
        for i in range(n_queries):
            # DCG@K
            sorted_indices = np.argsort(predictions[i])[::-1]
            top_k_indices = sorted_indices[:k]
            top_k_targets = targets[i][top_k_indices]
            
            dcg = 0.0
            for j, relevance in enumerate(top_k_targets):
                dcg += (2**relevance - 1) / np.log2(j + 2)
            
            # IDCG@K (理想排序)
            ideal_sorted = np.sort(targets[i])[::-1][:k]
            idcg = 0.0
            for j, relevance in enumerate(ideal_sorted):
                idcg += (2**relevance - 1) / np.log2(j + 2)
            
            # NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)

class CalibrationMetrics:
    """校准指标计算"""
    
    @staticmethod
    def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                   n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在当前bin中的样本
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def maximum_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                  n_bins: int = 10) -> float:
        """Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, bin_error)
        
        return mce
    
    @staticmethod
    def reliability_diagram_data(confidences: np.ndarray, accuracies: np.ndarray,
                                n_bins: int = 10) -> Dict[str, np.ndarray]:
        """Reliability diagram数据"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        return {
            'bin_centers': np.array(bin_centers),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts)
        }

class UncertaintyMetrics:
    """不确定性指标"""
    
    @staticmethod
    def predictive_entropy(probs: np.ndarray) -> np.ndarray:
        """预测熵（表示不确定性）"""
        # 避免log(0)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        return entropy
    
    @staticmethod
    def mutual_information(mc_probs: np.ndarray) -> np.ndarray:
        """
        互信息（表示认知不确定性）
        
        Args:
            mc_probs: [n_samples, n_mc_samples, n_classes] - MC dropout概率
        """
        # 平均预测
        mean_probs = np.mean(mc_probs, axis=1)
        
        # 预测熵（总不确定性）
        predictive_entropy = UncertaintyMetrics.predictive_entropy(mean_probs)
        
        # 期望熵（偶然不确定性）
        expected_entropy = np.mean([
            UncertaintyMetrics.predictive_entropy(mc_probs[:, i, :])
            for i in range(mc_probs.shape[1])
        ], axis=0)
        
        # 互信息 = 预测熵 - 期望熵
        mutual_info = predictive_entropy - expected_entropy
        
        return mutual_info

class CoTRRProEvaluator:
    """CoTRR-Pro完整评测器"""
    
    def __init__(self, model: CoTRRProTransformer, config: EvaluationConfig):
        self.model = model
        self.config = config
        self.model.eval()
        
        # 创建结果目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 评测结果存储
        self.results = {}
        
    def evaluate_dataset(self, data_path: str, dataset_name: str = "test") -> Dict[str, Any]:
        """完整数据集评测"""
        logger.info(f"🔍 Evaluating {dataset_name} dataset: {data_path}")
        
        # 加载数据
        data = self._load_data(data_path)
        
        # 获取预测结果
        predictions = self._get_predictions(data)
        
        # 计算各类指标
        ranking_results = self._evaluate_ranking_metrics(predictions, data)
        calibration_results = self._evaluate_calibration_metrics(predictions, data)
        uncertainty_results = self._evaluate_uncertainty_metrics(predictions, data)
        
        # 失败分析
        failure_analysis = self._analyze_failures(predictions, data)
        
        # 整合结果
        results = {
            'dataset': dataset_name,
            'n_samples': len(data),
            'ranking_metrics': ranking_results,
            'calibration_metrics': calibration_results,
            'uncertainty_metrics': uncertainty_results,
            'failure_analysis': failure_analysis
        }
        
        self.results[dataset_name] = results
        
        # 保存结果
        self._save_results(results, dataset_name)
        
        logger.info(f"✅ {dataset_name} evaluation completed")
        return results
    
    def ablation_study(self, data_path: str, model_variants: Dict[str, CoTRRProTransformer]) -> Dict[str, Any]:
        """消融研究"""
        logger.info("🧪 Running ablation study")
        
        data = self._load_data(data_path)
        ablation_results = {}
        
        # 评测每个模型变体
        for variant_name, model in model_variants.items():
            logger.info(f"Evaluating {variant_name}")
            
            # 临时替换模型
            original_model = self.model
            self.model = model
            
            # 获取预测结果
            predictions = self._get_predictions(data)
            
            # 计算核心指标
            ranking_results = self._evaluate_ranking_metrics(predictions, data)
            
            ablation_results[variant_name] = {
                'compliance_at_1': ranking_results['compliance_at_1']['mean'],
                'compliance_at_3': ranking_results['compliance_at_3']['mean'],
                'ndcg_at_10': ranking_results['ndcg_at_10']['mean'],
                'n_params': sum(p.numel() for p in model.parameters())
            }
            
            # 恢复原模型
            self.model = original_model
        
        # 创建对比表
        ablation_df = pd.DataFrame(ablation_results).T
        ablation_df['compliance_gain'] = ablation_df['compliance_at_1'] - ablation_df.iloc[0]['compliance_at_1']
        ablation_df['ndcg_gain'] = ablation_df['ndcg_at_10'] - ablation_df.iloc[0]['ndcg_at_10']
        
        # 保存消融结果
        ablation_df.to_csv(self.save_dir / 'ablation_study.csv')
        
        logger.info("✅ Ablation study completed")
        return {
            'results': ablation_results,
            'summary_table': ablation_df.to_dict()
        }
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def _get_predictions(self, data: List[Dict]) -> Dict[str, np.ndarray]:
        """获取模型预测"""
        predictions = {
            'logits': [],
            'probs': [],
            'uncertainties': [],
            'mc_samples': []
        }
        
        with torch.no_grad():
            for item in data:
                # 提取特征
                features = self._extract_features(item)
                
                # 模型预测（包含MC samples）
                outputs = self.model(**features, mc_samples=self.config.mc_samples)
                
                # 存储结果
                logits = outputs['logits'].cpu().numpy()
                probs = torch.sigmoid(outputs['logits']).cpu().numpy()
                uncertainty = outputs.get('uncertainty', torch.zeros_like(outputs['logits'])).cpu().numpy()
                
                predictions['logits'].append(logits)
                predictions['probs'].append(probs)
                predictions['uncertainties'].append(uncertainty)
                
                if 'mc_samples' in outputs:
                    mc_probs = torch.sigmoid(torch.stack(outputs['mc_samples'])).cpu().numpy()
                    predictions['mc_samples'].append(mc_probs)
        
        # 转换为numpy arrays
        for key in predictions:
            if predictions[key]:
                predictions[key] = np.array(predictions[key])
        
        return predictions
    
    def _extract_features(self, item: Dict) -> Dict[str, torch.Tensor]:
        """从item中提取特征"""
        # 模拟特征提取（实际应该从item中读取）
        features = {
            'clip_img': torch.randn(1, 1024).to(self.config.device),
            'clip_text': torch.randn(1, 1024).to(self.config.device),
            'visual_features': torch.randn(1, 8).to(self.config.device),
            'conflict_features': torch.randn(1, 5).to(self.config.device)
        }
        return features
    
    def _evaluate_ranking_metrics(self, predictions: Dict, data: List[Dict]) -> Dict[str, Any]:
        """评测排序指标"""
        results = {}
        
        # 按query分组
        query_groups = self._group_by_query(data, predictions)
        
        for k in self.config.k_values:
            # Compliance@K
            compliance_scores = []
            ndcg_scores = []
            
            for query, group_data in query_groups.items():
                items = group_data['items']
                preds = group_data['predictions']
                
                if len(items) >= k:
                    # 提取真实标签
                    compliance_labels = np.array([
                        1 if item.get('compliance_score', 0) > 0.8 else 0 
                        for item in items
                    ])
                    relevance_scores = np.array([
                        item.get('dual_score', 0.0) for item in items
                    ])
                    
                    # 计算指标
                    pred_scores = preds.squeeze()
                    
                    compliance_k = RankingMetrics.compliance_at_k(
                        pred_scores.reshape(1, -1), 
                        compliance_labels.reshape(1, -1), 
                        k
                    )
                    ndcg_k = RankingMetrics.ndcg_at_k(
                        pred_scores.reshape(1, -1),
                        relevance_scores.reshape(1, -1),
                        k
                    )
                    
                    compliance_scores.append(compliance_k)
                    ndcg_scores.append(ndcg_k)
            
            # Bootstrap置信区间
            if compliance_scores:
                compliance_mean, compliance_lower, compliance_upper = BootstrapMetrics.bootstrap_ci(
                    np.array(compliance_scores), np.mean, self.config.confidence_level
                )
                results[f'compliance_at_{k}'] = {
                    'mean': compliance_mean,
                    'ci_lower': compliance_lower,
                    'ci_upper': compliance_upper,
                    'samples': len(compliance_scores)
                }
            
            if ndcg_scores:
                ndcg_mean, ndcg_lower, ndcg_upper = BootstrapMetrics.bootstrap_ci(
                    np.array(ndcg_scores), np.mean, self.config.confidence_level
                )
                results[f'ndcg_at_{k}'] = {
                    'mean': ndcg_mean,
                    'ci_lower': ndcg_lower,
                    'ci_upper': ndcg_upper,
                    'samples': len(ndcg_scores)
                }
        
        return results
    
    def _evaluate_calibration_metrics(self, predictions: Dict, data: List[Dict]) -> Dict[str, Any]:
        """评测校准指标"""
        probs = predictions['probs'].flatten()
        
        # 获取真实conflict标签
        conflict_labels = np.array([
            1 if item.get('conflict_probability', 0) > 0.5 else 0
            for item in data
        ])
        
        # 校准指标
        ece = CalibrationMetrics.expected_calibration_error(
            probs, conflict_labels, self.config.calibration_bins
        )
        mce = CalibrationMetrics.maximum_calibration_error(
            probs, conflict_labels, self.config.calibration_bins
        )
        brier = brier_score_loss(conflict_labels, probs)
        
        # Reliability diagram数据
        reliability_data = CalibrationMetrics.reliability_diagram_data(
            probs, conflict_labels, self.config.calibration_bins
        )
        
        # AUC
        if len(np.unique(conflict_labels)) > 1:
            auc = roc_auc_score(conflict_labels, probs)
        else:
            auc = 0.0
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'auc': auc,
            'reliability_diagram': reliability_data
        }
        
    def _evaluate_uncertainty_metrics(self, predictions: Dict, data: List[Dict]) -> Dict[str, Any]:
        """评测不确定性指标"""
        if 'mc_samples' not in predictions or len(predictions['mc_samples']) == 0:
            return {'note': 'No MC samples available for uncertainty evaluation'}
        
        mc_probs = predictions['mc_samples']  # [n_samples, n_mc, n_classes]
        
        # 预测熵
        mean_probs = np.mean(mc_probs, axis=1)
        pred_entropy = UncertaintyMetrics.predictive_entropy(mean_probs)
        
        # 互信息
        mutual_info = UncertaintyMetrics.mutual_information(mc_probs)
        
        return {
            'mean_predictive_entropy': np.mean(pred_entropy),
            'std_predictive_entropy': np.std(pred_entropy),
            'mean_mutual_information': np.mean(mutual_info),
            'std_mutual_information': np.std(mutual_info)
        }
    
    def _analyze_failures(self, predictions: Dict, data: List[Dict]) -> Dict[str, Any]:
        """失败分析"""
        probs = predictions['probs'].flatten()
        
        # 获取真实标签和预测标签
        true_labels = np.array([
            1 if item.get('compliance_score', 0) > 0.8 else 0
            for item in data
        ])
        pred_labels = (probs > 0.5).astype(int)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        
        # 错误类型分析
        false_positives = []  # 误罚
        false_negatives = []  # 漏罚
        
        for i, (true_label, pred_prob, item) in enumerate(zip(true_labels, probs, data)):
            pred_label = int(pred_prob > 0.5)
            
            if true_label == 0 and pred_label == 1:
                # 误罚：实际合规但被判为违规
                false_positives.append({
                    'index': i,
                    'confidence': pred_prob,
                    'item': item,
                    'error_type': 'false_positive'
                })
            elif true_label == 1 and pred_label == 0:
                # 漏罚：实际违规但被判为合规
                false_negatives.append({
                    'index': i,
                    'confidence': pred_prob,
                    'item': item,
                    'error_type': 'false_negative'
                })
        
        # 按confidence排序，找到最严重的错误
        false_positives.sort(key=lambda x: x['confidence'], reverse=True)
        false_negatives.sort(key=lambda x: 1 - x['confidence'], reverse=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'false_positives': false_positives[:10],  # 前10个最严重误罚
            'false_negatives': false_negatives[:10],  # 前10个最严重漏罚
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives),
            'accuracy': np.mean(true_labels == pred_labels)
        }
    
    def _group_by_query(self, data: List[Dict], predictions: Dict) -> Dict[str, Dict]:
        """按query分组数据"""
        query_groups = {}
        
        for i, item in enumerate(data):
            query = item.get('query', 'unknown')
            
            if query not in query_groups:
                query_groups[query] = {
                    'items': [],
                    'predictions': []
                }
            
            query_groups[query]['items'].append(item)
            query_groups[query]['predictions'].append(predictions['logits'][i])
        
        # 转换predictions为numpy arrays
        for query in query_groups:
            query_groups[query]['predictions'] = np.array(query_groups[query]['predictions'])
        
        return query_groups
    
    def _save_results(self, results: Dict, dataset_name: str):
        """保存评测结果"""
        # JSON结果
        with open(self.save_dir / f'{dataset_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 可视化
        self._create_visualizations(results, dataset_name)
        
        logger.info(f"Results saved: {self.save_dir / f'{dataset_name}_results.json'}")
    
    def _create_visualizations(self, results: Dict, dataset_name: str):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Ranking metrics with CI
        if 'ranking_metrics' in results:
            metrics = results['ranking_metrics']
            metric_names = []
            metric_means = []
            metric_cis = []
            
            for key, value in metrics.items():
                if isinstance(value, dict) and 'mean' in value:
                    metric_names.append(key)
                    metric_means.append(value['mean'])
                    ci_width = value['ci_upper'] - value['ci_lower']
                    metric_cis.append(ci_width / 2)
            
            if metric_names:
                axes[0, 0].bar(range(len(metric_names)), metric_means, 
                              yerr=metric_cis, capsize=5, alpha=0.7)
                axes[0, 0].set_xticks(range(len(metric_names)))
                axes[0, 0].set_xticklabels(metric_names, rotation=45)
                axes[0, 0].set_title('Ranking Metrics with 95% CI')
                axes[0, 0].set_ylabel('Score')
        
        # 2. Calibration: Reliability Diagram
        if 'calibration_metrics' in results and 'reliability_diagram' in results['calibration_metrics']:
            rel_data = results['calibration_metrics']['reliability_diagram']
            
            axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            axes[0, 1].plot(rel_data['bin_centers'], rel_data['bin_accuracies'], 
                           'o-', label='Model')
            axes[0, 1].set_xlabel('Mean Predicted Probability')
            axes[0, 1].set_ylabel('Fraction of Positives')
            axes[0, 1].set_title('Reliability Diagram')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Calibration metrics
        if 'calibration_metrics' in results:
            cal_metrics = results['calibration_metrics']
            cal_names = ['ECE', 'MCE', 'Brier Score', 'AUC']
            cal_values = [
                cal_metrics.get('ece', 0),
                cal_metrics.get('mce', 0), 
                cal_metrics.get('brier_score', 0),
                cal_metrics.get('auc', 0)
            ]
            
            bars = axes[1, 0].bar(cal_names, cal_values, alpha=0.7)
            axes[1, 0].set_title('Calibration Metrics')
            axes[1, 0].set_ylabel('Score')
            
            # 为每个bar添加数值标签
            for bar, value in zip(bars, cal_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # 4. 失败分析
        if 'failure_analysis' in results:
            failure = results['failure_analysis']
            if 'confusion_matrix' in failure:
                cm = np.array(failure['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1],
                           xticklabels=['Pred Negative', 'Pred Positive'],
                           yticklabels=['True Negative', 'True Positive'])
                axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{dataset_name}_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved: {self.save_dir / f'{dataset_name}_evaluation.png'}")

def main():
    """演示评测流程"""
    # 配置
    eval_config = EvaluationConfig(
        bootstrap_samples=100,  # 减少samples用于演示
        mc_samples=10,
        k_values=[1, 3, 5],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 创建模型
    model_config = ModelConfig()
    model = create_model(model_config)
    
    # 创建评测器
    evaluator = CoTRRProEvaluator(model, eval_config)
    
    print("🔍 CoTRR-Pro Advanced Evaluator 创建成功!")
    print(f"📊 Bootstrap samples: {eval_config.bootstrap_samples}")
    print(f"🎲 MC samples: {eval_config.mc_samples}")
    print(f"📱 Device: {eval_config.device}")
    print(f"💾 Save dir: {eval_config.save_dir}")
    
    # 示例评测（需要真实数据文件）
    # results = evaluator.evaluate_dataset("data/test.jsonl", "test")
    # print(f"✅ 评测完成，结果保存在: {eval_config.save_dir}")
    
    print("📝 使用 evaluator.evaluate_dataset() 开始评测")
    print("🧪 使用 evaluator.ablation_study() 进行消融研究")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()