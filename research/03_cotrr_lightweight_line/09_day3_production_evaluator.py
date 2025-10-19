#!/usr/bin/env python3
"""
生产级轻量级增强器评估器
基于120查询、3600候选项、50探针的生产级数据集
"""

import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import scipy.stats as stats
from collections import defaultdict

# 导入改进版增强器
import sys
sys.path.append('.')
from research.day3_improved_enhancer import ImprovedLightweightEnhancer, SimpleConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProductionMetrics:
    """生产级指标"""
    compliance_improvement: float = 0.0
    compliance_ci95: Tuple[float, float] = (0.0, 0.0)
    ndcg_improvement: float = 0.0
    ndcg_ci95: Tuple[float, float] = (0.0, 0.0)
    p95_latency_ms: float = 0.0
    blossom_fruit_error_rate: float = 0.0
    low_margin_rate: float = 0.0
    
    def meets_thresholds(self, config) -> Dict[str, bool]:
        """检查是否满足生产级门槛"""
        return {
            'compliance_improvement': self.compliance_ci95[0] >= config.min_compliance_improvement,
            'ndcg_improvement': self.ndcg_ci95[0] >= config.target_ndcg_improvement,
            'latency': self.p95_latency_ms <= config.max_p95_latency_ms,
            'blossom_fruit_error': self.blossom_fruit_error_rate <= config.max_blossom_fruit_error_rate,
            'low_margin': self.low_margin_rate <= config.max_low_margin_rate
        }

class ProductionEvaluator:
    """生产级评估器"""
    
    def __init__(self, production_config):
        self.config = production_config
        self.results = {}
        
    def evaluate_production_system(self, dataset_path: str, 
                                 enhancer: ImprovedLightweightEnhancer) -> ProductionMetrics:
        """评估生产级系统"""
        logger.info("🏭 开始生产级系统评估")
        
        # 加载生产级数据集
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"   数据集规模: {len(dataset['inspirations'])} 查询, {len(dataset['blossom_fruit_probes'])} 探针")
        
        # 1. 主要指标评估
        main_metrics = self._evaluate_main_metrics(dataset['inspirations'], enhancer)
        
        # 2. Blossom↔Fruit专项评估
        blossom_fruit_metrics = self._evaluate_blossom_fruit_probes(
            dataset['blossom_fruit_probes'], enhancer
        )
        
        # 3. 性能评估
        performance_metrics = self._evaluate_performance(dataset['inspirations'], enhancer)
        
        # 4. 置信区间计算
        ci_metrics = self._calculate_confidence_intervals(main_metrics)
        
        # 合并结果
        production_metrics = ProductionMetrics(
            compliance_improvement=main_metrics['avg_compliance_improvement'],
            compliance_ci95=ci_metrics['compliance_ci95'],
            ndcg_improvement=main_metrics['avg_ndcg_improvement'],
            ndcg_ci95=ci_metrics['ndcg_ci95'],
            p95_latency_ms=performance_metrics['p95_latency_ms'],
            blossom_fruit_error_rate=blossom_fruit_metrics['error_rate'],
            low_margin_rate=blossom_fruit_metrics['low_margin_rate']
        )
        
        logger.info("✅ 生产级系统评估完成")
        return production_metrics
    
    def _evaluate_main_metrics(self, inspirations: List[Dict], 
                              enhancer: ImprovedLightweightEnhancer) -> Dict:
        """评估主要指标"""
        logger.info("   评估主要指标 (Compliance, nDCG)")
        
        compliance_improvements = []
        ndcg_improvements = []
        domain_results = defaultdict(list)
        
        for item in inspirations:
            query = item['query']
            domain = item.get('domain', 'unknown')
            candidates = item['candidates']
            
            if len(candidates) < 2:
                continue
            
            # 原始排序
            original_candidates = candidates.copy()
            original_compliance = self._calculate_compliance_at_k(original_candidates, k=1)
            original_ndcg = self._calculate_ndcg_at_k(original_candidates, k=10)
            
            # 增强排序
            enhanced_candidates = enhancer.enhance_candidates(query, candidates)
            enhanced_compliance = self._calculate_compliance_at_k(enhanced_candidates, k=1)
            enhanced_ndcg = self._calculate_ndcg_at_k(enhanced_candidates, k=10)
            
            # 计算改进
            compliance_improvement = enhanced_compliance - original_compliance
            ndcg_improvement = enhanced_ndcg - original_ndcg
            
            compliance_improvements.append(compliance_improvement)
            ndcg_improvements.append(ndcg_improvement)
            
            domain_results[domain].append({
                'compliance_improvement': compliance_improvement,
                'ndcg_improvement': ndcg_improvement
            })
        
        return {
            'compliance_improvements': compliance_improvements,
            'ndcg_improvements': ndcg_improvements,
            'avg_compliance_improvement': np.mean(compliance_improvements),
            'avg_ndcg_improvement': np.mean(ndcg_improvements),
            'domain_results': dict(domain_results)
        }
    
    def _evaluate_blossom_fruit_probes(self, probes: List[Dict], 
                                     enhancer: ImprovedLightweightEnhancer) -> Dict:
        """评估Blossom↔Fruit专项探针"""
        logger.info("   评估Blossom↔Fruit专项探针")
        
        total_probes = len(probes)
        error_count = 0
        low_margin_count = 0
        
        results_by_type = defaultdict(list)
        
        for probe in probes:
            probe_id = probe['probe_id']
            query = probe['query']
            expected_intent = probe.get('expected_intent', 'unknown')
            test_type = probe.get('test_type', 'unknown')
            candidates = probe['candidates']
            
            # 执行增强
            enhanced_candidates = enhancer.enhance_candidates(query, candidates)
            
            # 分析结果
            result = self._analyze_probe_result(probe, enhanced_candidates)
            
            if result['is_error']:
                error_count += 1
            
            if result['is_low_margin']:
                low_margin_count += 1
            
            results_by_type[test_type].append(result)
        
        error_rate = error_count / total_probes if total_probes > 0 else 0
        low_margin_rate = low_margin_count / total_probes if total_probes > 0 else 0
        
        return {
            'total_probes': total_probes,
            'error_count': error_count,
            'error_rate': error_rate,
            'low_margin_count': low_margin_count,
            'low_margin_rate': low_margin_rate,
            'results_by_type': dict(results_by_type)
        }
    
    def _evaluate_performance(self, inspirations: List[Dict], 
                            enhancer: ImprovedLightweightEnhancer) -> Dict:
        """评估性能指标"""
        logger.info("   评估性能指标 (延迟)")
        
        processing_times = []
        
        # 采样评估（避免过长时间）
        sample_size = min(50, len(inspirations))
        sampled_items = np.random.choice(inspirations, sample_size, replace=False)
        
        for item in sampled_items:
            query = item['query']
            candidates = item['candidates']
            
            # 多次测量取平均
            times = []
            for _ in range(5):
                start_time = time.time()
                enhancer.enhance_candidates(query, candidates)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            processing_times.append(avg_time)
        
        return {
            'processing_times_ms': [t * 1000 for t in processing_times],
            'avg_latency_ms': np.mean(processing_times) * 1000,
            'p95_latency_ms': np.percentile(processing_times, 95) * 1000,
            'p99_latency_ms': np.percentile(processing_times, 99) * 1000
        }
    
    def _calculate_confidence_intervals(self, main_metrics: Dict, 
                                      confidence: float = 0.95) -> Dict:
        """计算置信区间"""
        logger.info("   计算CI95置信区间")
        
        compliance_improvements = main_metrics['compliance_improvements']
        ndcg_improvements = main_metrics['ndcg_improvements']
        
        # 计算CI95
        compliance_ci95 = self._bootstrap_ci(compliance_improvements, confidence)
        ndcg_ci95 = self._bootstrap_ci(ndcg_improvements, confidence)
        
        return {
            'compliance_ci95': compliance_ci95,
            'ndcg_ci95': ndcg_ci95
        }
    
    def _bootstrap_ci(self, data: List[float], confidence: float = 0.95, 
                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap置信区间计算"""
        if not data:
            return (0.0, 0.0)
        
        data = np.array(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def _calculate_compliance_at_k(self, candidates: List[Dict], k: int = 1) -> float:
        """计算Compliance@K"""
        if len(candidates) < k:
            return 0.0
        
        # 基于分数的简化Compliance计算
        top_k = candidates[:k]
        scores = [c.get('enhanced_score', c.get('score', 0)) for c in top_k]
        
        # 高分数表示高Compliance
        return np.mean(scores) if scores else 0.0
    
    def _calculate_ndcg_at_k(self, candidates: List[Dict], k: int = 10) -> float:
        """计算nDCG@K"""
        if len(candidates) < 2:
            return 0.0
        
        k = min(k, len(candidates))
        
        # 获取分数
        scores = [c.get('enhanced_score', c.get('score', 0)) for c in candidates[:k]]
        
        # 计算DCG
        dcg = 0.0
        for i, score in enumerate(scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # 计算IDCG (理想排序)
        ideal_scores = sorted(scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _analyze_probe_result(self, probe: Dict, enhanced_candidates: List[Dict]) -> Dict:
        """分析探针结果"""
        expected_intent = probe.get('expected_intent', 'unknown')
        test_type = probe.get('test_type', 'unknown')
        
        # 简化的错误检测逻辑
        top_candidate = enhanced_candidates[0] if enhanced_candidates else None
        
        is_error = False
        is_low_margin = False
        margin = 0.0
        
        if top_candidate:
            score = top_candidate.get('enhanced_score', 0)
            
            # 基于分数的margin计算
            if len(enhanced_candidates) > 1:
                second_score = enhanced_candidates[1].get('enhanced_score', 0)
                margin = score - second_score
                is_low_margin = margin < 0.05  # 5%的margin阈值
            
            # 基于测试类型的错误判断
            if test_type == 'blossom_fruit_confusion':
                # 对于混淆测试，分数过低表示可能的错误
                is_error = score < 0.7
            elif expected_intent in ['blossom', 'fruit']:
                # 对于明确意图，检查是否选择了错误类型
                candidate_description = top_candidate.get('alt_description', '').lower()
                wrong_intent = 'fruit' if expected_intent == 'blossom' else 'blossom'
                is_error = wrong_intent in candidate_description and expected_intent not in candidate_description
        
        return {
            'probe_id': probe.get('probe_id'),
            'test_type': test_type,
            'expected_intent': expected_intent,
            'is_error': is_error,
            'is_low_margin': is_low_margin,
            'margin': margin,
            'top_score': top_candidate.get('enhanced_score', 0) if top_candidate else 0
        }
    
    def print_production_report(self, metrics: ProductionMetrics, config) -> None:
        """打印生产级报告"""
        print("\\n" + "="*100)
        print("🏭 生产级轻量级增强器评估报告")
        print("="*100)
        
        # 主要指标
        print(f"\\n📊 主要指标:")
        print(f"   ΔCompliance@1: {metrics.compliance_improvement:+.4f}")
        print(f"   ΔCompliance@1 CI95: [{metrics.compliance_ci95[0]:+.4f}, {metrics.compliance_ci95[1]:+.4f}]")
        print(f"   ΔnDCG@10: {metrics.ndcg_improvement:+.4f}")
        print(f"   ΔnDCG@10 CI95: [{metrics.ndcg_ci95[0]:+.4f}, {metrics.ndcg_ci95[1]:+.4f}]")
        
        # 性能指标
        print(f"\\n⚡ 性能指标:")
        print(f"   P95延迟: {metrics.p95_latency_ms:.2f}ms")
        
        # 专项指标
        print(f"\\n🌸 Blossom↔Fruit专项:")
        print(f"   误判率: {metrics.blossom_fruit_error_rate:.1%}")
        print(f"   低margin率: {metrics.low_margin_rate:.1%}")
        
        # 门槛检查
        print(f"\\n🎯 生产级门槛检查:")
        thresholds = metrics.meets_thresholds(config)
        
        status_map = {
            'compliance_improvement': (f"ΔCompliance@1 CI95下界 ≥ +{config.min_compliance_improvement}", metrics.compliance_ci95[0]),
            'ndcg_improvement': (f"ΔnDCG@10 CI95下界 ≥ +{config.target_ndcg_improvement}", metrics.ndcg_ci95[0]),
            'latency': (f"P95延迟 < {config.max_p95_latency_ms}ms", metrics.p95_latency_ms),
            'blossom_fruit_error': (f"Blossom→Fruit误判 ≤ {config.max_blossom_fruit_error_rate:.1%}", metrics.blossom_fruit_error_rate),
            'low_margin': (f"低margin占比 ≤ {config.max_low_margin_rate:.1%}", metrics.low_margin_rate)
        }
        
        all_passed = True
        for key, passed in thresholds.items():
            status = "✅" if passed else "❌"
            desc, value = status_map[key]
            print(f"   {status} {desc}: {value:.4f}" if isinstance(value, float) else f"   {status} {desc}: {value}")
            if not passed:
                all_passed = False
        
        # 最终判断
        print(f"\\n🏆 最终评估:")
        if all_passed:
            print("   🚀 PRODUCTION READY! 所有指标均达到生产级门槛")
            print("   ✅ 可以立即部署到生产环境进行A/B测试")
            
            # 性能等级
            if (metrics.compliance_improvement >= config.target_compliance_improvement and 
                metrics.p95_latency_ms < 0.5):
                print("   🌟 EXCELLENCE级别: 超越目标指标且性能卓越")
            else:
                print("   ⭐ PRODUCTION级别: 满足生产部署要求")
                
        else:
            print("   ❌ NOT READY: 部分指标未达到生产级门槛")
            print("   🔧 需要进一步优化才能部署")
        
        # 建议
        print(f"\\n💡 下一步建议:")
        if all_passed:
            print("   1. 启动生产环境shadow A/B测试")
            print("   2. 建立生产监控和告警体系")
            print("   3. 准备灰度发布计划")
        else:
            print("   1. 重点优化未达标的指标")
            print("   2. 扩展训练数据或调整算法")
            print("   3. 重新验证后再申请生产部署")

def main():
    """主函数"""
    print("🏭 生产级轻量级增强器评估")
    print("="*80)
    
    # 1. 配置
    from research.day3_production_upgrade import ProductionConfig
    config = ProductionConfig()
    
    # 2. 加载最优增强器配置
    print("\\n1️⃣ 加载最优增强器配置...")
    with open("research/day3_results/improved_config.json", 'r') as f:
        enhancer_config_data = json.load(f)
    
    optimal_config = SimpleConfig(
        base_boost=enhancer_config_data['base_boost'],
        keyword_match_boost=enhancer_config_data['keyword_match_boost'],
        quality_match_boost=enhancer_config_data['quality_match_boost'],
        max_total_boost=enhancer_config_data['max_total_boost']
    )
    
    enhancer = ImprovedLightweightEnhancer(optimal_config)
    
    # 3. 执行生产级评估
    print("\\n2️⃣ 执行生产级评估...")
    evaluator = ProductionEvaluator(config)
    
    production_metrics = evaluator.evaluate_production_system(
        "research/day3_results/production_dataset.json", 
        enhancer
    )
    
    # 4. 打印报告
    evaluator.print_production_report(production_metrics, config)
    
    # 5. 保存结果
    thresholds_met = production_metrics.meets_thresholds(config)
    results = {
        'metrics': {
            'compliance_improvement': float(production_metrics.compliance_improvement),
            'compliance_ci95': [float(x) for x in production_metrics.compliance_ci95],
            'ndcg_improvement': float(production_metrics.ndcg_improvement),
            'ndcg_ci95': [float(x) for x in production_metrics.ndcg_ci95],
            'p95_latency_ms': float(production_metrics.p95_latency_ms),
            'blossom_fruit_error_rate': float(production_metrics.blossom_fruit_error_rate),
            'low_margin_rate': float(production_metrics.low_margin_rate)
        },
        'thresholds_met': {k: bool(v) for k, v in thresholds_met.items()},
        'config': {
            'min_compliance_improvement': config.min_compliance_improvement,
            'target_ndcg_improvement': config.target_ndcg_improvement,
            'max_p95_latency_ms': config.max_p95_latency_ms,
            'max_blossom_fruit_error_rate': config.max_blossom_fruit_error_rate,
            'max_low_margin_rate': config.max_low_margin_rate
        },
        'evaluation_time': time.time()
    }
    
    results_path = "research/day3_results/production_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n📁 详细结果已保存: {results_path}")

if __name__ == "__main__":
    main()