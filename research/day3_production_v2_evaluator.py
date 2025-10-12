#!/usr/bin/env python3
"""
生产级评估器 - V2适配版本
用于评估生产级轻量增强器V2.0
"""

import json
import time
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from day3_production_evaluator import ProductionEvaluator, ProductionMetrics
from day3_production_enhancer_v2 import ProductionLightweightEnhancerV2, AdvancedConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数 - 评估V2增强器"""
    print("🏭 生产级轻量增强器V2.0完整评估")
    print("="*80)
    
    # 1. 加载V2最优配置
    print("\\n1️⃣ 加载V2最优配置...")
    with open("research/day3_results/production_v2_config.json", 'r') as f:
        v2_config_data = json.load(f)
    
    v2_config = AdvancedConfig(
        base_boost=v2_config_data['base_boost'],
        exact_match_boost=v2_config_data['exact_match_boost'],
        fuzzy_match_boost=v2_config_data['fuzzy_match_boost'],
        semantic_boost=v2_config_data['semantic_boost'],
        premium_quality_boost=v2_config_data['premium_quality_boost'],
        high_engagement_boost=v2_config_data['high_engagement_boost'],
        domain_adaptation_factor=v2_config_data['domain_adaptation_factor'],
        confidence_threshold=v2_config_data['confidence_threshold'],
        low_confidence_penalty=v2_config_data['low_confidence_penalty'],
        decision_sharpening=v2_config_data['decision_sharpening'],
        margin_amplification=v2_config_data['margin_amplification'],
        max_total_boost=v2_config_data['max_total_boost'],
        min_score_threshold=v2_config_data['min_score_threshold']
    )
    
    # 2. 创建V2增强器
    print("\\n2️⃣ 创建V2增强器...")
    enhancer_v2 = ProductionLightweightEnhancerV2(v2_config)
    
    # 3. 执行完整生产级评估
    print("\\n3️⃣ 执行完整生产级评估...")
    from day3_production_upgrade import ProductionConfig
    production_config = ProductionConfig()
    
    evaluator = ProductionEvaluator(production_config)
    
    production_metrics = evaluator.evaluate_production_system(
        "research/day3_results/production_dataset.json", 
        enhancer_v2
    )
    
    # 4. 打印V2报告
    print("\\n" + "="*100)
    print("🏭 生产级轻量增强器V2.0评估报告")
    print("="*100)
    
    # 主要指标
    print(f"\\n📊 V2.0主要指标:")
    print(f"   ΔCompliance@1: {production_metrics.compliance_improvement:+.4f}")
    print(f"   ΔCompliance@1 CI95: [{production_metrics.compliance_ci95[0]:+.4f}, {production_metrics.compliance_ci95[1]:+.4f}]")
    print(f"   ΔnDCG@10: {production_metrics.ndcg_improvement:+.4f}")
    print(f"   ΔnDCG@10 CI95: [{production_metrics.ndcg_ci95[0]:+.4f}, {production_metrics.ndcg_ci95[1]:+.4f}]")
    
    # 性能指标
    print(f"\\n⚡ V2.0性能指标:")
    print(f"   P95延迟: {production_metrics.p95_latency_ms:.2f}ms")
    
    # 专项指标
    print(f"\\n🌸 V2.0 Blossom↔Fruit专项:")
    print(f"   误判率: {production_metrics.blossom_fruit_error_rate:.1%}")
    print(f"   低margin率: {production_metrics.low_margin_rate:.1%}")
    
    # V1 vs V2 对比
    print(f"\\n🆚 V1 vs V2 对比:")
    
    # 加载V1结果进行对比
    try:
        with open("research/day3_results/production_evaluation.json", 'r') as f:
            v1_results = json.load(f)
        
        v1_compliance = v1_results['metrics']['compliance_improvement']
        v1_ndcg = v1_results['metrics']['ndcg_improvement']
        v1_latency = v1_results['metrics']['p95_latency_ms']
        v1_margin_rate = v1_results['metrics']['low_margin_rate']
        
        compliance_improvement = production_metrics.compliance_improvement - v1_compliance
        ndcg_improvement = production_metrics.ndcg_improvement - v1_ndcg
        latency_change = production_metrics.p95_latency_ms - v1_latency
        margin_improvement = v1_margin_rate - production_metrics.low_margin_rate
        
        print(f"   ΔCompliance@1改进: {compliance_improvement:+.4f} ({compliance_improvement/v1_compliance*100:+.1f}%)")
        print(f"   ΔnDCG@10改进: {ndcg_improvement:+.4f} ({ndcg_improvement/v1_ndcg*100:+.1f}%)")
        print(f"   P95延迟变化: {latency_change:+.3f}ms")
        print(f"   低margin率改进: {margin_improvement:+.3f} ({margin_improvement/v1_margin_rate*100:+.1f}%)")
        
    except Exception as e:
        print(f"   ⚠️ 无法加载V1结果进行对比: {e}")
    
    # 门槛检查
    print(f"\\n🎯 V2.0生产级门槛检查:")
    thresholds = production_metrics.meets_thresholds(production_config)
    
    status_map = {
        'compliance_improvement': (f"ΔCompliance@1 CI95下界 ≥ +{production_config.min_compliance_improvement}", production_metrics.compliance_ci95[0]),
        'ndcg_improvement': (f"ΔnDCG@10 CI95下界 ≥ +{production_config.target_ndcg_improvement}", production_metrics.ndcg_ci95[0]),
        'latency': (f"P95延迟 < {production_config.max_p95_latency_ms}ms", production_metrics.p95_latency_ms),
        'blossom_fruit_error': (f"Blossom→Fruit误判 ≤ {production_config.max_blossom_fruit_error_rate:.1%}", production_metrics.blossom_fruit_error_rate),
        'low_margin': (f"低margin占比 ≤ {production_config.max_low_margin_rate:.1%}", production_metrics.low_margin_rate)
    }
    
    all_passed = True
    for key, passed in thresholds.items():
        status = "✅" if passed else "❌"
        desc, value = status_map[key]
        if key in ['compliance_improvement', 'ndcg_improvement', 'blossom_fruit_error', 'low_margin']:
            print(f"   {status} {desc}: {value:.4f}")
        else:
            print(f"   {status} {desc}: {value:.3f}")
        if not passed:
            all_passed = False
    
    # 最终判断
    print(f"\\n🏆 V2.0最终评估:")
    if all_passed:
        print("   🚀 PRODUCTION READY! V2.0所有指标均达到生产级门槛")
        print("   ✅ 可以立即部署到生产环境进行A/B测试")
        
        # 性能等级
        if (production_metrics.compliance_improvement >= production_config.target_compliance_improvement and 
            production_metrics.p95_latency_ms < 0.5):
            print("   🌟 EXCELLENCE级别: 超越目标指标且性能卓越")
        else:
            print("   ⭐ PRODUCTION级别: 满足生产部署要求")
            
    else:
        print("   ❌ NOT READY: V2.0部分指标仍未达到生产级门槛")
        failed_metrics = [key for key, passed in thresholds.items() if not passed]
        print(f"   🔧 待优化指标: {', '.join(failed_metrics)}")
    
    # 技术改进总结
    print(f"\\n💡 V2.0技术改进总结:")
    print("   ✨ 多层级增强逻辑 (精确+模糊+语义)")
    print("   ✨ 领域自适应机制")
    print("   ✨ 动态权重调整")
    print("   ✨ 决策锐化和margin放大")
    print("   ✨ 网格搜索参数优化")
    
    # 保存V2结果
    v2_results = {
        'version': '2.0',
        'metrics': {
            'compliance_improvement': float(production_metrics.compliance_improvement),
            'compliance_ci95': [float(x) for x in production_metrics.compliance_ci95],
            'ndcg_improvement': float(production_metrics.ndcg_improvement),
            'ndcg_ci95': [float(x) for x in production_metrics.ndcg_ci95],
            'p95_latency_ms': float(production_metrics.p95_latency_ms),
            'blossom_fruit_error_rate': float(production_metrics.blossom_fruit_error_rate),
            'low_margin_rate': float(production_metrics.low_margin_rate)
        },
        'thresholds_met': {k: bool(v) for k, v in thresholds.items()},
        'config': {
            'min_compliance_improvement': production_config.min_compliance_improvement,
            'target_ndcg_improvement': production_config.target_ndcg_improvement,
            'max_p95_latency_ms': production_config.max_p95_latency_ms,
            'max_blossom_fruit_error_rate': production_config.max_blossom_fruit_error_rate,
            'max_low_margin_rate': production_config.max_low_margin_rate
        },
        'optimization_score': v2_config_data.get('optimization_score', 0),
        'evaluation_time': time.time()
    }
    
    results_path = "research/day3_results/production_v2_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(v2_results, f, indent=2)
    
    print(f"\\n📁 V2.0详细结果已保存: {results_path}")

if __name__ == "__main__":
    main()