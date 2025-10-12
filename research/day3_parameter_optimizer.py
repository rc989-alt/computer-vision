#!/usr/bin/env python3
"""
Day 3 Parameter Optimization
对轻量级增强器进行参数调优
"""

import json
import sys
sys.path.append('.')

from research.day3_lightweight_enhancer import LightweightPipelineEnhancer, OptimizationConfig
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, test_data_path: str = "data/input/sample_input.json"):
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        self.test_data = data.get('inspirations', [])
        logger.info(f"📊 加载了 {len(self.test_data)} 个测试查询")
    
    def grid_search(self) -> OptimizationConfig:
        """网格搜索最优参数"""
        logger.info("🔍 开始网格搜索参数优化")
        
        # 参数搜索空间
        param_grid = {
            'compliance_weight': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'conflict_penalty_alpha': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            'description_boost_weight': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        best_score = -float('inf')
        best_config = None
        best_details = None
        
        total_combinations = len(param_grid['compliance_weight']) * len(param_grid['conflict_penalty_alpha']) * len(param_grid['description_boost_weight'])
        current_combination = 0
        
        # 网格搜索
        for comp_weight in param_grid['compliance_weight']:
            for penalty_alpha in param_grid['conflict_penalty_alpha']:
                for desc_weight in param_grid['description_boost_weight']:
                    current_combination += 1
                    
                    # 测试配置
                    test_config = OptimizationConfig(
                        compliance_weight=comp_weight,
                        conflict_penalty_alpha=penalty_alpha,
                        description_boost_weight=desc_weight
                    )
                    
                    # 评估配置
                    score, details = self._evaluate_config(test_config)
                    
                    if current_combination % 10 == 0 or score > best_score:
                        logger.info(f"   进度: {current_combination}/{total_combinations}, 当前最佳: {best_score:.4f}, 测试分数: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_config = test_config
                        best_details = details
        
        logger.info(f"✅ 参数优化完成，最佳分数: {best_score:.4f}")
        logger.info(f"   最佳配置: compliance_weight={best_config.compliance_weight}, penalty_alpha={best_config.conflict_penalty_alpha}, desc_weight={best_config.description_boost_weight}")
        
        return best_config, best_score, best_details
    
    def _evaluate_config(self, config: OptimizationConfig) -> tuple:
        """评估配置性能"""
        enhancer = LightweightPipelineEnhancer(config)
        
        improvements = []
        processing_times = []
        rerank_count = 0
        
        for item in self.test_data:
            query = item.get('query', '')
            candidates = item.get('candidates', [])
            
            if len(candidates) < 2:
                continue
            
            # 原始分数
            original_scores = [c.get('score', 0) for c in candidates]
            original_ids = [c.get('id', f'c{i}') for i, c in enumerate(candidates)]
            
            # 增强处理
            import time
            start_time = time.time()
            enhanced_candidates = enhancer.enhance_candidates(query, candidates)
            processing_time = time.time() - start_time
            
            enhanced_scores = [c.get('enhanced_score', 0) for c in enhanced_candidates]
            enhanced_ids = [c.get('id', f'c{i}') for i, c in enumerate(enhanced_candidates)]
            
            # 计算改善
            improvement = np.mean(enhanced_scores) - np.mean(original_scores)
            improvements.append(improvement)
            processing_times.append(processing_time)
            
            # 检查是否重排序
            if original_ids != enhanced_ids:
                rerank_count += 1
        
        if not improvements:
            return 0.0, {}
        
        # 综合评分
        avg_improvement = np.mean(improvements)
        avg_processing_time = np.mean(processing_times)
        rerank_rate = rerank_count / len(improvements)
        
        # 评分函数：质量改进为主，性能为辅
        score = avg_improvement * 10  # 主要指标
        
        # 性能惩罚
        if avg_processing_time > 0.001:  # 超过1ms惩罚
            score -= (avg_processing_time - 0.001) * 100
        
        # 重排序奖励
        score += rerank_rate * 0.5
        
        details = {
            'avg_improvement': avg_improvement,
            'avg_processing_time': avg_processing_time,
            'rerank_rate': rerank_rate,
            'valid_queries': len(improvements)
        }
        
        return score, details
    
    def test_best_config(self, config: OptimizationConfig) -> dict:
        """使用最佳配置运行完整测试"""
        logger.info("🧪 使用最佳配置运行完整测试")
        
        enhancer = LightweightPipelineEnhancer(config)
        
        results = []
        for item in self.test_data:
            query = item.get('query', '')
            candidates = item.get('candidates', [])
            
            if len(candidates) < 2:
                continue
            
            # 原始状态
            original_scores = [c.get('score', 0) for c in candidates]
            original_order = [c.get('id', f'c{i}') for i, c in enumerate(candidates)]
            
            # 增强处理
            import time
            start_time = time.time()
            enhanced_candidates = enhancer.enhance_candidates(query, candidates)
            processing_time = time.time() - start_time
            
            enhanced_scores = [c.get('enhanced_score', 0) for c in enhanced_candidates]
            enhanced_order = [c.get('id', f'c{i}') for i, c in enumerate(enhanced_candidates)]
            
            result = {
                'query': query,
                'original_scores': original_scores,
                'enhanced_scores': enhanced_scores,
                'original_order': original_order,
                'enhanced_order': enhanced_order,
                'improvement': np.mean(enhanced_scores) - np.mean(original_scores),
                'processing_time': processing_time,
                'ranking_changed': original_order != enhanced_order
            }
            
            results.append(result)
            
            # 详细输出
            logger.info(f"   Query: '{query}'")
            logger.info(f"     原始分数: {original_scores}")
            logger.info(f"     增强分数: {[f'{s:.3f}' for s in enhanced_scores]}")
            logger.info(f"     改进: {result['improvement']:+.4f}")
            logger.info(f"     重排序: {result['ranking_changed']}")
            logger.info(f"     耗时: {processing_time*1000:.2f}ms")
        
        # 汇总统计
        if results:
            summary = {
                'total_queries': len(results),
                'avg_improvement': np.mean([r['improvement'] for r in results]),
                'avg_processing_time': np.mean([r['processing_time'] for r in results]),
                'rerank_rate': np.mean([r['ranking_changed'] for r in results]),
                'positive_improvements': sum(1 for r in results if r['improvement'] > 0),
                'config': {
                    'compliance_weight': config.compliance_weight,
                    'conflict_penalty_alpha': config.conflict_penalty_alpha,
                    'description_boost_weight': config.description_boost_weight
                }
            }
            
            logger.info(f"📊 汇总统计:")
            logger.info(f"   平均改进: {summary['avg_improvement']:+.4f}")
            logger.info(f"   平均耗时: {summary['avg_processing_time']*1000:.2f}ms")
            logger.info(f"   重排序率: {summary['rerank_rate']:.1%}")
            logger.info(f"   正向改进查询: {summary['positive_improvements']}/{summary['total_queries']}")
        
        return {
            'results': results,
            'summary': summary if results else {},
            'enhancer_stats': enhancer.get_performance_stats()
        }

if __name__ == "__main__":
    optimizer = ParameterOptimizer()
    
    # 执行参数优化
    best_config, best_score, best_details = optimizer.grid_search()
    
    print("\\n" + "="*60)
    print("🎯 Parameter Optimization Results")
    print("="*60)
    
    print(f"\\n🏆 Best Configuration:")
    print(f"   Compliance weight: {best_config.compliance_weight}")
    print(f"   Conflict penalty α: {best_config.conflict_penalty_alpha}")  
    print(f"   Description boost: {best_config.description_boost_weight}")
    print(f"   Score: {best_score:.4f}")
    
    if best_details:
        print(f"\\n📊 Best Performance:")
        print(f"   Avg improvement: {best_details['avg_improvement']:+.4f}")
        print(f"   Avg processing time: {best_details['avg_processing_time']*1000:.2f}ms")
        print(f"   Rerank rate: {best_details['rerank_rate']:.1%}")
    
    # 使用最佳配置运行完整测试
    print("\\n" + "-"*40)
    full_test = optimizer.test_best_config(best_config)
    
    summary = full_test.get('summary', {})
    if summary:
        print(f"\\n🎯 Final Verdict:")
        improvement = summary.get('avg_improvement', 0)
        processing_time = summary.get('avg_processing_time', 0)
        
        if improvement > 0.05 and processing_time < 0.002:
            print(f"   🚀 EXCELLENT: 显著改进且高效!")
            print(f"   ✅ 质量改进: {improvement:+.4f}")
            print(f"   ✅ 处理时间: {processing_time*1000:.1f}ms")
        elif improvement > 0.02 and processing_time < 0.005:
            print(f"   ✅ GOOD: 有效改进!")
            print(f"   📈 质量改进: {improvement:+.4f}")
            print(f"   ⚡ 处理时间: {processing_time*1000:.1f}ms")
        elif improvement > 0:
            print(f"   📈 MODERATE: 轻微改进")
            print(f"   📊 质量改进: {improvement:+.4f}")
            print(f"   ⏱️ 处理时间: {processing_time*1000:.1f}ms")
        else:
            print(f"   ❌ POOR: 需要进一步优化")
            print(f"   📉 质量改进: {improvement:+.4f}")
        
        print(f"\\n💡 建议:")
        if improvement > 0.02:
            print(f"   • 可以部署到生产环境测试")
            print(f"   • 建议开启A/B测试验证效果")
        elif improvement > 0:
            print(f"   • 继续优化参数或特征工程")
            print(f"   • 考虑增加更多启发式规则")
        else:
            print(f"   • 重新评估方案可行性")
            print(f"   • 可能需要更复杂的方法")
    
    # 保存最佳配置
    config_file = "research/day3_results/optimized_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            'config': {
                'compliance_weight': best_config.compliance_weight,
                'conflict_penalty_alpha': best_config.conflict_penalty_alpha,
                'description_boost_weight': best_config.description_boost_weight
            },
            'score': best_score,
            'details': best_details,
            'full_test_results': full_test
        }, f, indent=2)
    
    print(f"\\n📁 最佳配置已保存至: {config_file}")