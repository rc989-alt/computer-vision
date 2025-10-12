#!/usr/bin/env python3
"""
Day 3 立即强化方案
基于当前轻量级增强器的快速扩展和优化
"""

import json
import time
import logging
from typing import Dict, List, Any
import numpy as np
from pathlib import Path

# 导入改进版增强器
import sys
sys.path.append('.')
from research.day3_improved_enhancer import ImprovedLightweightEnhancer, SimpleConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTestDataGenerator:
    """增强测试数据生成器"""
    
    def __init__(self):
        self.domains = {
            'cocktails': {
                'queries': [
                    'pink floral cocktail', 'golden whiskey cocktail', 'blue martini', 
                    'clear gin fizz', 'amber old fashioned', 'red wine sangria',
                    'purple berry mojito', 'green mint julep'
                ],
                'patterns': ['cocktail', 'martini', 'whiskey', 'gin', 'wine', 'mojito'],
                'colors': ['pink', 'golden', 'blue', 'clear', 'amber', 'red', 'purple', 'green']
            },
            'flowers': {
                'queries': [
                    'pink rose bouquet', 'white lily arrangement', 'red tulip garden',
                    'purple lavender field', 'yellow sunflower bloom', 'blue iris flower'
                ],
                'patterns': ['rose', 'lily', 'tulip', 'lavender', 'sunflower', 'iris'],
                'colors': ['pink', 'white', 'red', 'purple', 'yellow', 'blue']
            },
            'food': {
                'queries': [
                    'golden crispy pastry', 'fresh green salad', 'rich chocolate dessert',
                    'creamy white sauce', 'spicy red curry', 'sweet berry tart'
                ],
                'patterns': ['pastry', 'salad', 'dessert', 'sauce', 'curry', 'tart'],
                'colors': ['golden', 'green', 'chocolate', 'white', 'red', 'berry']
            }
        }
    
    def generate_expanded_test_data(self, queries_per_domain: int = 20, candidates_per_query: int = 5) -> Dict:
        """生成扩展的测试数据"""
        logger.info(f"🔧 生成扩展测试数据: {queries_per_domain} queries/domain, {candidates_per_query} candidates/query")
        
        inspirations = []
        candidate_id = 1
        
        for domain_name, domain_data in self.domains.items():
            logger.info(f"   处理领域: {domain_name}")
            
            # 为每个领域生成查询
            base_queries = domain_data['queries']
            patterns = domain_data['patterns'] 
            colors = domain_data['colors']
            
            # 扩展查询变体
            extended_queries = []
            for base_query in base_queries:
                extended_queries.append(base_query)
                # 添加变体
                for pattern in patterns[:3]:  # 限制变体数量
                    if pattern not in base_query:
                        variant = f"{colors[0]} {pattern}"  # 简单变体
                        extended_queries.append(variant)
            
            # 限制到目标数量
            extended_queries = extended_queries[:queries_per_domain]
            
            for query in extended_queries:
                candidates = self._generate_candidates_for_query(
                    query, domain_name, candidates_per_query, candidate_id
                )
                candidate_id += len(candidates)
                
                inspirations.append({
                    'query': query,
                    'domain': domain_name,
                    'candidates': candidates
                })
        
        logger.info(f"✅ 生成完成: {len(inspirations)} 个查询, 总计 {candidate_id-1} 个候选项")
        
        return {'inspirations': inspirations}
    
    def _generate_candidates_for_query(self, query: str, domain: str, count: int, start_id: int) -> List[Dict]:
        """为查询生成候选项"""
        candidates = []
        query_words = query.lower().split()
        
        for i in range(count):
            # 生成不同质量的候选项
            quality_tier = 'high' if i < count//3 else 'medium' if i < 2*count//3 else 'low'
            
            candidate = self._create_candidate(
                candidate_id=f"gen_{start_id + i:04d}",
                query_words=query_words,
                domain=domain,
                quality_tier=quality_tier
            )
            candidates.append(candidate)
        
        return candidates
    
    def _create_candidate(self, candidate_id: str, query_words: List[str], domain: str, quality_tier: str) -> Dict:
        """创建单个候选项"""
        
        # 基础分数范围
        score_ranges = {
            'high': (0.80, 0.95),
            'medium': (0.60, 0.79), 
            'low': (0.30, 0.59)
        }
        
        base_score = np.random.uniform(*score_ranges[quality_tier])
        
        # 生成描述
        description = self._generate_description(query_words, domain, quality_tier)
        
        # 生成URL (模拟)
        url = f"https://images.unsplash.com/photo-{np.random.randint(1000000000, 9999999999)}"
        
        return {
            'id': candidate_id,
            'regular': url,
            'alt_description': description,
            'score': round(base_score, 3),
            'domain': domain,
            'quality_tier': quality_tier
        }
    
    def _generate_description(self, query_words: List[str], domain: str, quality_tier: str) -> str:
        """生成候选项描述"""
        
        # 领域特定词汇库
        domain_vocab = {
            'cocktails': {
                'objects': ['cocktail', 'drink', 'glass', 'martini', 'whiskey'],
                'qualities': ['elegant', 'crystal', 'premium', 'craft', 'artisanal'],
                'garnishes': ['rose petal', 'orange peel', 'mint leaf', 'cherry', 'olive']
            },
            'flowers': {
                'objects': ['flower', 'bouquet', 'bloom', 'arrangement', 'garden'],
                'qualities': ['beautiful', 'fresh', 'vibrant', 'delicate', 'fragrant'],
                'garnishes': ['petals', 'leaves', 'stems', 'buds', 'thorns']
            },
            'food': {
                'objects': ['dish', 'plate', 'meal', 'recipe', 'cuisine'],
                'qualities': ['delicious', 'fresh', 'gourmet', 'homemade', 'organic'],
                'garnishes': ['herbs', 'sauce', 'garnish', 'seasoning', 'spices']
            }
        }
        
        vocab = domain_vocab.get(domain, domain_vocab['cocktails'])
        
        # 根据质量层级控制描述质量
        if quality_tier == 'high':
            # 高质量: 包含多个查询词汇和高质量描述词
            matched_words = [word for word in query_words if len(word) > 2]
            quality_words = np.random.choice(vocab['qualities'], 2, replace=False)
            object_word = np.random.choice(vocab['objects'])
            garnish = np.random.choice(vocab['garnishes'])
            
            description = f"A {' '.join(quality_words)} {' '.join(matched_words)} {object_word} with {garnish}"
            
        elif quality_tier == 'medium':
            # 中等质量: 部分匹配
            matched_words = query_words[:2] if len(query_words) >= 2 else query_words
            quality_word = np.random.choice(vocab['qualities'])
            object_word = np.random.choice(vocab['objects'])
            
            description = f"A {quality_word} {' '.join(matched_words)} {object_word}"
            
        else:
            # 低质量: 最少匹配
            object_word = np.random.choice(vocab['objects'])
            basic_adj = np.random.choice(['simple', 'basic', 'standard'])
            
            description = f"A {basic_adj} {object_word}"
        
        return description

class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_enhanced_system(self, test_data: Dict, enhancer: ImprovedLightweightEnhancer) -> Dict:
        """评估增强系统"""
        logger.info("🧪 开始综合评估")
        
        results = {
            'overall_metrics': {},
            'domain_metrics': {},
            'quality_tier_metrics': {},
            'performance_metrics': {},
            'detailed_results': []
        }
        
        all_improvements = []
        all_processing_times = []
        domain_results = {}
        quality_tier_results = {}
        
        for item in test_data['inspirations']:
            query = item['query']
            domain = item.get('domain', 'unknown')
            candidates = item['candidates']
            
            if len(candidates) < 2:
                continue
            
            # 执行增强
            start_time = time.time()
            enhanced_candidates = enhancer.enhance_candidates(query, candidates)
            processing_time = time.time() - start_time
            
            # 计算指标
            original_scores = [c.get('score', 0) for c in candidates]
            enhanced_scores = [c.get('enhanced_score', 0) for c in enhanced_candidates]
            
            improvement = np.mean(enhanced_scores) - np.mean(original_scores)
            all_improvements.append(improvement)
            all_processing_times.append(processing_time)
            
            # 按领域统计
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(improvement)
            
            # 按质量层级统计
            for candidate in candidates:
                tier = candidate.get('quality_tier', 'unknown')
                if tier not in quality_tier_results:
                    quality_tier_results[tier] = []
                quality_tier_results[tier].append(improvement)
            
            # 详细结果
            results['detailed_results'].append({
                'query': query,
                'domain': domain,
                'improvement': improvement,
                'processing_time': processing_time,
                'original_avg': np.mean(original_scores),
                'enhanced_avg': np.mean(enhanced_scores)
            })
        
        # 计算总体指标
        results['overall_metrics'] = {
            'total_queries': len(all_improvements),
            'avg_improvement': np.mean(all_improvements),
            'std_improvement': np.std(all_improvements),
            'median_improvement': np.median(all_improvements),
            'positive_improvement_rate': np.mean([x > 0 for x in all_improvements]),
            'avg_processing_time_ms': np.mean(all_processing_times) * 1000,
            'p95_processing_time_ms': np.percentile(all_processing_times, 95) * 1000
        }
        
        # 按领域统计
        for domain, improvements in domain_results.items():
            results['domain_metrics'][domain] = {
                'queries': len(improvements),
                'avg_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements)
            }
        
        # 按质量层级统计
        for tier, improvements in quality_tier_results.items():
            results['quality_tier_metrics'][tier] = {
                'samples': len(improvements),
                'avg_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements)
            }
        
        logger.info("✅ 综合评估完成")
        return results
    
    def print_evaluation_report(self, results: Dict):
        """打印评估报告"""
        print("\n" + "="*80)
        print("🎯 Day 3 增强系统综合评估报告")
        print("="*80)
        
        # 总体指标
        overall = results['overall_metrics']
        print(f"\n📊 总体性能:")
        print(f"   测试查询数: {overall['total_queries']}")
        print(f"   平均改进: {overall['avg_improvement']:+.4f} ({overall['avg_improvement']*100:+.2f}%)")
        print(f"   改进标准差: {overall['std_improvement']:.4f}")
        print(f"   中位数改进: {overall['median_improvement']:+.4f}")
        print(f"   正向改进率: {overall['positive_improvement_rate']:.1%}")
        print(f"   平均处理时间: {overall['avg_processing_time_ms']:.2f}ms")
        print(f"   P95处理时间: {overall['p95_processing_time_ms']:.2f}ms")
        
        # 领域分析
        print(f"\n🎨 按领域分析:")
        for domain, metrics in results['domain_metrics'].items():
            print(f"   {domain}: {metrics['avg_improvement']:+.4f} ± {metrics['std_improvement']:.4f} ({metrics['queries']} queries)")
        
        # 质量层级分析
        print(f"\n📈 按质量层级分析:")
        for tier, metrics in results['quality_tier_metrics'].items():
            print(f"   {tier}: {metrics['avg_improvement']:+.4f} ± {metrics['std_improvement']:.4f} ({metrics['samples']} samples)")
        
        # 性能评级
        print(f"\n🏆 性能评级:")
        avg_improvement = overall['avg_improvement']
        processing_time = overall['avg_processing_time_ms']
        
        if avg_improvement > 0.05 and processing_time < 1.0:
            print("   🚀 EXCELLENT: 显著改进且高效!")
        elif avg_improvement > 0.02 and processing_time < 2.0:
            print("   ✅ GOOD: 有效改进且性能良好")
        elif avg_improvement > 0:
            print("   📈 MODERATE: 轻微改进")
        else:
            print("   ❌ POOR: 需要进一步优化")

def main():
    """主函数"""
    print("🚀 Day 3 立即强化方案执行")
    print("="*60)
    
    # 1. 生成扩展测试数据
    print("\n1️⃣ 生成扩展测试数据...")
    generator = EnhancedTestDataGenerator()
    expanded_data = generator.generate_expanded_test_data(
        queries_per_domain=15,  # 每个领域15个查询
        candidates_per_query=5   # 每个查询5个候选项
    )
    
    # 保存扩展数据
    expanded_data_path = "research/day3_results/expanded_test_data.json"
    with open(expanded_data_path, 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print(f"   ✅ 扩展数据已保存: {expanded_data_path}")
    
    # 2. 加载最优配置
    print("\n2️⃣ 加载最优配置...")
    with open("research/day3_results/improved_config.json", 'r') as f:
        config_data = json.load(f)
    
    optimal_config = SimpleConfig(
        base_boost=config_data['base_boost'],
        keyword_match_boost=config_data['keyword_match_boost'],
        quality_match_boost=config_data['quality_match_boost'],
        max_total_boost=config_data['max_total_boost']
    )
    
    # 3. 创建增强器并评估
    print("\n3️⃣ 执行综合评估...")
    enhancer = ImprovedLightweightEnhancer(optimal_config)
    evaluator = ComprehensiveEvaluator()
    
    results = evaluator.evaluate_enhanced_system(expanded_data, enhancer)
    
    # 4. 打印报告
    evaluator.print_evaluation_report(results)
    
    # 5. 保存详细结果
    results_path = "research/day3_results/comprehensive_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 详细结果已保存: {results_path}")
    
    # 6. 对比分析
    print("\n" + "="*60)
    print("📋 与目标指标对比:")
    
    avg_improvement = results['overall_metrics']['avg_improvement']
    processing_time = results['overall_metrics']['avg_processing_time_ms']
    
    print(f"   当前改进: {avg_improvement:+.4f} vs 目标: +0.04-0.06")
    print(f"   处理时间: {processing_time:.2f}ms vs 目标: <1ms")
    
    if avg_improvement >= 0.04:
        print("   🎯 质量目标: ✅ 已达标")
    else:
        print("   🎯 质量目标: ❌ 需进一步优化")
    
    if processing_time < 1.0:
        print("   ⚡ 性能目标: ✅ 已达标")
    else:
        print("   ⚡ 性能目标: ❌ 需优化")
    
    # 7. 下一步建议
    print(f"\n💡 下一步建议:")
    if avg_improvement >= 0.04 and processing_time < 1.0:
        print("   🚀 可以进入生产环境A/B测试")
        print("   📈 建议扩展到更大规模数据集验证")
    elif avg_improvement >= 0.02:
        print("   🔧 继续特征工程优化")
        print("   📊 分析不同领域的表现差异")
    else:
        print("   🤔 考虑混合策略或算法调整")
        print("   🔍 深入分析失败案例")

if __name__ == "__main__":
    main()