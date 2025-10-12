#!/usr/bin/env python3
"""
Day 3 Lightweight Pipeline Enhancer
基于Day 3发现的轻量级、实用主义优化方案
"""

import json
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass

# 导入核心模块
import sys
sys.path.append('.')  # 添加当前目录到路径
from src.subject_object import check_subject_object
from src.conflict_penalty import conflict_penalty
from src.dual_score import fuse_dual_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """优化配置"""
    compliance_weight: float = 0.6
    conflict_penalty_alpha: float = 0.25
    description_boost_weight: float = 0.3
    quality_threshold: float = 0.1
    enable_description_enhancement: bool = True
    enable_caching: bool = True

class LightweightPipelineEnhancer:
    """轻量级pipeline增强器 - 实用主义方案"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.cache = {} if self.config.enable_caching else None
        self.stats = {
            'processed_queries': 0,
            'cache_hits': 0,
            'enhancement_applied': 0,
            'avg_processing_time': 0.0
        }
        
        # 预编译正则表达式
        self._compile_patterns()
        
        logger.info("🚀 轻量级Pipeline增强器初始化完成")
        logger.info(f"   Compliance weight: {self.config.compliance_weight}")
        logger.info(f"   Conflict penalty α: {self.config.conflict_penalty_alpha}")
        logger.info(f"   Description boost: {self.config.description_boost_weight}")
    
    def _compile_patterns(self):
        """预编译常用正则表达式"""
        self.patterns = {
            # 玻璃类型
            'glass_types': re.compile(r'\\b(coupe|martini|rocks|old.fashioned|wine|crystal|champagne|highball)\\b', re.I),
            # 颜色词汇
            'colors': re.compile(r'\\b(pink|golden|amber|clear|blue|green|red|purple|yellow|orange|black|white)\\b', re.I),
            # 装饰元素
            'garnishes': re.compile(r'\\b(rose|petal|orange|lime|lemon|mint|berry|fruit|herb|cherry|olive)\\b', re.I),
            # 质量形容词
            'quality_words': re.compile(r'\\b(elegant|crystal|premium|artisanal|craft|fresh|vibrant|beautiful)\\b', re.I),
            # 负面词汇
            'negative_words': re.compile(r'\\b(generic|basic|plain|simple|ordinary|dull)\\b', re.I)
        }
    
    def enhance_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """增强候选列表"""
        start_time = time.time()
        
        # 统计更新
        self.stats['processed_queries'] += 1
        
        # 缓存检查
        cache_key = self._generate_cache_key(query, candidates)
        if self.cache and cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # 执行增强
        enhanced_candidates = []
        for candidate in candidates:
            enhanced = self._enhance_single_candidate(candidate, query)
            enhanced_candidates.append(enhanced)
        
        # 重排序
        enhanced_candidates.sort(key=lambda x: x.get('enhanced_score', 0), reverse=True)
        
        # 缓存结果
        if self.cache:
            self.cache[cache_key] = enhanced_candidates
        
        # 更新统计
        processing_time = time.time() - start_time
        self.stats['avg_processing_time'] = (
            self.stats['avg_processing_time'] * (self.stats['processed_queries'] - 1) + processing_time
        ) / self.stats['processed_queries']
        
        return enhanced_candidates
    
    def _enhance_single_candidate(self, candidate: Dict, query: str) -> Dict:
        """增强单个候选"""
        enhanced = candidate.copy()
        
        # 1. 基础分数
        base_score = candidate.get('score', 0.5)
        
        # 2. 核心模块处理
        regions = self._extract_regions_from_description(candidate)
        compliance_score, compliance_details = check_subject_object(regions=regions)
        penalty_score, penalty_details = conflict_penalty(regions, alpha=self.config.conflict_penalty_alpha)
        
        # 3. 核心分数融合
        core_enhanced_score = fuse_dual_score(
            compliance_score, penalty_score, 
            w_c=self.config.compliance_weight, w_n=0.4
        )
        
        # 4. 描述增强 (新增)
        description_boost = 0.0
        if self.config.enable_description_enhancement:
            description_boost = self._calculate_description_boost(candidate, query)
        
        # 5. 最终分数计算
        final_score = (
            0.5 * base_score +  # 保留50%原始分数
            0.3 * core_enhanced_score +  # 30%核心模块增强
            0.2 * description_boost  # 20%描述增强
        )
        
        # 6. 更新候选信息
        enhanced.update({
            'original_score': base_score,
            'compliance_score': compliance_score,
            'conflict_penalty': penalty_score,
            'core_enhanced_score': core_enhanced_score,
            'description_boost': description_boost,
            'enhanced_score': final_score,
            'score': final_score,  # 更新主分数
            'enhancement_details': {
                'regions_detected': len(regions),
                'compliance_details': compliance_details,
                'penalty_details': penalty_details,
                'description_features': self._get_description_features(candidate, query)
            }
        })
        
        return enhanced
    
    def _extract_regions_from_description(self, candidate: Dict) -> List[Dict]:
        """从描述中提取regions (改进版)"""
        description = candidate.get('alt_description', '').lower()
        regions = []
        
        # 玻璃检测 (改进)
        glass_match = self.patterns['glass_types'].search(description)
        if glass_match:
            glass_type = glass_match.group(1)
            confidence = 0.95 if glass_type in ['coupe', 'martini'] else 0.85
            regions.append({
                'label': 'glass',
                'type': f'{glass_type}_glass',
                'confidence': confidence
            })
        elif any(word in description for word in ['glass', 'cup']):
            regions.append({
                'label': 'glass',
                'type': 'generic_glass',
                'confidence': 0.7
            })
        
        # 颜色检测 (改进)
        color_matches = self.patterns['colors'].findall(description)
        for color in set(color_matches):  # 去重
            confidence = 0.9 if color in ['pink', 'golden', 'amber'] else 0.8
            regions.append({
                'label': f'{color}_liquid',
                'color': color,
                'type': 'cocktail',
                'confidence': confidence
            })
        
        # 装饰检测 (改进)
        garnish_matches = self.patterns['garnishes'].findall(description)
        for garnish in set(garnish_matches):
            garnish_type = 'floral' if garnish in ['rose', 'petal'] else 'fruit'
            confidence = 0.85 if garnish in ['rose', 'orange', 'mint'] else 0.75
            regions.append({
                'label': f'{garnish}_garnish',
                'type': garnish_type,
                'confidence': confidence
            })
        
        return regions
    
    def _calculate_description_boost(self, candidate: Dict, query: str) -> float:
        """计算描述增强分数"""
        description = candidate.get('alt_description', '').lower()
        query_lower = query.lower()
        
        boost = 0.0
        
        # 1. 查询匹配度
        query_words = set(query_lower.split())
        desc_words = set(description.split())
        word_overlap = len(query_words & desc_words)
        if word_overlap > 0:
            boost += 0.3 * (word_overlap / len(query_words))
        
        # 2. 质量词汇加分
        quality_matches = len(self.patterns['quality_words'].findall(description))
        boost += min(0.2, quality_matches * 0.05)
        
        # 3. 负面词汇减分
        negative_matches = len(self.patterns['negative_words'].findall(description))
        boost -= min(0.15, negative_matches * 0.05)
        
        # 4. 特定查询增强
        if 'floral' in query_lower and any(word in description for word in ['rose', 'petal', 'flower']):
            boost += 0.25
        
        if 'whiskey' in query_lower and 'whiskey' in description:
            boost += 0.2
        
        if 'cocktail' in query_lower and 'cocktail' in description:
            boost += 0.1
        
        # 5. 颜色一致性
        query_colors = self.patterns['colors'].findall(query_lower)
        desc_colors = self.patterns['colors'].findall(description)
        if query_colors and desc_colors:
            color_match = len(set(query_colors) & set(desc_colors))
            if color_match > 0:
                boost += 0.2 * color_match
        
        # 限制在合理范围内
        return max(0.0, min(1.0, boost))
    
    def _get_description_features(self, candidate: Dict, query: str) -> Dict:
        """获取描述特征分析 (用于调试和监控)"""
        description = candidate.get('alt_description', '').lower()
        query_lower = query.lower()
        
        return {
            'glass_types': self.patterns['glass_types'].findall(description),
            'colors': self.patterns['colors'].findall(description),
            'garnishes': self.patterns['garnishes'].findall(description),
            'quality_words': self.patterns['quality_words'].findall(description),
            'negative_words': self.patterns['negative_words'].findall(description),
            'query_word_overlap': len(set(query_lower.split()) & set(description.split()))
        }
    
    def _generate_cache_key(self, query: str, candidates: List[Dict]) -> str:
        """生成缓存键"""
        candidate_ids = [c.get('id', str(hash(str(c)))) for c in candidates]
        return f"{hash(query)}_{hash(tuple(candidate_ids))}"
    
    def optimize_parameters(self, test_data: List[Dict], target_metric: str = 'enhanced_score') -> OptimizationConfig:
        """参数优化 (网格搜索)"""
        logger.info("🔍 开始参数优化")
        
        # 参数搜索空间
        param_grid = {
            'compliance_weight': [0.4, 0.5, 0.6, 0.7, 0.8],
            'conflict_penalty_alpha': [0.15, 0.2, 0.25, 0.3, 0.35],
            'description_boost_weight': [0.1, 0.2, 0.3, 0.4]
        }
        
        best_score = -float('inf')
        best_config = None
        
        # 网格搜索
        for comp_weight in param_grid['compliance_weight']:
            for penalty_alpha in param_grid['conflict_penalty_alpha']:
                for desc_weight in param_grid['description_boost_weight']:
                    # 测试配置
                    test_config = OptimizationConfig(
                        compliance_weight=comp_weight,
                        conflict_penalty_alpha=penalty_alpha,
                        description_boost_weight=desc_weight
                    )
                    
                    # 评估配置
                    score = self._evaluate_config(test_config, test_data, target_metric)
                    
                    if score > best_score:
                        best_score = score
                        best_config = test_config
        
        logger.info(f"✅ 参数优化完成，最佳分数: {best_score:.4f}")
        return best_config
    
    def _evaluate_config(self, config: OptimizationConfig, test_data: List[Dict], metric: str) -> float:
        """评估配置性能"""
        # 临时创建测试增强器
        temp_enhancer = LightweightPipelineEnhancer(config)
        
        total_improvement = 0.0
        valid_queries = 0
        
        for item in test_data:
            query = item.get('query', '')
            candidates = item.get('candidates', [])
            
            if len(candidates) < 2:
                continue
            
            # 原始分数
            original_scores = [c.get('score', 0) for c in candidates]
            
            # 增强后分数
            enhanced_candidates = temp_enhancer.enhance_candidates(query, candidates)
            enhanced_scores = [c.get('enhanced_score', 0) for c in enhanced_candidates]
            
            # 计算改善
            improvement = np.mean(enhanced_scores) - np.mean(original_scores)
            total_improvement += improvement
            valid_queries += 1
        
        return total_improvement / valid_queries if valid_queries > 0 else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.stats.copy()
        
        # 计算衍生指标
        if stats['processed_queries'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['processed_queries']
        else:
            stats['cache_hit_rate'] = 0.0
        
        stats['config'] = {
            'compliance_weight': self.config.compliance_weight,
            'conflict_penalty_alpha': self.config.conflict_penalty_alpha,
            'description_boost_weight': self.config.description_boost_weight
        }
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试基本功能
            test_candidate = {
                'id': 'health_check',
                'alt_description': 'Pink cocktail with rose petals in elegant coupe glass',
                'score': 0.75
            }
            
            result = self._enhance_single_candidate(test_candidate, 'pink floral cocktail')
            
            return {
                'status': 'healthy',
                'basic_enhancement': result.get('enhanced_score', 0) > 0,
                'processing_time': self.stats.get('avg_processing_time', 0),
                'cache_enabled': self.cache is not None
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# 性能测试和对比工具
class LightweightTester:
    """轻量级增强器测试工具"""
    
    def __init__(self):
        self.enhancer = LightweightPipelineEnhancer()
        self.results_dir = Path("research/day3_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_performance_test(self, test_data_path: str = "data/input/sample_input.json") -> Dict[str, Any]:
        """运行性能测试"""
        logger.info("🚀 开始轻量级增强器性能测试")
        
        # 加载测试数据
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        
        test_queries = data.get('inspirations', [])
        
        results = []
        total_start_time = time.time()
        
        for item in test_queries:
            query = item.get('query', '')
            original_candidates = item.get('candidates', [])
            
            if len(original_candidates) < 2:
                continue
            
            # 记录原始分数
            original_scores = [c.get('score', 0) for c in original_candidates]
            
            # 增强处理
            start_time = time.time()
            enhanced_candidates = self.enhancer.enhance_candidates(query, original_candidates)
            processing_time = time.time() - start_time
            
            # 记录增强分数
            enhanced_scores = [c.get('enhanced_score', 0) for c in enhanced_candidates]
            
            # 分析结果
            result = {
                'query': query,
                'candidates_count': len(original_candidates),
                'processing_time': processing_time,
                'scores': {
                    'original_mean': np.mean(original_scores),
                    'enhanced_mean': np.mean(enhanced_scores),
                    'improvement': np.mean(enhanced_scores) - np.mean(original_scores)
                },
                'ranking_changed': [c.get('id') for c in original_candidates] != [c.get('id') for c in enhanced_candidates]
            }
            
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        # 生成报告
        report = {
            'test_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'queries_tested': len(results),
                'total_processing_time': total_time
            },
            'performance_summary': {
                'avg_processing_time': np.mean([r['processing_time'] for r in results]),
                'total_improvement': np.mean([r['scores']['improvement'] for r in results]),
                'queries_reranked': sum(1 for r in results if r['ranking_changed']),
                'rerank_rate': sum(1 for r in results if r['ranking_changed']) / len(results) if results else 0
            },
            'detailed_results': results,
            'enhancer_stats': self.enhancer.get_performance_stats(),
            'health_check': self.enhancer.health_check()
        }
        
        # 保存报告
        report_file = self.results_dir / f"lightweight_test_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ 测试完成，报告保存至: {report_file}")
        
        return report

if __name__ == "__main__":
    # 运行轻量级增强器测试
    tester = LightweightTester()
    report = tester.run_performance_test()
    
    # 打印结果
    print("\\n" + "="*60)
    print("🎯 Lightweight Pipeline Enhancer Results")
    print("="*60)
    
    perf = report.get('performance_summary', {})
    print(f"\\n⚡ Performance:")
    print(f"   Avg processing time: {perf.get('avg_processing_time', 0)*1000:.2f}ms")
    print(f"   Total improvement: {perf.get('total_improvement', 0):+.4f}")
    print(f"   Rerank rate: {perf.get('rerank_rate', 0):.1%}")
    print(f"   Queries reranked: {perf.get('queries_reranked', 0)}")
    
    health = report.get('health_check', {})
    print(f"\\n🔍 Health Check:")
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Basic enhancement: {health.get('basic_enhancement', False)}")
    
    stats = report.get('enhancer_stats', {})
    print(f"\\n📊 Statistics:")
    print(f"   Processed queries: {stats.get('processed_queries', 0)}")
    print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
    
    # 判断是否成功
    improvement = perf.get('total_improvement', 0)
    processing_time = perf.get('avg_processing_time', 0)
    
    if improvement > 0.02 and processing_time < 0.01:  # 10ms以内
        print(f"\\n🏆 Result: SUCCESS - 轻量级方案有效！")
        print(f"   ✅ 质量改进: {improvement:+.4f}")
        print(f"   ✅ 性能开销: {processing_time*1000:.1f}ms (acceptable)")
    elif improvement > 0.01:
        print(f"\\n📈 Result: MODERATE SUCCESS - 有改进空间")
        print(f"   📊 质量改进: {improvement:+.4f}")
        print(f"   ⚡ 性能开销: {processing_time*1000:.1f}ms")
    else:
        print(f"\\n❌ Result: NEEDS IMPROVEMENT")
        print(f"   📉 质量改进不足: {improvement:+.4f}")
        print(f"   🔧 需要进一步参数调优")
    
    print(f"\\n📁 详细报告: research/day3_results/")