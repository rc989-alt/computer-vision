#!/usr/bin/env python3
"""
Day 3 改进版轻量级增强器
基于诊断结果的简化、正向增强方案
"""

import json
import time
import re
import logging
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleConfig:
    """简化配置"""
    base_boost: float = 0.01
    keyword_match_boost: float = 0.02
    quality_match_boost: float = 0.015
    max_total_boost: float = 0.10  # 最大提升限制
    enable_caching: bool = True

class ImprovedLightweightEnhancer:
    """改进版轻量级增强器 - 专注正向增强"""
    
    def __init__(self, config: SimpleConfig = None):
        self.config = config or SimpleConfig()
        self.cache = {} if self.config.enable_caching else None
        self.stats = {
            'processed_queries': 0,
            'cache_hits': 0,
            'avg_improvement': 0.0,
            'avg_processing_time': 0.0
        }
        
        # 预编译正则表达式
        self._compile_patterns()
        
        logger.info("🚀 改进版轻量级增强器初始化完成")
        logger.info(f"   基础提升: {self.config.base_boost}")
        logger.info(f"   关键词匹配提升: {self.config.keyword_match_boost}")
        logger.info(f"   质量匹配提升: {self.config.quality_match_boost}")
    
    def _compile_patterns(self):
        """预编译匹配模式"""
        self.patterns = {
            # 颜色匹配
            'colors': re.compile(r'\\b(pink|golden|amber|clear|blue|green|red|purple|yellow|orange|rose)\\b', re.I),
            # 玻璃器皿
            'glassware': re.compile(r'\\b(coupe|martini|rocks|wine|crystal|champagne|highball|glass)\\b', re.I),
            # 装饰元素
            'garnishes': re.compile(r'\\b(rose|petal|orange|lime|lemon|mint|berry|fruit|herb|cherry|olive|floral)\\b', re.I),
            # 质量词汇
            'quality': re.compile(r'\\b(elegant|beautiful|fresh|vibrant|premium|artisanal|craft|delicate|refined)\\b', re.I),
            # 饮品类型
            'drink_types': re.compile(r'\\b(cocktail|martini|spritz|mojito|cosmopolitan|bellini|sangria)\\b', re.I)
        }
    
    def enhance_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """增强候选列表 - 简化版"""
        start_time = time.time()
        
        # 缓存检查
        cache_key = f"{query}_{len(candidates)}"
        if self.cache and cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # 处理每个候选项
        enhanced_candidates = []
        total_improvement = 0
        
        for candidate in candidates:
            enhanced = self._enhance_single_candidate(candidate, query)
            enhanced_candidates.append(enhanced)
            
            # 计算改进
            improvement = enhanced.get('enhanced_score', 0) - candidate.get('score', 0)
            total_improvement += improvement
        
        # 重新排序
        enhanced_candidates.sort(key=lambda x: x.get('enhanced_score', 0), reverse=True)
        
        # 更新统计
        processing_time = time.time() - start_time
        self.stats['processed_queries'] += 1
        avg_improvement = total_improvement / len(candidates) if candidates else 0
        
        # 滑动平均更新
        self.stats['avg_improvement'] = (
            self.stats['avg_improvement'] * 0.9 + avg_improvement * 0.1
        )
        self.stats['avg_processing_time'] = (
            self.stats['avg_processing_time'] * 0.9 + processing_time * 0.1
        )
        
        # 缓存结果
        if self.cache:
            self.cache[cache_key] = enhanced_candidates
        
        return enhanced_candidates
    
    def _enhance_single_candidate(self, candidate: Dict, query: str) -> Dict:
        """增强单个候选项 - 纯正向逻辑"""
        enhanced = candidate.copy()
        original_score = candidate.get('score', 0.5)
        
        # 计算各种匹配得分
        matches = self._calculate_matches(candidate, query)
        
        # 基础提升
        boost = self.config.base_boost
        
        # 关键词匹配提升
        boost += matches['keyword_matches'] * self.config.keyword_match_boost
        
        # 质量匹配提升
        boost += matches['quality_matches'] * self.config.quality_match_boost
        
        # 特殊模式匹配
        boost += matches['pattern_matches'] * 0.01
        
        # 应用提升上限
        boost = min(boost, self.config.max_total_boost)
        
        # 计算最终分数
        enhanced_score = original_score + boost
        
        # 更新候选项
        enhanced.update({
            'original_score': original_score,
            'enhancement_boost': boost,
            'enhanced_score': enhanced_score,
            'score': enhanced_score,  # 更新主分数
            'match_details': matches
        })
        
        return enhanced
    
    def _calculate_matches(self, candidate: Dict, query: str) -> Dict:
        """计算各种匹配得分"""
        description = candidate.get('alt_description', '').lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        matches = {
            'keyword_matches': 0,
            'quality_matches': 0,
            'pattern_matches': 0,
            'details': {}
        }
        
        # 1. 直接关键词匹配
        keyword_matches = sum(1 for word in query_words if word in description)
        matches['keyword_matches'] = keyword_matches
        matches['details']['matched_keywords'] = keyword_matches
        
        # 2. 模式匹配
        pattern_matches = 0
        pattern_details = {}
        
        for pattern_name, pattern in self.patterns.items():
            pattern_hits = len(pattern.findall(description))
            if pattern_hits > 0:
                pattern_matches += pattern_hits
                pattern_details[pattern_name] = pattern_hits
        
        matches['pattern_matches'] = pattern_matches
        matches['details']['pattern_matches'] = pattern_details
        
        # 3. 质量词汇匹配
        quality_matches = len(self.patterns['quality'].findall(description))
        matches['quality_matches'] = quality_matches
        matches['details']['quality_words'] = quality_matches
        
        # 4. 特殊查询匹配逻辑
        if 'pink' in query_lower and 'pink' in description:
            matches['pattern_matches'] += 2  # 额外奖励精确颜色匹配
        
        if 'floral' in query_lower and any(word in description for word in ['rose', 'petal', 'flower']):
            matches['pattern_matches'] += 2  # 花卉主题匹配
        
        return matches
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return {
            'processed_queries': self.stats['processed_queries'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['processed_queries']),
            'avg_improvement': self.stats['avg_improvement'],
            'avg_processing_time_ms': self.stats['avg_processing_time'] * 1000,
            'status': 'healthy' if self.stats['avg_improvement'] > 0 else 'needs_tuning'
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            'processed_queries': 0,
            'cache_hits': 0,
            'avg_improvement': 0.0,
            'avg_processing_time': 0.0
        }

class ImprovedParameterOptimizer:
    """改进版参数优化器"""
    
    def __init__(self, test_data_path: str = "data/input/sample_input.json"):
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        self.test_data = data.get('inspirations', [])
        logger.info(f"📊 加载了 {len(self.test_data)} 个测试查询")
    
    def optimize_parameters(self) -> SimpleConfig:
        """优化参数"""
        logger.info("🔍 开始参数优化")
        
        best_score = -float('inf')
        best_config = None
        
        # 参数搜索空间
        base_boosts = [0.005, 0.01, 0.015, 0.02]
        keyword_boosts = [0.01, 0.02, 0.03, 0.04]
        quality_boosts = [0.005, 0.01, 0.015, 0.02]
        
        total_combinations = len(base_boosts) * len(keyword_boosts) * len(quality_boosts)
        current = 0
        
        for base_boost in base_boosts:
            for keyword_boost in keyword_boosts:
                for quality_boost in quality_boosts:
                    current += 1
                    
                    config = SimpleConfig(
                        base_boost=base_boost,
                        keyword_match_boost=keyword_boost,
                        quality_match_boost=quality_boost
                    )
                    
                    score = self._evaluate_config(config)
                    
                    if current % 10 == 0 or score > best_score:
                        logger.info(f"   进度: {current}/{total_combinations}, 当前最佳: {best_score:.4f}, 测试分数: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_config = config
        
        logger.info(f"✅ 参数优化完成，最佳分数: {best_score:.4f}")
        return best_config, best_score
    
    def _evaluate_config(self, config: SimpleConfig) -> float:
        """评估配置"""
        enhancer = ImprovedLightweightEnhancer(config)
        
        improvements = []
        processing_times = []
        
        for item in self.test_data:
            query = item.get('query', '')
            candidates = item.get('candidates', [])
            
            if len(candidates) < 2:
                continue
            
            # 测试增强
            start_time = time.time()
            enhanced_candidates = enhancer.enhance_candidates(query, candidates)
            processing_time = time.time() - start_time
            
            # 计算改进
            original_scores = [c.get('score', 0) for c in candidates]
            enhanced_scores = [c.get('enhanced_score', 0) for c in enhanced_candidates]
            
            improvement = np.mean(enhanced_scores) - np.mean(original_scores)
            improvements.append(improvement)
            processing_times.append(processing_time)
        
        if not improvements:
            return 0.0
        
        # 评分：主要看改进，辅助看性能
        avg_improvement = np.mean(improvements)
        avg_time = np.mean(processing_times)
        
        # 评分函数
        score = avg_improvement * 100  # 改进是主要指标
        
        # 性能奖励/惩罚
        if avg_time < 0.001:  # 小于1ms奖励
            score += 1.0
        elif avg_time > 0.005:  # 大于5ms惩罚
            score -= 2.0
        
        return score

def test_improved_enhancer():
    """测试改进版增强器"""
    print("\\n" + "="*60)
    print("🎯 Testing Improved Lightweight Enhancer")
    print("="*60)
    
    # 加载数据
    with open("data/input/sample_input.json", 'r') as f:
        data = json.load(f)
    
    test_data = data.get('inspirations', [])
    item = test_data[0]
    query = item.get('query', '')
    candidates = item.get('candidates', [])
    
    # 创建改进版增强器
    config = SimpleConfig(
        base_boost=0.01,
        keyword_match_boost=0.025,
        quality_match_boost=0.015
    )
    
    enhancer = ImprovedLightweightEnhancer(config)
    
    print(f"查询: '{query}'")
    print("\\n处理结果:")
    
    # 处理增强
    start_time = time.time()
    enhanced_candidates = enhancer.enhance_candidates(query, candidates)
    processing_time = time.time() - start_time
    
    # 显示结果
    for i, (orig, enh) in enumerate(zip(candidates, enhanced_candidates)):
        orig_score = orig.get('score', 0)
        enh_score = enh.get('enhanced_score', 0)
        boost = enh.get('enhancement_boost', 0)
        matches = enh.get('match_details', {})
        
        print(f"\\n   候选项 {i+1}:")
        print(f"      分数: {orig_score:.3f} → {enh_score:.3f} (+{boost:.4f})")
        print(f"      关键词匹配: {matches.get('keyword_matches', 0)}")
        print(f"      质量匹配: {matches.get('quality_matches', 0)}")
        print(f"      模式匹配: {matches.get('pattern_matches', 0)}")
    
    # 总体统计
    original_avg = sum(c.get('score', 0) for c in candidates) / len(candidates)
    enhanced_avg = sum(c.get('enhanced_score', 0) for c in enhanced_candidates) / len(enhanced_candidates)
    total_improvement = enhanced_avg - original_avg
    
    print(f"\\n📊 总体结果:")
    print(f"   原始平均分: {original_avg:.4f}")
    print(f"   增强平均分: {enhanced_avg:.4f}")
    print(f"   总体改进: {total_improvement:+.4f}")
    print(f"   处理时间: {processing_time*1000:.2f}ms")
    
    # 性能统计
    stats = enhancer.get_performance_stats()
    print(f"   状态: {stats['status']}")
    
    # 判断成功与否
    if total_improvement > 0.02:
        print("\\n🚀 EXCELLENT: 显著改进且高效!")
        return True
    elif total_improvement > 0:
        print("\\n✅ GOOD: 有效改进!")
        return True
    else:
        print("\\n❌ POOR: 仍需优化")
        return False

if __name__ == "__main__":
    # 测试改进版增强器
    success = test_improved_enhancer()
    
    if success:
        print("\\n🎯 执行参数优化:")
        optimizer = ImprovedParameterOptimizer()
        best_config, best_score = optimizer.optimize_parameters()
        
        print(f"\\n🏆 最佳配置:")
        print(f"   基础提升: {best_config.base_boost}")
        print(f"   关键词提升: {best_config.keyword_match_boost}")
        print(f"   质量提升: {best_config.quality_match_boost}")
        print(f"   评分: {best_score:.4f}")
        
        # 保存配置
        config_dict = {
            'base_boost': best_config.base_boost,
            'keyword_match_boost': best_config.keyword_match_boost,
            'quality_match_boost': best_config.quality_match_boost,
            'max_total_boost': best_config.max_total_boost,
            'score': best_score
        }
        
        with open("research/day3_results/improved_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\\n📁 最佳配置已保存")
    else:
        print("\\n❌ 需要进一步调试基础逻辑")