#!/usr/bin/env python3
"""
Day 3 Focused Integration Test: CoTRR-Stable vs Baseline Performance
专注于核心性能比较，避免复杂的pipeline依赖
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# 设置路径
sys.path.append('research/src')

from step5_integration import CoTRRStableStep5Integration, IntegrationConfig
from src.subject_object import check_subject_object
from src.conflict_penalty import conflict_penalty
from src.dual_score import fuse_dual_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedPipelineTest:
    """专注的pipeline性能测试"""
    
    def __init__(self):
        self.results_dir = Path("research/day3_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化CoTRR-Stable
        config = IntegrationConfig(
            model_path="nonexistent.pt",
            calibrator_path="nonexistent.pkl", 
            rollout_percentage=100.0,
            shadow_mode=False,
            top_m=10
        )
        self.cotrr_integration = CoTRRStableStep5Integration(config)
        
        logger.info("🔧 专注pipeline测试器初始化完成")
    
    def load_test_data(self, file_path: str = "data/input/sample_input.json") -> List[Dict]:
        """加载测试数据"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('inspirations', [])
    
    def baseline_processing(self, candidates: List[Dict]) -> List[Dict]:
        """Baseline处理：只使用核心模块增强"""
        enhanced_candidates = []
        
        for candidate in candidates:
            # 提取mock regions
            regions = self._extract_mock_regions(candidate)
            
            # 应用核心模块
            compliance_score, compliance_details = check_subject_object(regions=regions)
            penalty_score, penalty_details = conflict_penalty(regions, alpha=0.25)
            
            # 融合分数
            base_score = candidate.get('score', 0.5)
            enhanced_score = fuse_dual_score(compliance_score, penalty_score, w_c=0.6, w_n=0.4)
            final_score = 0.7 * enhanced_score + 0.3 * base_score
            
            # 创建增强候选
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                'original_score': base_score,
                'compliance_score': compliance_score,
                'conflict_penalty': penalty_score,
                'enhanced_score': enhanced_score,
                'final_score': final_score,
                'score': final_score,
                'method': 'baseline_enhanced'
            })
            
            enhanced_candidates.append(enhanced_candidate)
        
        # 按分数排序
        enhanced_candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return enhanced_candidates
    
    def cotrr_processing(self, candidates: List[Dict], query: str) -> List[Dict]:
        """CoTRR-Stable处理"""
        # 转换为CoTRR格式
        cotrr_candidates = []
        for i, candidate in enumerate(candidates):
            # 生成mock特征（实际中会从CLIP等提取）
            text_features = self._generate_mock_text_features(candidate, query)
            image_features = self._generate_mock_image_features(candidate)
            
            cotrr_candidates.append({
                "candidate_id": candidate.get("id", f"cand_{i}"),
                "text_features": text_features,
                "image_features": image_features,
                "original_score": candidate.get("score", 0.5),
                "metadata": candidate
            })
        
        # CoTRR重排序
        query_data = {"query_id": f"query_{hash(query)}", "query_text": query}
        result = self.cotrr_integration.rerank_candidates(
            query_data, cotrr_candidates, {"return_scores": True}
        )
        
        # 转换回原格式
        if result['metadata']['status'] == 'success':
            enhanced_candidates = []
            for cotrr_cand in result['candidates']:
                original = cotrr_cand.get('metadata', {})
                enhanced = original.copy()
                enhanced.update({
                    'cotrr_score': float(cotrr_cand.get('_cotrr_score', original.get('score', 0.5))),
                    'cotrr_rank': int(cotrr_cand.get('_cotrr_rank', 1)),
                    'original_rank': int(cotrr_cand.get('_original_rank', 1)),
                    'score': float(cotrr_cand.get('_cotrr_score', original.get('score', 0.5))),
                    'method': 'cotrr_enhanced'
                })
                enhanced_candidates.append(enhanced)
            return enhanced_candidates
        else:
            # Fallback to original
            fallback_candidates = []
            for c in candidates:
                fallback_c = c.copy()
                fallback_c['method'] = 'cotrr_fallback'
                fallback_candidates.append(fallback_c)
            return fallback_candidates
    
    def _extract_mock_regions(self, candidate: Dict) -> List[Dict]:
        """提取mock regions（简化版）"""
        regions = []
        description = candidate.get('alt_description', '').lower()
        
        # Glass detection
        if any(glass in description for glass in ['glass', 'coupe', 'martini']):
            regions.append({
                'label': 'glass', 
                'type': 'crystal_glass' if 'crystal' in description else 'glass',
                'confidence': 0.9
            })
        
        # Color detection
        colors = ['pink', 'golden', 'amber', 'clear']
        for color in colors:
            if color in description:
                regions.append({
                    'label': f'{color}_liquid',
                    'color': color,
                    'type': 'cocktail',
                    'confidence': 0.8
                })
        
        # Garnish detection
        garnishes = ['rose', 'petal', 'orange', 'fruit', 'berry']
        for garnish in garnishes:
            if garnish in description:
                garnish_type = 'floral' if garnish in ['rose', 'petal'] else 'fruit'
                regions.append({
                    'label': f'{garnish}_garnish',
                    'type': garnish_type,
                    'confidence': 0.7
                })
        
        return regions
    
    def _generate_mock_text_features(self, candidate: Dict, query: str) -> List[float]:
        """生成mock文本特征（基于描述和查询的相似性）"""
        # 简单的文本相似性模拟
        description = candidate.get('alt_description', '').lower()
        query_words = set(query.lower().split())
        desc_words = set(description.split())
        
        # 基础特征向量
        base_features = np.random.randn(256) * 0.1
        
        # 相似性增强
        similarity = len(query_words & desc_words) / max(len(query_words | desc_words), 1)
        base_features[:50] += similarity * 0.5  # 前50维用于相似性
        
        # 扩展到1024维
        extended_features = np.concatenate([base_features, np.zeros(768)])
        return extended_features.tolist()
    
    def _generate_mock_image_features(self, candidate: Dict) -> List[float]:
        """生成mock图像特征（基于描述的视觉线索）"""
        description = candidate.get('alt_description', '').lower()
        
        # 基础特征向量
        base_features = np.random.randn(256) * 0.1
        
        # 颜色特征
        if 'pink' in description:
            base_features[0:10] += 0.8
        elif 'golden' in description or 'amber' in description:
            base_features[10:20] += 0.8
        
        # 形状特征
        if 'coupe' in description:
            base_features[20:30] += 0.6
        elif 'old fashioned' in description:
            base_features[30:40] += 0.6
        
        # 装饰特征
        if any(garnish in description for garnish in ['rose', 'petal', 'orange', 'fruit']):
            base_features[40:50] += 0.5
        
        # 扩展到1024维
        extended_features = np.concatenate([base_features, np.zeros(768)])
        return extended_features.tolist()
    
    def compare_methods(self, query_item: Dict) -> Dict[str, Any]:
        """比较两种方法的性能"""
        query = query_item.get('query', '')
        original_candidates = query_item.get('candidates', [])
        
        if len(original_candidates) < 2:
            return {
                'query': query,
                'candidates_count': len(original_candidates),
                'skipped': True,
                'reason': 'insufficient_candidates'
            }
        
        logger.info(f"🔍 处理查询: '{query}' ({len(original_candidates)} candidates)")
        
        # Baseline处理
        start_time = time.time()
        baseline_results = self.baseline_processing(original_candidates.copy())
        baseline_time = time.time() - start_time
        
        # CoTRR处理
        start_time = time.time()
        cotrr_results = self.cotrr_processing(original_candidates.copy(), query)
        cotrr_time = time.time() - start_time
        
        # 分析结果
        comparison = {
            'query': query,
            'candidates_count': len(original_candidates),
            'timing': {
                'baseline_time': float(baseline_time),
                'cotrr_time': float(cotrr_time),
                'overhead_ratio': float(cotrr_time / baseline_time if baseline_time > 0 else 999.0)
            },
            'rankings': {
                'original_order': [c.get('id', f'c{i}') for i, c in enumerate(original_candidates)],
                'baseline_order': [c.get('id', f'c{i}') for i, c in enumerate(baseline_results)],
                'cotrr_order': [c.get('id', f'c{i}') for i, c in enumerate(cotrr_results)]
            },
            'scores': {
                'original_scores': [c.get('score', 0) for c in original_candidates],
                'baseline_scores': [c.get('final_score', 0) for c in baseline_results],
                'cotrr_scores': [c.get('score', 0) for c in cotrr_results]
            },
            'quality_metrics': self._calculate_quality_metrics(
                original_candidates, baseline_results, cotrr_results
            )
        }
        
        return comparison
    
    def _calculate_quality_metrics(self, original: List, baseline: List, cotrr: List) -> Dict:
        """计算质量指标"""
        metrics = {}
        
        # 分数统计
        orig_scores = [c.get('score', 0) for c in original]
        base_scores = [c.get('final_score', 0) for c in baseline]
        cotrr_scores = [c.get('score', 0) for c in cotrr]
        
        metrics['score_stats'] = {
            'original_mean': float(np.mean(orig_scores)),
            'baseline_mean': float(np.mean(base_scores)),
            'cotrr_mean': float(np.mean(cotrr_scores)),
            'baseline_improvement': float(np.mean(base_scores) - np.mean(orig_scores)),
            'cotrr_improvement': float(np.mean(cotrr_scores) - np.mean(orig_scores))
        }
        
        # 排序变化
        orig_ids = [c.get('id', f'c{i}') for i, c in enumerate(original)]
        base_ids = [c.get('id', f'c{i}') for i, c in enumerate(baseline)]
        cotrr_ids = [c.get('id', f'c{i}') for i, c in enumerate(cotrr)]
        
        metrics['ranking_changes'] = {
            'baseline_vs_original': bool(orig_ids != base_ids),
            'cotrr_vs_original': bool(orig_ids != cotrr_ids),
            'cotrr_vs_baseline': bool(base_ids != cotrr_ids)
        }
        
        # Top-1准确性（如果有ground truth的话，这里简化处理）
        if len(original) > 1:
            metrics['top1_analysis'] = {
                'original_top1': orig_ids[0] if orig_ids else None,
                'baseline_top1': base_ids[0] if base_ids else None,
                'cotrr_top1': cotrr_ids[0] if cotrr_ids else None
            }
        
        return metrics
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行完整测试"""
        logger.info("🚀 开始Day 3专注集成测试")
        
        # 加载测试数据
        test_queries = self.load_test_data()
        logger.info(f"📊 加载了 {len(test_queries)} 个测试查询")
        
        # 处理每个查询
        results = []
        for query_item in test_queries:
            result = self.compare_methods(query_item)
            results.append(result)
        
        # 汇总统计
        summary = self._generate_summary(results)
        
        # 完整报告
        report = {
            'test_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'queries_tested': len(test_queries),
                'successful_comparisons': len([r for r in results if not r.get('skipped', False)])
            },
            'individual_results': results,
            'summary': summary
        }
        
        # 保存报告
        report_file = self.results_dir / f"focused_comparison_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ 测试完成，报告保存至: {report_file}")
        
        return report
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """生成汇总统计"""
        valid_results = [r for r in results if not r.get('skipped', False)]
        
        if not valid_results:
            return {'status': 'no_valid_results'}
        
        # 时间性能统计
        baseline_times = [r['timing']['baseline_time'] for r in valid_results]
        cotrr_times = [r['timing']['cotrr_time'] for r in valid_results]
        overhead_ratios = [r['timing']['overhead_ratio'] for r in valid_results if r['timing']['overhead_ratio'] != float('inf')]
        
        # 质量改进统计
        baseline_improvements = [r['quality_metrics']['score_stats']['baseline_improvement'] for r in valid_results]
        cotrr_improvements = [r['quality_metrics']['score_stats']['cotrr_improvement'] for r in valid_results]
        
        # 排序变化统计
        cotrr_changes = sum(1 for r in valid_results if r['quality_metrics']['ranking_changes']['cotrr_vs_original'])
        baseline_changes = sum(1 for r in valid_results if r['quality_metrics']['ranking_changes']['baseline_vs_original'])
        
        summary = {
            'performance_summary': {
                'avg_baseline_time': float(np.mean(baseline_times)),
                'avg_cotrr_time': float(np.mean(cotrr_times)),
                'avg_overhead_ratio': float(np.mean(overhead_ratios)) if overhead_ratios else 999.0,
                'overhead_acceptable': bool(np.mean(overhead_ratios) <= 3.0 if overhead_ratios else False)
            },
            'quality_summary': {
                'avg_baseline_improvement': float(np.mean(baseline_improvements)),
                'avg_cotrr_improvement': float(np.mean(cotrr_improvements)),
                'cotrr_vs_baseline': float(np.mean(cotrr_improvements) - np.mean(baseline_improvements)),
                'cotrr_superior': bool(np.mean(cotrr_improvements) > np.mean(baseline_improvements))
            },
            'ranking_summary': {
                'queries_reranked_by_baseline': int(baseline_changes),
                'queries_reranked_by_cotrr': int(cotrr_changes),
                'baseline_rerank_rate': float(baseline_changes / len(valid_results)),
                'cotrr_rerank_rate': float(cotrr_changes / len(valid_results))
            },
            'overall_verdict': str(self._determine_verdict(valid_results))
        }
        
        return summary
    
    def _determine_verdict(self, results: List[Dict]) -> str:
        """确定最终判断"""
        if not results:
            return "INSUFFICIENT_DATA"
        
        cotrr_improvements = [r['quality_metrics']['score_stats']['cotrr_improvement'] for r in results]
        baseline_improvements = [r['quality_metrics']['score_stats']['baseline_improvement'] for r in results]
        overhead_ratios = [r['timing']['overhead_ratio'] for r in results if r['timing']['overhead_ratio'] != float('inf')]
        
        avg_cotrr_improvement = np.mean(cotrr_improvements)
        avg_baseline_improvement = np.mean(baseline_improvements)
        avg_overhead = np.mean(overhead_ratios) if overhead_ratios else float('inf')
        
        if avg_cotrr_improvement > avg_baseline_improvement + 0.05 and avg_overhead <= 2.0:
            return "COTRR_SIGNIFICANTLY_BETTER"
        elif avg_cotrr_improvement > avg_baseline_improvement + 0.02 and avg_overhead <= 3.0:
            return "COTRR_MODERATELY_BETTER"
        elif avg_cotrr_improvement > avg_baseline_improvement and avg_overhead <= 5.0:
            return "COTRR_SLIGHTLY_BETTER"
        elif avg_overhead > 10.0:
            return "COTRR_TOO_SLOW"
        else:
            return "COTRR_NOT_BETTER"

if __name__ == "__main__":
    tester = FocusedPipelineTest()
    report = tester.run_comprehensive_test()
    
    # 打印关键结果
    print("\n" + "="*60)
    print("🎯 Day 3 Focused Pipeline Test Results")
    print("="*60)
    
    summary = report.get('summary', {})
    
    # 性能结果
    perf = summary.get('performance_summary', {})
    print(f"\n⚡ Performance Analysis:")
    print(f"   Baseline avg time: {perf.get('avg_baseline_time', 0):.4f}s")
    print(f"   CoTRR avg time: {perf.get('avg_cotrr_time', 0):.4f}s")
    print(f"   Overhead ratio: {perf.get('avg_overhead_ratio', 0):.2f}x")
    print(f"   Overhead acceptable: {perf.get('overhead_acceptable', False)}")
    
    # 质量结果
    qual = summary.get('quality_summary', {})
    print(f"\n📊 Quality Analysis:")
    print(f"   Baseline improvement: {qual.get('avg_baseline_improvement', 0):+.4f}")
    print(f"   CoTRR improvement: {qual.get('avg_cotrr_improvement', 0):+.4f}")
    print(f"   CoTRR vs Baseline: {qual.get('cotrr_vs_baseline', 0):+.4f}")
    print(f"   CoTRR superior: {qual.get('cotrr_superior', False)}")
    
    # 排序分析
    rank = summary.get('ranking_summary', {})
    print(f"\n🔄 Ranking Analysis:")
    print(f"   Baseline rerank rate: {rank.get('baseline_rerank_rate', 0):.1%}")
    print(f"   CoTRR rerank rate: {rank.get('cotrr_rerank_rate', 0):.1%}")
    
    # 最终判断
    verdict = summary.get('overall_verdict', 'UNKNOWN')
    print(f"\n🏆 Final Verdict: {verdict}")
    
    verdict_explanations = {
        'COTRR_SIGNIFICANTLY_BETTER': '🚀 CoTRR显著优于baseline，建议部署',
        'COTRR_MODERATELY_BETTER': '✅ CoTRR适度优于baseline，可考虑部署',
        'COTRR_SLIGHTLY_BETTER': '📈 CoTRR略优于baseline，需要权衡成本',
        'COTRR_TOO_SLOW': '🐌 CoTRR性能开销过大，需要优化',
        'COTRR_NOT_BETTER': '❌ CoTRR未显示明显优势，需要重新评估'
    }
    
    explanation = verdict_explanations.get(verdict, '❓ 需要进一步分析')
    print(f"   {explanation}")
    
    print(f"\n📁 详细报告: research/day3_results/")