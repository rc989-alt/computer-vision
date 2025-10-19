#!/usr/bin/env python3
"""
Day 3: CoTRR-Stable Pipeline Integration & Performance Test
对比baseline pipeline vs CoTRR-Stable系统的真实性能
"""

import json
import time
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime

# Import CoTRR-Stable components
COTRR_AVAILABLE = False
try:
    import sys
    sys.path.append('research/src')
    from step5_integration import CoTRRStableStep5Integration, IntegrationConfig
    from isotonic_calibration import IsotonicCalibrator
    from cotrr_stable import StableCrossAttnReranker, StableConfig
    COTRR_AVAILABLE = True
    print("✅ CoTRR-Stable components loaded successfully")
except ImportError as e:
    print(f"Warning: CoTRR-Stable components not available: {e}")
    COTRR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineComparator:
    """Pipeline性能对比工具"""
    
    def __init__(self, test_data_path: str = "data/input/sample_input.json"):
        self.test_data_path = test_data_path
        self.results_dir = Path("research/day3_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化CoTRR-Stable集成器
        if COTRR_AVAILABLE:
            config = IntegrationConfig(
                model_path="research/models/cotrr_stable.pt",  # 不存在，将使用默认
                calibrator_path="research/models/calibrator.pkl",  # 不存在，将使用原始分数
                rollout_percentage=100.0,
                shadow_mode=False,
                top_m=20
            )
            self.cotrr_integration = CoTRRStableStep5Integration(config)
        else:
            self.cotrr_integration = None
        
        logger.info(f"🔧 Pipeline比较器初始化完成")
        logger.info(f"📂 测试数据: {test_data_path}")
        logger.info(f"📁 结果目录: {self.results_dir}")
    
    def run_baseline_pipeline(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """运行baseline pipeline"""
        logger.info("🚀 运行Baseline Pipeline")
        
        start_time = time.time()
        
        try:
            # 运行现有pipeline (baseline模式)
            result = subprocess.run([
                sys.executable, "pipeline.py",
                "--config", "config/default.json",
                "--input", input_file,
                "--output", output_file,
                "--mode", "baseline"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Baseline pipeline失败: {result.stderr}")
                return {"success": False, "error": result.stderr}
            
            # 读取结果
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            end_time = time.time()
            
            return {
                "success": True,
                "execution_time": end_time - start_time,
                "results": results,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Baseline pipeline超时")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Baseline pipeline执行异常: {e}")
            return {"success": False, "error": str(e)}
    
    def run_cotrr_pipeline(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """运行CoTRR-Stable pipeline"""
        logger.info("🧠 运行CoTRR-Stable Pipeline")
        
        if not COTRR_AVAILABLE:
            return {"success": False, "error": "CoTRR-Stable不可用"}
        
        start_time = time.time()
        
        try:
            # 读取输入数据
            with open(input_file, 'r') as f:
                input_data = json.load(f)
            
            enhanced_results = []
            
            # 处理每个query
            for item in input_data.get('inspirations', []):
                query = item.get('query', '')
                candidates = item.get('candidates', [])
                
                if not candidates:
                    enhanced_results.append(item)
                    continue
                
                # 转换为CoTRR-Stable期望的格式
                cotrr_candidates = []
                for candidate in candidates:
                    # 提取特征 (mock data for now)
                    text_features = np.random.randn(256).tolist()  # Mock text features
                    image_features = np.random.randn(256).tolist()  # Mock image features
                    
                    cotrr_candidates.append({
                        "candidate_id": candidate.get("id", "unknown"),
                        "text_features": text_features,
                        "image_features": image_features,
                        "original_score": candidate.get("score", 0.5),
                        "metadata": {
                            "alt_description": candidate.get("alt_description", ""),
                            "regular": candidate.get("regular", "")
                        }
                    })
                
                # 使用CoTRR-Stable重排序
                query_data = {"query_id": f"query_{len(enhanced_results)}", "query_text": query}
                rerank_result = self.cotrr_integration.rerank_candidates(
                    query_data, cotrr_candidates, {"return_scores": True}
                )
                
                # 转换回原格式
                if rerank_result['metadata']['status'] == 'success':
                    reranked_candidates = []
                    for i, candidate in enumerate(rerank_result['candidates']):
                        original_candidate = next(
                            (c for c in candidates if c.get('id') == candidate.get('candidate_id')),
                            {}
                        )
                        
                        enhanced_candidate = original_candidate.copy()
                        enhanced_candidate.update({
                            'score': candidate.get('_cotrr_score', original_candidate.get('score', 0.5)),
                            'cotrr_rank': candidate.get('_cotrr_rank', i + 1),
                            'original_rank': candidate.get('_original_rank', i + 1),
                            'enhanced_with_cotrr': True
                        })
                        reranked_candidates.append(enhanced_candidate)
                    
                    enhanced_item = item.copy()
                    enhanced_item['candidates'] = reranked_candidates
                    enhanced_item['cotrr_metadata'] = {
                        'inference_time': rerank_result['metadata'].get('inference_time', 0),
                        'strategy': rerank_result['metadata'].get('strategy', 'unknown')
                    }
                    enhanced_results.append(enhanced_item)
                else:
                    # Fallback to original
                    enhanced_results.append(item)
            
            # 保存结果
            output_data = {
                "inspirations": enhanced_results,
                "metadata": {
                    "enhanced_with_cotrr": True,
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            end_time = time.time()
            
            return {
                "success": True,
                "execution_time": end_time - start_time,
                "results": output_data,
                "cotrr_stats": self.cotrr_integration.get_performance_stats()
            }
            
        except Exception as e:
            logger.error(f"CoTRR pipeline执行异常: {e}")
            return {"success": False, "error": str(e)}
    
    def compare_results(self, baseline_results: Dict, cotrr_results: Dict) -> Dict[str, Any]:
        """对比两个pipeline的结果"""
        logger.info("📊 对比pipeline结果")
        
        comparison = {
            "execution_time_comparison": {
                "baseline_time": baseline_results.get("execution_time", 0),
                "cotrr_time": cotrr_results.get("execution_time", 0),
                "overhead_ratio": 0,
                "overhead_absolute": 0
            },
            "quality_metrics": {},
            "reranking_analysis": {},
            "error_analysis": {}
        }
        
        # 计算时间开销
        baseline_time = baseline_results.get("execution_time", 0)
        cotrr_time = cotrr_results.get("execution_time", 0)
        
        if baseline_time > 0:
            comparison["execution_time_comparison"]["overhead_ratio"] = cotrr_time / baseline_time
            comparison["execution_time_comparison"]["overhead_absolute"] = cotrr_time - baseline_time
        
        # 分析成功率
        comparison["success_rates"] = {
            "baseline_success": baseline_results.get("success", False),
            "cotrr_success": cotrr_results.get("success", False)
        }
        
        # 如果两个都成功，进行详细对比
        if baseline_results.get("success") and cotrr_results.get("success"):
            baseline_data = baseline_results.get("results", {})
            cotrr_data = cotrr_results.get("results", {})
            
            # 分析排序变化
            reranking_stats = self._analyze_reranking_changes(
                baseline_data.get("inspirations", []),
                cotrr_data.get("inspirations", [])
            )
            comparison["reranking_analysis"] = reranking_stats
            
            # 分析质量指标
            quality_stats = self._analyze_quality_metrics(
                baseline_data.get("inspirations", []),
                cotrr_data.get("inspirations", [])
            )
            comparison["quality_metrics"] = quality_stats
        
        return comparison
    
    def _analyze_reranking_changes(self, baseline_items: List, cotrr_items: List) -> Dict:
        """分析重排序变化"""
        stats = {
            "total_queries": len(baseline_items),
            "queries_reranked": 0,
            "average_rank_changes": [],
            "top1_changes": 0,
            "score_improvements": []
        }
        
        for i, (baseline_item, cotrr_item) in enumerate(zip(baseline_items, cotrr_items)):
            baseline_candidates = baseline_item.get("candidates", [])
            cotrr_candidates = cotrr_item.get("candidates", [])
            
            if len(baseline_candidates) != len(cotrr_candidates):
                continue
            
            # 检查Top-1是否改变
            if len(baseline_candidates) > 0 and len(cotrr_candidates) > 0:
                baseline_top1 = baseline_candidates[0].get("id")
                cotrr_top1 = cotrr_candidates[0].get("id")
                
                if baseline_top1 != cotrr_top1:
                    stats["top1_changes"] += 1
                    stats["queries_reranked"] += 1
            
            # 计算分数改进
            for baseline_cand, cotrr_cand in zip(baseline_candidates, cotrr_candidates):
                if baseline_cand.get("id") == cotrr_cand.get("id"):
                    baseline_score = baseline_cand.get("score", 0)
                    cotrr_score = cotrr_cand.get("score", 0)
                    improvement = cotrr_score - baseline_score
                    stats["score_improvements"].append(improvement)
        
        # 计算统计量
        if stats["score_improvements"]:
            stats["average_score_improvement"] = np.mean(stats["score_improvements"])
            stats["score_improvement_std"] = np.std(stats["score_improvements"])
        
        return stats
    
    def _analyze_quality_metrics(self, baseline_items: List, cotrr_items: List) -> Dict:
        """分析质量指标"""
        metrics = {
            "average_scores": {},
            "score_distributions": {},
            "quality_indicators": {}
        }
        
        baseline_scores = []
        cotrr_scores = []
        
        for baseline_item, cotrr_item in zip(baseline_items, cotrr_items):
            for baseline_cand in baseline_item.get("candidates", []):
                baseline_scores.append(baseline_cand.get("score", 0))
            
            for cotrr_cand in cotrr_item.get("candidates", []):
                cotrr_scores.append(cotrr_cand.get("score", 0))
        
        if baseline_scores and cotrr_scores:
            metrics["average_scores"] = {
                "baseline_avg": np.mean(baseline_scores),
                "cotrr_avg": np.mean(cotrr_scores),
                "improvement": np.mean(cotrr_scores) - np.mean(baseline_scores)
            }
            
            metrics["score_distributions"] = {
                "baseline_std": np.std(baseline_scores),
                "cotrr_std": np.std(cotrr_scores),
                "baseline_range": [np.min(baseline_scores), np.max(baseline_scores)],
                "cotrr_range": [np.min(cotrr_scores), np.max(cotrr_scores)]
            }
        
        return metrics
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行完整的对比测试"""
        logger.info("🧪 开始Day 3综合对比测试")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 文件路径
        baseline_output = self.results_dir / f"baseline_results_{timestamp}.json"
        cotrr_output = self.results_dir / f"cotrr_results_{timestamp}.json"
        comparison_output = self.results_dir / f"comparison_report_{timestamp}.json"
        
        # 运行baseline
        logger.info("1️⃣ 运行Baseline Pipeline")
        baseline_results = self.run_baseline_pipeline(self.test_data_path, str(baseline_output))
        
        # 运行CoTRR-Stable
        logger.info("2️⃣ 运行CoTRR-Stable Pipeline")
        cotrr_results = self.run_cotrr_pipeline(self.test_data_path, str(cotrr_output))
        
        # 对比结果
        logger.info("3️⃣ 分析和对比结果")
        comparison = self.compare_results(baseline_results, cotrr_results)
        
        # 生成完整报告
        full_report = {
            "test_metadata": {
                "timestamp": timestamp,
                "test_data": self.test_data_path,
                "cotrr_available": COTRR_AVAILABLE
            },
            "baseline_results": baseline_results,
            "cotrr_results": cotrr_results,
            "comparison": comparison,
            "summary": self._generate_summary(baseline_results, cotrr_results, comparison)
        }
        
        # 保存报告
        with open(comparison_output, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        logger.info(f"✅ 综合测试完成，报告保存至: {comparison_output}")
        
        return full_report
    
    def _generate_summary(self, baseline_results: Dict, cotrr_results: Dict, comparison: Dict) -> Dict:
        """生成测试摘要"""
        summary = {
            "overall_status": "UNKNOWN",
            "key_findings": [],
            "performance_verdict": "UNKNOWN",
            "recommendations": []
        }
        
        # 检查基本成功率
        baseline_success = baseline_results.get("success", False)
        cotrr_success = cotrr_results.get("success", False)
        
        if not baseline_success and not cotrr_success:
            summary["overall_status"] = "BOTH_FAILED"
            summary["key_findings"].append("Both pipelines failed to execute")
            summary["recommendations"].append("Debug pipeline setup and dependencies")
        elif not baseline_success:
            summary["overall_status"] = "BASELINE_FAILED"
            summary["key_findings"].append("Baseline pipeline failed, CoTRR succeeded")
            summary["recommendations"].append("Check baseline pipeline configuration")
        elif not cotrr_success:
            summary["overall_status"] = "COTRR_FAILED"
            summary["key_findings"].append("CoTRR pipeline failed, baseline succeeded")
            summary["recommendations"].append("Debug CoTRR integration issues")
        else:
            summary["overall_status"] = "BOTH_SUCCESS"
            
            # 分析性能
            overhead_ratio = comparison.get("execution_time_comparison", {}).get("overhead_ratio", 0)
            if overhead_ratio <= 2.0:
                summary["performance_verdict"] = "ACCEPTABLE_OVERHEAD"
                summary["key_findings"].append(f"CoTRR overhead: {overhead_ratio:.2f}x baseline time")
            else:
                summary["performance_verdict"] = "HIGH_OVERHEAD"
                summary["key_findings"].append(f"High CoTRR overhead: {overhead_ratio:.2f}x baseline time")
                summary["recommendations"].append("Optimize CoTRR inference performance")
            
            # 分析质量改进
            score_improvement = comparison.get("quality_metrics", {}).get("average_scores", {}).get("improvement", 0)
            if score_improvement > 0.05:
                summary["key_findings"].append(f"Significant quality improvement: +{score_improvement:.3f}")
                summary["recommendations"].append("CoTRR shows promise, consider production deployment")
            elif score_improvement > 0.01:
                summary["key_findings"].append(f"Modest quality improvement: +{score_improvement:.3f}")
                summary["recommendations"].append("Further optimization may be beneficial")
            else:
                summary["key_findings"].append(f"Limited quality improvement: +{score_improvement:.3f}")
                summary["recommendations"].append("Reassess CoTRR value proposition")
        
        return summary

if __name__ == "__main__":
    comparator = PipelineComparator()
    report = comparator.run_comprehensive_test()
    
    # 打印关键结果
    print("\n" + "="*50)
    print("🎯 Day 3 Pipeline Comparison Results")
    print("="*50)
    
    summary = report.get("summary", {})
    print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"Performance Verdict: {summary.get('performance_verdict', 'UNKNOWN')}")
    
    print("\n🔍 Key Findings:")
    for finding in summary.get("key_findings", []):
        print(f"• {finding}")
    
    print("\n💡 Recommendations:")
    for rec in summary.get("recommendations", []):
        print(f"• {rec}")
    
    print("\n📊 Detailed report saved to research/day3_results/")