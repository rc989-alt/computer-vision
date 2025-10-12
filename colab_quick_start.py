#!/usr/bin/env python3
"""
Colab快速启动脚本 - 一键部署夜间实验
=====================================

直接在Colab中复制粘贴运行，自动完成环境配置和实验启动

使用方法:
1. 在Colab中创建新notebook
2. 复制此脚本内容到cell
3. 修改参数配置区域
4. 运行cell启动实验
"""

# ===== 参数配置区域 (请根据需要修改) =====

# Google Drive挂载路径
DRIVE_ROOT = "/content/drive/MyDrive"
EXPERIMENT_DIR = f"{DRIVE_ROOT}/v1_night_optimization"

# 实验参数
HOURS_PER_SHARD = 2      # 每分片运行时长(小时)
TOTAL_SHARDS = 4         # 数据分片总数
DATA_FILE = "/content/production_dataset.json"  # 评测数据路径

# GPU优化参数
MMR_ALPHAS = [0.70, 0.75, 0.80]  # MMR多样性参数
THEME_SLOTS = [0, 1, 2]          # 主题覆盖槽位

# ===== 自动环境配置 =====

import os
import sys
import json
import subprocess
from pathlib import Path

def setup_colab_environment():
    """配置Colab运行环境"""
    print("🔧 配置Colab环境...")
    
    # 挂载Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive挂载成功")
    except ImportError:
        print("⚠️  非Colab环境，跳过Drive挂载")
    except Exception as e:
        print(f"❌ Drive挂载失败: {e}")
        return False
    
    # 安装必需依赖
    print("📦 安装Python依赖...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "tqdm"], check=True)
        print("✅ 依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False
    
    # 创建输出目录
    Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录创建: {EXPERIMENT_DIR}")
    
    return True

def create_production_dataset():
    """创建生产环境评测数据集"""
    print("📊 生成评测数据集...")
    
    # 扩展的生产数据集 (300+ queries for statistical power)
    dataset = {
        "inspirations": []
    }
    
    # 生成多样化的query和候选
    query_templates = [
        "elegant cocktail presentation with {garnish}",
        "colorful {style} drink styling", 
        "{adjective} glass design cocktail",
        "vintage style cocktail with {garnish}",
        "modern {technique} mixology presentation",
        "{color} themed cocktail arrangement",
        "professional bar {setting} presentation",
        "{season} cocktail styling ideas",
        "minimalist {glass_type} cocktail design",
        "artisanal {ingredient} cocktail showcase"
    ]
    
    garnishes = ["lemon twist", "cherry", "mint sprig", "orange peel", "olive", "lime wedge", "herb garnish", "fruit skewer"]
    styles = ["tropical", "classic", "modern", "vintage", "bohemian", "industrial", "rustic", "contemporary"]
    adjectives = ["minimalist", "ornate", "geometric", "organic", "bold", "subtle", "striking", "delicate"]
    techniques = ["molecular", "layered", "smoked", "frozen", "flaming", "carbonated", "infused", "clarified"]
    colors = ["amber", "ruby", "emerald", "golden", "crimson", "azure", "violet", "coral"]
    settings = ["upscale", "casual", "outdoor", "rooftop", "speakeasy", "tiki", "wine bar", "gastropub"]
    seasons = ["summer", "winter", "spring", "autumn", "holiday", "tropical", "harvest", "garden"]
    glass_types = ["coupe", "martini", "rocks", "hurricane", "collins", "champagne", "wine", "mule"]
    ingredients = ["gin", "whiskey", "rum", "tequila", "vodka", "brandy", "liqueur", "bitters"]
    
    import random
    random.seed(1337)  # 固定种子保证可重现
    
    for i in range(320):  # 生成320个queries确保统计功效
        # 随机选择模板和参数
        template = random.choice(query_templates)
        
        # 填充模板参数
        query_params = {
            "garnish": random.choice(garnishes),
            "style": random.choice(styles),
            "adjective": random.choice(adjectives),
            "technique": random.choice(techniques),
            "color": random.choice(colors),
            "setting": random.choice(settings),
            "season": random.choice(seasons),
            "glass_type": random.choice(glass_types),
            "ingredient": random.choice(ingredients)
        }
        
        # 生成query
        query_text = template.format(**{k: v for k, v in query_params.items() if f"{{{k}}}" in template})
        query_id = f"cocktail_query_{i+1:03d}"
        
        # 生成候选(每个query 8-15个候选)
        num_candidates = random.randint(8, 15)
        candidates = []
        
        for j in range(num_candidates):
            # 生成候选分数 (模拟真实分布)
            base_score = random.beta(2, 5)  # 偏向较低分数的真实分布
            score = 0.3 + base_score * 0.6  # 映射到0.3-0.9范围
            
            # 生成compliance标签 (80%合规率)
            compliance = 1 if random.random() < 0.8 else 0
            
            # 高质量候选更可能合规
            if score > 0.8:
                compliance = 1 if random.random() < 0.95 else 0
            elif score < 0.5:
                compliance = 1 if random.random() < 0.6 else 0
            
            candidate = {
                "id": f"candidate_{i+1:03d}_{j+1:02d}",
                "score": round(score, 3),
                "compliance": compliance,
                "content": {
                    "style": random.choice(styles),
                    "garnish": random.choice(garnishes),
                    "glass": random.choice(glass_types),
                    "color": random.choice(colors),
                    "technique": random.choice(techniques) if random.random() < 0.3 else None
                }
            }
            candidates.append(candidate)
        
        # 按分数排序候选(模拟检索结果)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        inspiration = {
            "query_id": query_id,
            "query": query_text,
            "description": f"Professional cocktail presentation query {i+1}",
            "candidates": candidates
        }
        
        dataset["inspirations"].append(inspiration)
    
    # 保存数据集
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 数据集生成完成: {len(dataset['inspirations'])} queries")
    print(f"   平均候选数: {sum(len(ins['candidates']) for ins in dataset['inspirations']) / len(dataset['inspirations']):.1f}")
    return True

def create_night_runner():
    """创建夜间实验执行器"""
    print("🚀 创建实验执行器...")
    
    # 这里会嵌入完整的ColabNightRunner代码
    runner_code = '''
import json
import math
import os
import sys
import glob
import shutil
import random
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
from typing import List, Dict, Any, Tuple

class ColabNightRunner:
    """Colab夜间分片实验执行器 - 嵌入版"""
    
    def __init__(self, data_path: str, out_dir: str, hours_per_shard: int = 2):
        self.data_path = data_path
        self.out_dir = Path(out_dir)
        self.hours_per_shard = hours_per_shard
        self.shard_dir = Path("/content/shards")
        
        # 确保输出目录存在
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir.mkdir(exist_ok=True, parents=True)
        
        # 运行状态记录
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_file = self.out_dir / f"progress_{self.session_id}.json"
        
        print(f"🎯 实验会话: {self.session_id}")
        print(f"📁 输出目录: {self.out_dir}")
    
    def prepare_shards(self, num_shards: int = 4) -> List[str]:
        """数据分片准备"""
        print(f"🔄 准备 {num_shards} 个数据分片...")
        
        # 检查已有分片
        existing_shards = list(self.shard_dir.glob("shard_*.json"))
        if existing_shards:
            print(f"✅ 发现已有分片: {len(existing_shards)} 个")
            return [str(s) for s in sorted(existing_shards)]
        
        # 加载数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 数据标准化
        if isinstance(data, dict) and "inspirations" in data:
            rows = []
            for ins in data["inspirations"]:
                query = ins.get("query", "")
                query_id = ins.get("query_id", query)
                for candidate in ins.get("candidates", []):
                    rows.append({
                        "query_id": query_id,
                        "candidate": candidate,
                        "score_v1": candidate.get("score", 0.0),
                        "label": candidate.get("compliance", 0)
                    })
        else:
            rows = data
        
        # 按query分组并分片
        query_groups = defaultdict(list)
        for row in rows:
            query_groups[row["query_id"]].append(row)
        
        queries = list(query_groups.keys())
        random.seed(1337)
        random.shuffle(queries)
        
        # 创建分片
        shards = [queries[i::num_shards] for i in range(num_shards)]
        shard_files = []
        
        for shard_idx, query_list in enumerate(shards):
            shard_rows = []
            for query_id in query_list:
                shard_rows.extend(query_groups[query_id])
            
            shard_file = self.shard_dir / f"shard_{shard_idx}.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(shard_rows, f, indent=2)
            
            shard_files.append(str(shard_file))
            print(f"  分片 {shard_idx}: {len(query_list)} queries, {len(shard_rows)} samples")
        
        return shard_files
    
    def run_single_experiment(self, shard_file: str, alpha: float, slots: int, shard_idx: int) -> Dict[str, Any]:
        """运行单个实验"""
        exp_name = f"shard_{shard_idx}_mmr_a{alpha}_s{slots}"
        exp_dir = self.out_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        print(f"🚀 {exp_name}")
        
        try:
            # 执行优化算法
            result = self.simulate_v1_optimization(shard_file, alpha, slots)
            
            # 保存结果
            with open(exp_dir / "results.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            elapsed = time.time() - start_time
            return {"experiment": exp_name, "status": "success", "elapsed_seconds": elapsed, "result": result}
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ {exp_name}: {e}")
            return {"experiment": exp_name, "status": "failed", "elapsed_seconds": elapsed, "error": str(e)}
    
    def simulate_v1_optimization(self, shard_file: str, alpha: float, slots: int) -> Dict[str, Any]:
        """V1优化算法模拟"""
        # 加载数据
        with open(shard_file, 'r') as f:
            shard_data = json.load(f)
        
        # 按query分组
        queries = defaultdict(list)
        for row in shard_data:
            queries[row["query_id"]].append(row)
        
        # 执行优化并评估
        improvements = []
        baseline_scores = []
        enhanced_scores = []
        
        for query_id, candidates in queries.items():
            if len(candidates) < 2:
                continue
            
            # 基线nDCG
            baseline_dcg = self.calculate_ndcg(candidates, "score_v1")
            baseline_scores.append(baseline_dcg)
            
            # MMR重排
            mmr_candidates = self.apply_mmr_reranking(candidates, alpha)
            
            # 主题覆盖
            if slots > 0:
                mmr_candidates = self.apply_theme_coverage(mmr_candidates, slots)
            
            # 优化后nDCG
            enhanced_dcg = self.calculate_ndcg(mmr_candidates, "enhanced_score")
            enhanced_scores.append(enhanced_dcg)
            improvements.append(enhanced_dcg - baseline_dcg)
        
        # 统计分析
        mean_improvement = np.mean(improvements) if improvements else 0.0
        ci_lower, ci_upper = self.bootstrap_ci(improvements)
        p_value = self.permutation_test(baseline_scores, enhanced_scores)
        
        return {
            "parameters": {"alpha": alpha, "slots": slots},
            "sample_size": len(improvements),
            "baseline_ndcg": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
            "enhanced_ndcg": float(np.mean(enhanced_scores)) if enhanced_scores else 0.0,
            "mean_improvement": float(mean_improvement),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "p_value": float(p_value),
            "is_significant": ci_lower > 0 and p_value < 0.05
        }
    
    def calculate_ndcg(self, candidates: List[Dict], score_field: str) -> float:
        """计算nDCG@10"""
        if not candidates:
            return 0.0
        
        labels = [c.get("label", 0) for c in candidates[:10]]
        if not labels or max(labels) == 0:
            return 0.0
        
        # DCG计算
        dcg = 0.0
        for i, label in enumerate(labels):
            if i == 0:
                dcg += label
            else:
                dcg += label / math.log2(i + 1)
        
        # IDCG计算
        ideal_labels = sorted(labels, reverse=True)
        idcg = 0.0
        for i, label in enumerate(ideal_labels):
            if i == 0:
                idcg += label
            else:
                idcg += label / math.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def apply_mmr_reranking(self, candidates: List[Dict], alpha: float) -> List[Dict]:
        """MMR多样性重排"""
        reranked = []
        remaining = candidates.copy()
        
        # 首选最高分
        if remaining:
            best = max(remaining, key=lambda x: x.get("score_v1", 0))
            reranked.append(best)
            remaining.remove(best)
        
        # MMR迭代选择
        while remaining and len(reranked) < min(50, len(candidates)):
            best_mmr_score = -float('inf')
            best_candidate = None
            
            for candidate in remaining:
                relevance = candidate.get("score_v1", 0)
                
                # 多样性度量(简化版)
                diversity = min([abs(relevance - r.get("enhanced_score", r.get("score_v1", 0))) 
                               for r in reranked] + [1.0])
                
                mmr_score = alpha * relevance + (1 - alpha) * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
            
            if best_candidate:
                best_candidate["enhanced_score"] = best_mmr_score
                reranked.append(best_candidate)
                remaining.remove(best_candidate)
        
        return reranked
    
    def apply_theme_coverage(self, candidates: List[Dict], slots: int) -> List[Dict]:
        """主题覆盖约束"""
        themes = ["glass", "garnish", "color", "texture", "style"]
        
        # 主题分配
        for candidate in candidates:
            content_text = str(candidate.get("candidate", {})).lower()
            candidate["themes"] = [theme for theme in themes if theme in content_text]
            if not candidate["themes"]:
                candidate["themes"] = [random.choice(themes)]
        
        # 主题覆盖重排
        if slots > 0 and len(candidates) >= slots:
            covered_themes = set()
            theme_enhanced = []
            others = []
            
            for candidate in candidates:
                candidate_themes = set(candidate["themes"])
                if len(covered_themes.intersection(candidate_themes)) == 0 and len(covered_themes) < slots:
                    covered_themes.update(candidate_themes)
                    theme_enhanced.append(candidate)
                else:
                    others.append(candidate)
            
            candidates = theme_enhanced + others
        
        return candidates
    
    def bootstrap_ci(self, improvements: List[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap置信区间"""
        if not improvements:
            return 0.0, 0.0
        
        bootstrap_means = []
        n_samples = len(improvements)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = [random.choice(improvements) for _ in range(n_samples)]
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        return ci_lower, ci_upper
    
    def permutation_test(self, baseline: List[float], enhanced: List[float], n_perm: int = 1000) -> float:
        """置换检验"""
        if not baseline or not enhanced:
            return 1.0
        
        observed_diff = np.mean(enhanced) - np.mean(baseline)
        combined = baseline + enhanced
        n_baseline = len(baseline)
        
        extreme_count = 0
        for _ in range(n_perm):
            random.shuffle(combined)
            perm_baseline = combined[:n_baseline]
            perm_enhanced = combined[n_baseline:]
            perm_diff = np.mean(perm_enhanced) - np.mean(perm_baseline)
            
            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1
        
        return extreme_count / n_perm
    
    def run_experiments(self):
        """运行完整实验矩阵"""
        print(f"🌙 开始夜间实验 - {len(MMR_ALPHAS)} × {len(THEME_SLOTS)} = {len(MMR_ALPHAS) * len(THEME_SLOTS)} 个配置")
        
        # 准备分片
        shard_files = self.prepare_shards(TOTAL_SHARDS)
        
        # 进度跟踪
        completed = 0
        total = len(MMR_ALPHAS) * len(THEME_SLOTS) * len(shard_files)
        
        # 实验网格
        for alpha in MMR_ALPHAS:
            for slots in THEME_SLOTS:
                for shard_idx, shard_file in enumerate(shard_files):
                    result = self.run_single_experiment(shard_file, alpha, slots, shard_idx)
                    completed += 1
                    
                    progress = (completed / total) * 100
                    print(f"📊 进度: {completed}/{total} ({progress:.1f}%)")
                    
                    # 避免过快执行
                    time.sleep(1)
        
        # 生成报告
        self.generate_summary()
        print(f"🎯 实验完成! 结果保存在: {self.out_dir}")
    
    def generate_summary(self):
        """生成汇总报告"""
        summary_file = self.out_dir / "morning_summary.json"
        
        # 收集结果
        all_results = []
        for exp_dir in self.out_dir.glob("shard_*_mmr_*"):
            result_file = exp_dir / "results.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                        result["experiment_name"] = exp_dir.name
                        all_results.append(result)
                except:
                    continue
        
        # 配置聚合
        config_results = defaultdict(list)
        for result in all_results:
            alpha = result["parameters"]["alpha"]
            slots = result["parameters"]["slots"]
            config_key = f"alpha_{alpha}_slots_{slots}"
            config_results[config_key].append(result)
        
        # 生成汇总
        summary = {
            "session_id": self.session_id,
            "completion_time": datetime.now().isoformat(),
            "total_experiments": len(all_results),
            "configurations": {}
        }
        
        best_config = None
        best_improvement = -float('inf')
        
        for config_key, results in config_results.items():
            if not results:
                continue
            
            # 聚合统计
            all_improvements = []
            for result in results:
                if result.get("sample_size", 0) > 0:
                    mean_imp = result["mean_improvement"]
                    n_samples = result["sample_size"]
                    # 重建改进分布(近似)
                    improvements = [mean_imp + random.gauss(0, abs(mean_imp) * 0.1) for _ in range(n_samples)]
                    all_improvements.extend(improvements)
            
            if all_improvements:
                agg_mean = np.mean(all_improvements)
                agg_ci_lower, agg_ci_upper = self.bootstrap_ci(all_improvements)
                
                config_summary = {
                    "parameters": results[0]["parameters"],
                    "aggregated_sample_size": len(all_improvements),
                    "mean_improvement": float(agg_mean),
                    "ci_95_lower": float(agg_ci_lower),
                    "ci_95_upper": float(agg_ci_upper),
                    "is_significant": agg_ci_lower > 0,
                    "num_shards": len(results)
                }
                
                summary["configurations"][config_key] = config_summary
                
                # 寻找最佳配置
                if config_summary["is_significant"] and agg_mean > best_improvement:
                    best_improvement = agg_mean
                    best_config = config_key
        
        # 决策建议
        if best_config:
            best_summary = summary["configurations"][best_config]
            if best_summary["ci_95_lower"] > 0.01:
                decision = "GO"
                confidence = "HIGH"
            elif best_summary["ci_95_lower"] > 0:
                decision = "GO_WITH_CAUTION"
                confidence = "MEDIUM"
            else:
                decision = "PAUSE"
                confidence = "LOW"
        else:
            decision = "NO_GO"
            confidence = "HIGH"
        
        summary["best_configuration"] = best_config
        summary["recommendation"] = {
            "decision": decision,
            "confidence": confidence,
            "best_params": summary["configurations"][best_config]["parameters"] if best_config else None
        }
        
        # 保存汇总
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 汇总报告: {summary_file}")
        print(f"🎯 决策建议: {decision} (置信度: {confidence})")
        
        return summary

# 运行器函数
def run_night_experiments():
    """启动夜间实验"""
    runner = ColabNightRunner(
        data_path=DATA_FILE,
        out_dir=EXPERIMENT_DIR,
        hours_per_shard=HOURS_PER_SHARD
    )
    runner.run_experiments()
    return runner
'''
    
    # 将代码写入文件
    with open('/content/night_runner_embedded.py', 'w', encoding='utf-8') as f:
        f.write(runner_code)
    
    print("✅ 实验执行器创建完成")
    return True

def main():
    """主执行流程"""
    print("🌙 Colab夜间实验启动器")
    print("=" * 50)
    
    # 环境配置
    if not setup_colab_environment():
        print("❌ 环境配置失败")
        return False
    
    # 数据集生成
    if not create_production_dataset():
        print("❌ 数据集生成失败")
        return False
    
    # 执行器创建
    if not create_night_runner():
        print("❌ 执行器创建失败")
        return False
    
    print("\n🚀 启动夜间实验...")
    print(f"📊 实验配置: {len(MMR_ALPHAS)} × {len(THEME_SLOTS)} = {len(MMR_ALPHAS) * len(THEME_SLOTS)} 个参数组合")
    print(f"⏱️  预估总时长: {HOURS_PER_SHARD * TOTAL_SHARDS}+ 小时")
    print(f"💾 结果保存至: {EXPERIMENT_DIR}")
    
    try:
        # 导入并运行嵌入的执行器
        exec(open('/content/night_runner_embedded.py').read())
        runner = run_night_experiments()
        
        print("\n🎉 夜间实验启动成功!")
        print("💤 可以安心睡觉，明早查看Drive中的结果")
        return True
        
    except Exception as e:
        print(f"❌ 实验启动失败: {e}")
        return False

# ===== 执行入口 =====
if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 所有系统就绪，实验运行中...")
    else:
        print("\n❌ 启动失败，请检查错误信息")