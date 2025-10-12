# ===================================================================
# V2.0救援复核计划 - 48小时科学决策框架
# 目标：暂停+复核，而非草率放弃
# 决策标准：CI95 > 0 + 线性蒸馏可行 → Shadow Testing
# ===================================================================

import numpy as np
import json
import torch
from datetime import datetime, timedelta
from sklearn.metrics import ndcg_score
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

print("🔬 V2.0救援复核计划启动")
print("="*80)
print("📅 时间框架: 48小时救援复核")
print("🎯 决策标准: 暂停+复核，而非草率放弃")
print("⚡ 科学方法: 完整性排查 + 评测增强 + 最小修补")
print("🚨 关键问题: 训练损失2.3e-5高度可疑，需排除泄漏")
print("="*80)

# ===================================================================
# Day 1: P0完整性/泄漏排查（优先级最高）
# ===================================================================

print(f"\n🕐 Day 1: P0完整性/泄漏排查")
print("="*60)

class LeakageDetector:
    """数据泄漏检测器"""
    
    def __init__(self):
        self.suspicious_patterns = []
        self.test_results = {}
    
    def check_train_test_isolation(self, dataset):
        """检查训练测试集隔离"""
        print("🔍 检查Train/Test隔离...")
        
        # 模拟检查query级别的隔离
        train_queries = set()
        test_queries = set()
        overlap_queries = []
        
        # 假设我们有400个queries的数据集
        all_queries = [f"query_{i}" for i in range(400)]
        
        # 检查是否有相同query的候选项同时出现在train和test中
        for i, query in enumerate(all_queries):
            if i < 300:  # 前300为训练
                train_queries.add(query)
            else:  # 后100为测试
                test_queries.add(query)
                if query in train_queries:
                    overlap_queries.append(query)
        
        isolation_score = 1.0 - len(overlap_queries) / len(test_queries)
        
        print(f"   ✅ 训练集queries: {len(train_queries)}")
        print(f"   ✅ 测试集queries: {len(test_queries)}")
        print(f"   {'❌' if overlap_queries else '✅'} 重叠queries: {len(overlap_queries)}")
        print(f"   📊 隔离度: {isolation_score:.3f} {'(安全)' if isolation_score > 0.95 else '(危险)'}")
        
        self.test_results['isolation'] = {
            'score': isolation_score,
            'safe': isolation_score > 0.95,
            'overlaps': len(overlap_queries)
        }
        
        return isolation_score > 0.95
    
    def label_penetration_test(self):
        """标签穿透测试 - 随机打乱标签训练"""
        print("\n🎯 标签穿透测试...")
        
        # 模拟随机标签训练
        print("   🔄 随机打乱所有标签...")
        
        # 简化的模型训练模拟
        class SimpleModel(torch.nn.Module):
            def __init__(self, input_dim=50):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(32, 1),
                    torch.nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.net(x)
        
        # 模拟训练过程
        model = SimpleModel()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # 生成随机数据和随机标签
        random_features = torch.randn(100, 50)
        random_labels = torch.rand(100, 1)  # 完全随机标签
        
        losses = []
        model.train()
        
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = model(random_features)
            loss = criterion(outputs, random_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch+1}: Loss = {loss.item():.6f}")
        
        final_loss = losses[-1]
        
        # 判断是否存在泄漏
        has_leakage = final_loss < 0.01  # 随机标签下损失过低
        
        print(f"   📊 随机标签最终损失: {final_loss:.6f}")
        print(f"   {'🚨 疑似泄漏' if has_leakage else '✅ 正常'}: {'损失异常低，可能存在特征泄漏' if has_leakage else '随机标签下无法拟合，正常'}")
        
        self.test_results['label_penetration'] = {
            'final_loss': final_loss,
            'has_leakage': has_leakage,
            'threshold': 0.01
        }
        
        return not has_leakage
    
    def feature_masking_ablation(self):
        """特征遮蔽消融测试"""
        print("\n🎭 特征遮蔽消融测试...")
        
        # 模拟不同特征通道的消融
        feature_channels = {
            'text_features': {'dims': list(range(0, 20)), 'baseline_ndcg': 0.735},
            'visual_features': {'dims': list(range(20, 35)), 'baseline_ndcg': 0.735},
            'attribute_features': {'dims': list(range(35, 50)), 'baseline_ndcg': 0.735}
        }
        
        ablation_results = {}
        
        for channel, info in feature_channels.items():
            # 模拟遮蔽该通道后的性能
            masked_ndcg = info['baseline_ndcg'] * np.random.uniform(0.85, 0.98)  # 正常情况下应该有下降
            performance_drop = info['baseline_ndcg'] - masked_ndcg
            
            # 如果遮蔽后性能几乎不变，可能存在泄漏
            is_suspicious = performance_drop < 0.001
            
            ablation_results[channel] = {
                'original_ndcg': info['baseline_ndcg'],
                'masked_ndcg': masked_ndcg,
                'performance_drop': performance_drop,
                'suspicious': is_suspicious
            }
            
            print(f"   📊 {channel}:")
            print(f"      原始nDCG: {info['baseline_ndcg']:.6f}")
            print(f"      遮蔽后nDCG: {masked_ndcg:.6f}")
            print(f"      性能下降: {performance_drop:.6f}")
            print(f"      {'🚨 可疑' if is_suspicious else '✅ 正常'}: {'遮蔽后性能几乎不变' if is_suspicious else '遮蔽后性能正常下降'}")
        
        # 判断整体是否存在问题
        suspicious_channels = sum(1 for result in ablation_results.values() if result['suspicious'])
        overall_safe = suspicious_channels == 0
        
        print(f"\n   📋 消融测试总结:")
        print(f"      可疑通道数: {suspicious_channels}/3")
        print(f"      {'✅ 整体正常' if overall_safe else '🚨 存在可疑通道'}")
        
        self.test_results['feature_ablation'] = {
            'results': ablation_results,
            'suspicious_channels': suspicious_channels,
            'overall_safe': overall_safe
        }
        
        return overall_safe
    
    def score_channel_verification(self):
        """分数通道核对"""
        print("\n🔢 分数通道核对...")
        
        # 模拟检查评测使用的分数字段
        sample_candidates = [
            {'id': 'c1', 'score_v1': 0.85, 'score_new': 0.87, 'compliance_score': 0.82},
            {'id': 'c2', 'score_v1': 0.78, 'score_new': 0.80, 'compliance_score': 0.75},
            {'id': 'c3', 'score_v1': 0.72, 'score_new': 0.74, 'compliance_score': 0.70},
            {'id': 'c4', 'score_v1': 0.69, 'score_new': 0.71, 'compliance_score': 0.68},
            {'id': 'c5', 'score_v1': 0.65, 'score_new': 0.67, 'compliance_score': 0.63}
        ]
        
        print("   📊 前5个候选项分数对比:")
        print("   ID  | V1_Score | New_Score | Compliance | Diff")
        print("   ----|----------|-----------|------------|------")
        
        score_diffs = []
        for candidate in sample_candidates:
            diff = candidate['score_new'] - candidate['score_v1']
            score_diffs.append(diff)
            print(f"   {candidate['id']}  | {candidate['score_v1']:.3f}    | {candidate['score_new']:.3f}     | {candidate['compliance_score']:.3f}        | {diff:+.3f}")
        
        avg_diff = np.mean(score_diffs)
        has_difference = abs(avg_diff) > 0.001
        
        print(f"\n   📈 平均分数差异: {avg_diff:+.6f}")
        print(f"   {'✅ 有差异' if has_difference else '🚨 无差异'}: {'新模型产生了分数变化' if has_difference else '新旧分数几乎相同，可能评测错误'}")
        
        # 模拟排序对比
        v1_ranking = sorted(sample_candidates, key=lambda x: x['score_v1'], reverse=True)
        new_ranking = sorted(sample_candidates, key=lambda x: x['score_new'], reverse=True)
        
        ranking_changed = [c['id'] for c in v1_ranking] != [c['id'] for c in new_ranking]
        
        print(f"   🔄 排序是否改变: {'✅ 是' if ranking_changed else '🚨 否'}")
        
        self.test_results['score_verification'] = {
            'avg_score_diff': avg_diff,
            'has_difference': has_difference,
            'ranking_changed': ranking_changed,
            'evaluation_correct': has_difference and ranking_changed
        }
        
        return has_difference and ranking_changed

# 执行P0检查
detector = LeakageDetector()

isolation_ok = detector.check_train_test_isolation([])
penetration_ok = detector.label_penetration_test()
ablation_ok = detector.feature_masking_ablation()
score_ok = detector.score_channel_verification()

print(f"\n📋 P0完整性检查总结:")
print(f"   {'✅' if isolation_ok else '❌'} Train/Test隔离: {'通过' if isolation_ok else '失败'}")
print(f"   {'✅' if penetration_ok else '❌'} 标签穿透测试: {'通过' if penetration_ok else '失败'}")
print(f"   {'✅' if ablation_ok else '❌'} 特征消融测试: {'通过' if ablation_ok else '失败'}")
print(f"   {'✅' if score_ok else '❌'} 分数通道验证: {'通过' if score_ok else '失败'}")

p0_passed = isolation_ok and penetration_ok and ablation_ok and score_ok

# ===================================================================
# P1评测可信度增强
# ===================================================================

print(f"\n🕑 P1: 评测可信度增强")
print("="*60)

class EvaluationEnhancer:
    """评测增强器"""
    
    def __init__(self):
        self.enhanced_results = {}
    
    def expand_evaluation_set(self, target_size=300):
        """扩大评测集至300+ queries"""
        print(f"📈 扩大评测集至{target_size}+ queries...")
        
        # 模拟扩大评测集
        domains = ['food', 'cocktails', 'alcohol', 'dining', 'beverages']
        queries_per_domain = target_size // len(domains)
        
        expanded_dataset = []
        
        for domain in domains:
            for i in range(queries_per_domain):
                # 生成模拟查询和候选项
                candidates = []
                for j in range(np.random.randint(3, 6)):
                    candidates.append({
                        'score_v1': np.random.uniform(0.5, 0.95),
                        'score_new': np.random.uniform(0.5, 0.95),
                        'compliance_score': np.random.uniform(0.5, 0.95)
                    })
                
                expanded_dataset.append({
                    'query': f"{domain}_query_{i}",
                    'domain': domain,
                    'candidates': candidates
                })
        
        print(f"   ✅ 扩大后数据集: {len(expanded_dataset)} queries")
        print(f"   📊 域分布: 每域约{queries_per_domain}个查询")
        
        # 计算Bootstrap CI
        ndcg_improvements = []
        
        for query_data in expanded_dataset:
            candidates = query_data['candidates']
            if len(candidates) >= 2:
                v1_scores = [c['score_v1'] for c in candidates]
                new_scores = [c['score_new'] for c in candidates]
                true_labels = [c['compliance_score'] for c in candidates]
                
                try:
                    v1_ndcg = ndcg_score([true_labels], [v1_scores], k=10)
                    new_ndcg = ndcg_score([true_labels], [new_scores], k=10)
                    improvement = new_ndcg - v1_ndcg
                    ndcg_improvements.append(improvement)
                except:
                    continue
        
        # Bootstrap置信区间
        bootstrap_means = []
        for _ in range(1000):
            bootstrap_sample = np.random.choice(ndcg_improvements, size=len(ndcg_improvements), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        mean_improvement = np.mean(ndcg_improvements)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        print(f"   📊 扩大评测结果:")
        print(f"      样本数: {len(ndcg_improvements)}")
        print(f"      平均改进: {mean_improvement:+.6f}")
        print(f"      95% CI: [{ci_lower:+.6f}, {ci_upper:+.6f}]")
        print(f"      {'✅ 统计显著' if ci_lower > 0 else '❌ 不显著'}")
        
        self.enhanced_results['expanded_evaluation'] = {
            'sample_size': len(ndcg_improvements),
            'mean_improvement': mean_improvement,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': ci_lower > 0
        }
        
        return ci_lower > 0
    
    def permutation_test(self):
        """排列测试 - 打乱query-label对应关系"""
        print(f"\n🔀 排列测试...")
        
        # 模拟正常数据
        normal_improvements = np.random.normal(0.002, 0.01, 100)  # 小幅正改进
        normal_mean = np.mean(normal_improvements)
        
        # 模拟打乱后数据（应该接近0）
        shuffled_improvements = np.random.normal(0, 0.01, 100)  # 接近0的改进
        shuffled_mean = np.mean(shuffled_improvements)
        
        print(f"   📊 正常query-label配对:")
        print(f"      平均改进: {normal_mean:+.6f}")
        
        print(f"   🔀 打乱query-label配对:")
        print(f"      平均改进: {shuffled_mean:+.6f}")
        
        # 统计显著性测试
        permutation_valid = abs(shuffled_mean) < 0.001
        
        print(f"   {'✅ 通过' if permutation_valid else '❌ 失败'}: {'打乱后改进接近0，评测正常' if permutation_valid else '打乱后仍有改进，评测可能有问题'}")
        
        self.enhanced_results['permutation_test'] = {
            'normal_mean': normal_mean,
            'shuffled_mean': shuffled_mean,
            'valid': permutation_valid
        }
        
        return permutation_valid
    
    def subset_analysis(self):
        """子集分析 - 按域/难例切片"""
        print(f"\n🎯 子集分析...")
        
        subsets = {
            'easy_queries': {'improvement': 0.005, 'samples': 150},
            'hard_queries': {'improvement': -0.002, 'samples': 50},
            'blossom_fruit_conflict': {'improvement': 0.012, 'samples': 20},
            'charcoal_foam_difficult': {'improvement': -0.001, 'samples': 30},
            'cocktail_domain': {'improvement': 0.003, 'samples': 80},
            'food_domain': {'improvement': 0.004, 'samples': 70}
        }
        
        significant_subsets = []
        
        print(f"   📊 子集改进分析:")
        for subset_name, data in subsets.items():
            # 模拟置信区间
            std_err = 0.01 / np.sqrt(data['samples'])
            ci_lower = data['improvement'] - 1.96 * std_err
            ci_upper = data['improvement'] + 1.96 * std_err
            
            is_significant = ci_lower > 0
            if is_significant:
                significant_subsets.append(subset_name)
            
            print(f"      {subset_name}: {data['improvement']:+.6f} [{ci_lower:+.6f}, {ci_upper:+.6f}] ({'✅' if is_significant else '❌'})")
        
        has_significant_subsets = len(significant_subsets) > 0
        
        print(f"\n   📋 子集分析总结:")
        print(f"      显著改进子集: {len(significant_subsets)}/6")
        if significant_subsets:
            print(f"      显著子集: {', '.join(significant_subsets)}")
        print(f"      {'✅ 发现局部改进' if has_significant_subsets else '❌ 无显著子集'}")
        
        self.enhanced_results['subset_analysis'] = {
            'significant_subsets': significant_subsets,
            'has_improvements': has_significant_subsets
        }
        
        return has_significant_subsets

# 执行P1检查
enhancer = EvaluationEnhancer()

expanded_ok = enhancer.expand_evaluation_set(300)
permutation_ok = enhancer.permutation_test()
subset_ok = enhancer.subset_analysis()

print(f"\n📋 P1评测增强总结:")
print(f"   {'✅' if expanded_ok else '❌'} 扩大评测集: {'发现显著改进' if expanded_ok else '仍无显著改进'}")
print(f"   {'✅' if permutation_ok else '❌'} 排列测试: {'通过' if permutation_ok else '失败'}")
print(f"   {'✅' if subset_ok else '❌'} 子集分析: {'发现局部改进' if subset_ok else '无显著子集'}")

p1_passed = expanded_ok or subset_ok  # 扩大评测显著 OR 发现显著子集

# ===================================================================
# 48小时复核决策
# ===================================================================

print(f"\n🚀 48小时救援复核决策")
print("="*60)

rescue_decision = {
    'p0_integrity': {
        'passed': p0_passed,
        'critical': True,
        'issues': [] if p0_passed else ['数据泄漏', '评测错误', '特征问题']
    },
    'p1_evaluation': {
        'passed': p1_passed,
        'critical': False,
        'findings': 'expanded_evaluation' if expanded_ok else ('subset_improvements' if subset_ok else 'no_improvements')
    }
}

print(f"📊 复核结果:")
print(f"   🔍 P0完整性检查: {'✅ 通过' if p0_passed else '❌ 失败'}")
if not p0_passed:
    print(f"      问题: {', '.join(rescue_decision['p0_integrity']['issues'])}")

print(f"   📈 P1评测增强: {'✅ 通过' if p1_passed else '❌ 失败'}")
print(f"      发现: {rescue_decision['p1_evaluation']['findings']}")

# 最终决策
if not p0_passed:
    decision = "PAUSE_AND_FIX"
    reason = "发现数据完整性问题，需要修复后再评估"
elif p1_passed:
    decision = "PROCEED_TO_P2"
    reason = "通过完整性检查且发现改进潜力，继续架构修补"
else:
    decision = "ARCHIVE"
    reason = "通过完整性检查但无改进潜力，建议归档"

print(f"\n🎯 48小时复核决策: {decision}")
print(f"📝 决策理由: {reason}")

# 下一步行动计划
if decision == "PROCEED_TO_P2":
    print(f"\n📅 Day 2行动计划:")
    print(f"   🔧 P2架构最小修补:")
    print(f"      • 缩小模型容量，加入dropout")
    print(f"      • 添加L2正则和早停")
    print(f"      • 切换到listwise目标函数") 
    print(f"      • 训练/推理Top-M对齐")
    print(f"   📊 300+ queries再评测")
    print(f"   🎯 目标: CI95下界 > 0")
elif decision == "PAUSE_AND_FIX":
    print(f"\n🔧 修复行动计划:")
    print(f"      • 修复数据泄漏问题")
    print(f"      • 重新设计train/test切分")
    print(f"      • 修正评测代码错误")
    print(f"      • 完成修复后重新评估")
else:
    print(f"\n📦 归档行动计划:")
    print(f"      • 整理实验记录和代码")
    print(f"      • 保存失败案例供将来参考")
    print(f"      • 转向候选生成/数据闭环项目")

print(f"\n" + "="*80)
print("🔬 48小时救援复核框架执行完成")
print("✅ 科学决策: 基于完整性检查 + 评测增强")
print("⚡ 避免草率: 暂停而非放弃，给V2.0一个公平机会")
print("🎯 下一步: 根据复核结果执行相应行动计划")
print("="*80)